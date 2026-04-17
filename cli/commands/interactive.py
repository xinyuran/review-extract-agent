"""
interactive 命令 — 交互式 REPL 模式
"""

import sys
from typing import Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.table import Table

from ..config_loader import load_config, DEFAULT_OUTPUT_DIR
from ..session import CLISession
from ..formatter import (
    format_single_result,
    format_full_json,
    format_batch_table,
    create_batch_progress,
)
from ..file_reader import read_comments_from_file, FileReadError

console = Console()

HELP_TEXT = """\
[bold cyan]Extract Agent 交互模式[/bold cyan]

  直接输入评论文本即可分析。

[bold]内置命令:[/bold]
  [cyan]/mode fast|agent[/cyan]   切换分析模式
  [cyan]/full on|off[/cyan]       切换完整 JSON 输出
  [cyan]/file <path>[/cyan]       从文件加载评论批量分析
  [cyan]/history[/cyan]           查看当前 session 的分析历史
  [cyan]/session[/cyan]           显示当前 session 信息
  [cyan]/resume[/cyan]            恢复之前保存的 session 继续对话
  [cyan]/help[/cyan]              显示此帮助
  [cyan]/exit[/cyan]              退出（也可按 Ctrl+D）
"""


def interactive(
    mode: str = typer.Option(
        "agent",
        "--mode",
        "-m",
        help="初始分析模式: fast / agent（默认）",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="初始启用完整 JSON 输出",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="指定配置文件路径",
    ),
) -> None:
    """启动交互式 REPL 分析会话。"""

    if mode not in ("fast", "agent"):
        console.print(f"[red]错误: 无效的模式 '{mode}'[/red]")
        raise typer.Exit(code=1)

    config = load_config(config_path)

    console.print()
    console.print("[bold cyan]Extract Agent 交互模式[/bold cyan]")
    console.print(f"[dim]模式: {mode} | 完整输出: {'开' if full else '关'}[/dim]")
    console.print("[dim]输入评论文本开始分析，输入 /resume 恢复会话，/help 查看帮助，/exit 退出[/dim]")
    console.print()

    prompt_session = PromptSession(history=InMemoryHistory())

    ctx = _ReplContext(config=config, mode=mode, full=full)

    try:
        session = _repl_loop(prompt_session, ctx)
    except KeyboardInterrupt:
        console.print("\n[dim]已中断[/dim]")
        session = ctx.session
    finally:
        if session is not None:
            session.close()
            console.print(
                f"\n[dim]Session {session.session_id} 已关闭，"
                f"共分析 {session.result_counter} 条评论[/dim]"
            )
        else:
            console.print("\n[dim]未创建任何 Session，已退出[/dim]")


class _ReplContext:
    """延迟创建 session 的上下文：进入交互模式时不立即创建 session，
    等用户第一次分析或 /resume 时再决定。"""

    def __init__(self, config, mode: str, full: bool):
        self.config = config
        self.mode = mode
        self.full = full
        self.session: Optional[CLISession] = None

    def ensure_session(self) -> CLISession:
        """如果尚未创建 session，立即创建一个新的。"""
        if self.session is None:
            output_dir = getattr(self.config, "_cli_output_dir", DEFAULT_OUTPUT_DIR)
            self.session = CLISession(
                config=self.config,
                mode=self.mode,
                full_output=self.full,
                output_root=output_dir,
                cli_command="interactive",
            )
            console.print(
                f"[dim]已创建新 Session: {self.session.session_id}[/dim]"
            )
        return self.session

    def set_session(self, session: CLISession) -> None:
        """设置 session（用于 /resume 恢复）。"""
        self.session = session
        self.mode = session.mode
        self.full = session.full_output


def _repl_loop(
    prompt_session: PromptSession, ctx: _ReplContext
) -> Optional[CLISession]:
    """REPL 主循环。返回最终活跃的 session（可能为 None 如果用户直接退出）。"""
    while True:
        try:
            user_input = prompt_session.prompt(">> ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print()
            continue

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd_result = _handle_command(user_input, ctx)
            if cmd_result is True:
                break
            continue
        else:
            session = ctx.ensure_session()
            _handle_analyze(user_input, session)

    return ctx.session


def _handle_command(command: str, ctx: _ReplContext):
    """
    处理 REPL 内置命令。
    返回 True 表示退出，其余返回 False 继续循环。
    """
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/exit" or cmd == "/quit":
        return True

    elif cmd == "/help":
        console.print(HELP_TEXT)

    elif cmd == "/mode":
        _cmd_mode(arg, ctx)

    elif cmd == "/full":
        _cmd_full(arg, ctx)

    elif cmd == "/file":
        session = ctx.ensure_session()
        _cmd_file(arg, session)

    elif cmd == "/history":
        if ctx.session is None:
            console.print("[dim]尚未创建 Session，暂无分析历史[/dim]")
        else:
            _cmd_history(ctx.session)

    elif cmd == "/session":
        if ctx.session is None:
            console.print("[dim]尚未创建 Session[/dim]")
            console.print(f"[dim]模式: {ctx.mode} | 完整输出: {'开' if ctx.full else '关'}[/dim]")
        else:
            _cmd_session_info(ctx.session)

    elif cmd == "/resume":
        new_session = _cmd_resume(arg, ctx)
        if new_session is not None:
            if ctx.session is not None:
                ctx.session.close()
            ctx.set_session(new_session)
            console.print(
                f"[bold green]已恢复 Session {new_session.session_id}[/bold green] "
                f"(历史 {new_session.result_counter} 条)"
            )
            console.print(f"[dim]模式: {new_session.mode} | "
                          f"完整输出: {'开' if new_session.full_output else '关'}[/dim]")

    else:
        console.print(f"[yellow]未知命令: {cmd}，输入 /help 查看帮助[/yellow]")

    return False


def _handle_analyze(text: str, session: CLISession) -> None:
    """分析用户输入的评论文本"""
    try:
        with console.status("[bold cyan]正在分析...[/bold cyan]"):
            result = session.analyze(text)

        if session.full_output:
            format_full_json(result)
            result_path = session.get_result_path()
            if result_path:
                console.print(f"[green]已保存:[/green] {result_path}")
        else:
            format_single_result(result, text)
    except Exception as e:
        console.print(f"[red]分析失败: {e}[/red]")


def _cmd_mode(arg: str, ctx: _ReplContext) -> None:
    if arg not in ("fast", "agent"):
        console.print("[yellow]用法: /mode fast|agent[/yellow]")
        return
    ctx.mode = arg
    if ctx.session is not None:
        ctx.session.mode = arg
    console.print(f"[green]已切换到 {arg} 模式[/green]")


def _cmd_full(arg: str, ctx: _ReplContext) -> None:
    if arg.lower() in ("on", "true", "1"):
        ctx.full = True
        if ctx.session is not None:
            ctx.session.full_output = True
        console.print("[green]完整 JSON 输出已开启[/green]")
    elif arg.lower() in ("off", "false", "0"):
        ctx.full = False
        if ctx.session is not None:
            ctx.session.full_output = False
        console.print("[green]完整 JSON 输出已关闭[/green]")
    else:
        console.print("[yellow]用法: /full on|off[/yellow]")


def _cmd_file(arg: str, session: CLISession) -> None:
    if not arg:
        console.print("[yellow]用法: /file <路径> [--column 列名] [--field 字段名][/yellow]")
        return

    file_parts = arg.split()
    file_path = file_parts[0]
    column = None
    field = None

    i = 1
    while i < len(file_parts):
        if file_parts[i] == "--column" and i + 1 < len(file_parts):
            column = file_parts[i + 1]
            i += 2
        elif file_parts[i] == "--field" and i + 1 < len(file_parts):
            field = file_parts[i + 1]
            i += 2
        else:
            i += 1

    try:
        texts = read_comments_from_file(file_path, column=column, field=field)
    except FileReadError as e:
        console.print(f"[red]文件读取错误: {e}[/red]")
        return

    console.print(f"[bold]从文件读取了 {len(texts)} 条评论[/bold]")

    results = []
    progress = create_batch_progress()
    with progress:
        task = progress.add_task("分析进度", total=len(texts))
        for text in texts:
            try:
                result = session.analyze(text)
                results.append(result)
            except Exception as e:
                console.print(f"[red]分析失败: {e}[/red]")
                results.append({"analysis_complete": False, "elapsed_ms": 0})
            progress.advance(task)

    format_batch_table(results, texts)


def _cmd_history(session: CLISession) -> None:
    summary = session.get_history_summary()
    if not summary:
        console.print("[dim]暂无分析历史[/dim]")
        return

    table = Table(title="分析历史", show_header=True)
    table.add_column("#", width=4, justify="right")
    table.add_column("评论", max_width=30)
    table.add_column("关键词", max_width=30)
    table.add_column("情感", width=10)
    table.add_column("耗时", width=8, justify="right")

    for item in summary:
        elapsed = item["elapsed_ms"]
        elapsed_str = f"{elapsed / 1000:.1f}s" if elapsed >= 1000 else f"{elapsed:.0f}ms"
        table.add_row(
            str(item["index"]),
            item["text_preview"],
            item["keywords"],
            item["sentiment"],
            elapsed_str,
        )

    console.print()
    console.print(table)
    console.print()


def _cmd_resume(arg: str, ctx: _ReplContext) -> Optional[CLISession]:
    """
    /resume 命令：列出已保存的 session 并恢复指定的 session。

    用法：
      /resume           列出最近 10 个 session 供选择
      /resume <id>      直接恢复指定 session id
    """
    from pathlib import Path

    output_root = getattr(ctx.config, "_cli_output_dir", DEFAULT_OUTPUT_DIR)
    saved = CLISession.list_saved_sessions(output_root)

    if not saved:
        console.print("[dim]没有已保存的 session[/dim]")
        return None

    if arg:
        matched = [s for s in saved if s["session_id"] == arg]
        if not matched:
            matched = [s for s in saved if arg in s["session_id"]]
        if not matched:
            console.print(f"[red]未找到匹配的 session: {arg}[/red]")
            return None
        target = matched[0]
    else:
        display_sessions = saved[:10]
        table = Table(title="可恢复的 Session", show_header=True)
        table.add_column("#", width=3, justify="right")
        table.add_column("Session ID", width=10)
        table.add_column("创建时间", width=20)
        table.add_column("模式", width=6)
        table.add_column("分析条数", width=8, justify="right")
        table.add_column("历史摘要", max_width=40)

        for i, s in enumerate(display_sessions, 1):
            created = s.get("created_at", "")[:19]
            total = s.get("total_analyzed", 0)
            summaries = s.get("history_summary", [])
            preview = ""
            if summaries:
                first = summaries[0].get("text_preview", "")
                preview = first[:35] + ("..." if len(first) > 35 else "")
            table.add_row(
                str(i), s["session_id"], created,
                s.get("mode", "?"), str(total), preview,
            )

        console.print()
        console.print(table)
        console.print()
        console.print("[dim]输入序号或 session ID 选择，按 Enter 取消[/dim]")

        try:
            from prompt_toolkit import prompt as pt_prompt
            choice = pt_prompt("选择> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not choice:
            return None

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(display_sessions):
                target = display_sessions[idx]
            else:
                console.print("[red]无效的序号[/red]")
                return None
        else:
            matched = [s for s in saved if choice in s["session_id"]]
            if not matched:
                console.print(f"[red]未找到匹配的 session: {choice}[/red]")
                return None
            target = matched[0]

    session_dir = Path(target["_dir"])
    try:
        new_session = CLISession.resume_from(session_dir, ctx.config)
        return new_session
    except Exception as e:
        console.print(f"[red]恢复 session 失败: {e}[/red]")
        return None


def _cmd_session_info(session: CLISession) -> None:
    info = session.get_session_info()
    console.print()
    console.print("[bold]Session 信息:[/bold]")
    for key, value in info.items():
        console.print(f"  [cyan]{key}:[/cyan] {value}")
    console.print()

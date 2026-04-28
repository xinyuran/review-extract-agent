"""
analyze 命令 — 单次/批量分析评论文本
"""

import logging
from typing import Optional

import typer

from ..config_loader import load_config, DEFAULT_OUTPUT_DIR
from ..session import CLISession
from ..formatter import (
    console,
    format_single_result,
    format_full_json,
    format_batch_table,
    create_batch_progress,
)
from ..file_reader import read_comments_from_file, FileReadError


def analyze(
    text: Optional[str] = typer.Argument(
        None,
        help="要分析的评论文本（直接传入字符串）",
    ),
    file: Optional[str] = typer.Option(
        None,
        "--file",
        "-f",
        help="从文件读取评论（支持 .txt / .csv / .json）",
    ),
    column: Optional[str] = typer.Option(
        None,
        "--column",
        help="CSV 文件中评论所在的列名",
    ),
    field: Optional[str] = typer.Option(
        None,
        "--field",
        help="JSON 文件中评论字段名",
    ),
    mode: str = typer.Option(
        "agent",
        "--mode",
        "-m",
        help="分析模式: fast（快速管线）/ agent（Agent 自主规划，默认）",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="输出完整 JSON 结果并持久化到文件",
    ),
    no_reflect: bool = typer.Option(
        False,
        "--no-reflect",
        help="禁用反思机制",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="指定配置文件路径",
    ),
    reviewer_id: Optional[str] = typer.Option(
        None,
        "--reviewer-id",
        help="评论者 ID（启用知识积累追踪）",
    ),
    product_id: Optional[str] = typer.Option(
        None,
        "--product-id",
        help="商品 ID（启用知识积累追踪）",
    ),
    product_name: Optional[str] = typer.Option(
        None,
        "--product-name",
        help="商品名称（与 --product-id 配合使用）",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="启用调试日志",
    ),
) -> None:
    """分析评论文本，输出关键词和情感。支持直接输入文本或从文件读取。"""

    if text is None and file is None:
        console.print("[red]错误: 请提供评论文本或使用 -f 指定输入文件[/red]")
        raise typer.Exit(code=1)

    if mode not in ("fast", "agent"):
        console.print(f"[red]错误: 无效的模式 '{mode}'，可选: fast / agent[/red]")
        raise typer.Exit(code=1)

    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    config = load_config(config_path)

    backend_label = config.get_backend_label()
    backend_mode = config.get_backend_mode()
    console.print(f"[dim]后端: {backend_label}[/dim]")

    if backend_mode == "offline":
        mode = "fast"

    if no_reflect:
        config.ENABLE_REFLECTION = False
    if debug:
        config.DEBUG = True

    output_dir = getattr(config, "_cli_output_dir", DEFAULT_OUTPUT_DIR)

    knowledge_ctx = {
        "reviewer_id": reviewer_id,
        "product_id": product_id,
        "product_name": product_name or "",
    }

    if file:
        _analyze_from_file(file, column, field, mode, full, config, output_dir, knowledge_ctx)
    else:
        _analyze_single_text(text, mode, full, config, output_dir, knowledge_ctx)


def _analyze_single_text(
    text: str,
    mode: str,
    full: bool,
    config,
    output_dir: str,
    knowledge_ctx: dict | None = None,
) -> None:
    """分析单条文本"""
    cli_cmd = f"analyze \"{text[:20]}...\" --mode {mode}" + (" --full" if full else "")
    session = CLISession(
        config=config, mode=mode, full_output=full,
        output_root=output_dir, cli_command=cli_cmd,
    )

    try:
        with console.status("[bold cyan]正在分析评论...[/bold cyan]"):
            result = session.analyze(text, **(knowledge_ctx or {}))

        if full:
            format_full_json(result)
        else:
            format_single_result(result, text)

        console.print(f"[dim]结果已保存: {session.output_dir}[/dim]")
    finally:
        session.close()


def _analyze_from_file(
    file: str,
    column: Optional[str],
    field: Optional[str],
    mode: str,
    full: bool,
    config,
    output_dir: str,
    knowledge_ctx: dict | None = None,
) -> None:
    """从文件读取并批量分析"""
    try:
        texts = read_comments_from_file(file, column=column, field=field)
    except FileReadError as e:
        console.print(f"[red]文件读取错误: {e}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]从文件读取了 {len(texts)} 条评论[/bold]")
    console.print()

    cli_cmd = f"analyze -f {file} --mode {mode}" + (" --full" if full else "")
    session = CLISession(
        config=config, mode=mode, full_output=full,
        output_root=output_dir, cli_command=cli_cmd,
    )

    results = []
    try:
        progress = create_batch_progress()
        with progress:
            task = progress.add_task("分析进度", total=len(texts))
            for text in texts:
                result = session.analyze(text, **(knowledge_ctx or {}))
                results.append(result)
                progress.advance(task)

        format_batch_table(results, texts)
        console.print(f"[dim]结果已保存: {session.output_dir}[/dim]")
    finally:
        session.close()

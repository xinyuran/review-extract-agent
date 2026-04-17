"""
终端输出格式化 — 使用 rich 美化显示分析结果
"""

import json
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

console = Console()


def format_single_result(result: Dict[str, Any], comment: str) -> None:
    """在终端输出单条评论的简化分析结果"""

    console.print()
    console.rule(style="bold cyan")
    console.print(f"[bold]评论:[/bold] {comment}")
    console.rule(style="bold cyan")
    console.print()

    keywords = result.get("keywords", [])
    if keywords:
        console.print("[bold green]关键词:[/bold green]")
        for i, kw in enumerate(keywords, 1):
            keyword = kw.get("keyword", "")
            score = kw.get("score", 0)
            console.print(f"   {i}. {keyword} [dim](score: {score})[/dim]")
    else:
        console.print("[bold green]关键词:[/bold green] [dim](无)[/dim]")

    console.print()

    sentiment = result.get("sentiment", {})
    label = sentiment.get("label", "unknown")
    confidence = sentiment.get("confidence", 0)
    label_color = {
        "positive": "green",
        "negative": "red",
        "neutral": "yellow",
    }.get(label, "white")
    console.print(
        f"[bold]情感:[/bold] [{label_color}]{label}[/{label_color}] "
        f"[dim](confidence: {confidence})[/dim]"
    )

    console.print()

    tool_errors = result.get("tool_errors", [])
    for te in tool_errors:
        console.print(f"[yellow]⚠ {te}[/yellow]")

    debug_logs = result.get("debug_logs", [])
    for dl in debug_logs:
        console.print(f"[dim]  → debug 日志: {dl}[/dim]")

    if tool_errors or debug_logs:
        console.print()

    warnings = result.get("warnings", [])
    for w in warnings:
        console.print(f"[bold yellow]⚠ {w}[/bold yellow]")
    if warnings:
        console.print()

    errors = [
        t.get("content", "")
        for t in result.get("agent_trace", [])
        if t.get("type") == "error"
    ]
    for err in errors:
        console.print(f"[bold red]✗ {err}[/bold red]")
    if errors:
        console.print()

    elapsed = result.get("elapsed_ms", 0)
    mode = result.get("mode", "unknown")
    steps = result.get("steps", "-")
    console.print(
        f"[dim]耗时: {elapsed}ms | 模式: {mode} | 步数: {steps}[/dim]"
    )
    console.rule(style="bold cyan")
    console.print()


def format_full_json(result: Dict[str, Any]) -> None:
    """在终端输出完整的 JSON 结果"""

    console.print()
    console.print_json(json.dumps(result, ensure_ascii=False))
    console.print()


def format_batch_table(results: List[Dict[str, Any]], texts: List[str]) -> None:
    """在终端输出批量分析结果的汇总表格"""

    table = Table(title="批量分析结果", show_header=True, show_lines=True)
    table.add_column("#", style="bold", width=4, justify="right")
    table.add_column("评论 (截断)", max_width=30)
    table.add_column("关键词", max_width=35)
    table.add_column("情感", width=10, justify="center")
    table.add_column("耗时", width=8, justify="right")

    success_count = 0
    fail_count = 0

    for i, (result, text) in enumerate(zip(results, texts), 1):
        text_preview = text[:25] + ("..." if len(text) > 25 else "")

        keywords = result.get("keywords", [])
        kw_str = ", ".join(k.get("keyword", "") for k in keywords[:5])
        if not kw_str:
            kw_str = "(无)"

        sentiment = result.get("sentiment", {})
        label = sentiment.get("label", "unknown")
        label_color = {
            "positive": "green",
            "negative": "red",
            "neutral": "yellow",
        }.get(label, "white")

        elapsed = result.get("elapsed_ms", 0)
        elapsed_str = f"{elapsed / 1000:.1f}s" if elapsed >= 1000 else f"{elapsed:.0f}ms"

        if result.get("analysis_complete"):
            success_count += 1
        else:
            fail_count += 1

        table.add_row(
            str(i),
            text_preview,
            kw_str,
            f"[{label_color}]{label}[/{label_color}]",
            elapsed_str,
        )

    console.print()
    console.print(table)

    total_elapsed = sum(r.get("elapsed_ms", 0) for r in results)
    total_elapsed_str = f"{total_elapsed / 1000:.1f}s"
    console.print()
    console.print(
        f"[bold]总计:[/bold] {len(results)} 条 | "
        f"[green]成功: {success_count}[/green] | "
        f"[red]失败: {fail_count}[/red] | "
        f"总耗时: {total_elapsed_str}"
    )
    console.print()


def create_batch_progress() -> Progress:
    """创建批量分析的进度条"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )

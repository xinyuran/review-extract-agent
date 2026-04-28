"""
export 子命令 — 导出 SFT 训练数据

从轨迹采集目录读取 JSONL 数据，转换为指定格式导出。
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()

export_app = typer.Typer(
    name="export",
    help="导出 SFT 训练数据",
    no_args_is_help=True,
)


def _get_trajectory_dir(input_dir: Optional[str] = None) -> str:
    if input_dir:
        return input_dir
    try:
        from ..config_loader import load_config
        cfg = load_config()
        return cfg.TRAJECTORY_OUTPUT_DIR
    except Exception:
        return "extract_agent_output/trajectory"


@export_app.command(name="sft")
def export_sft(
    format: str = typer.Option(
        "openai",
        "--format", "-f",
        help="导出格式: openai (Agent SFT) | tool-sft (Tool LLM SFT) | tool-supervision (工具监督三元组)",
    ),
    input_dir: Optional[str] = typer.Option(
        None,
        "--input", "-i",
        help="轨迹数据目录（默认使用配置中的 TRAJECTORY_OUTPUT_DIR）",
    ),
    output: str = typer.Option(
        ...,
        "--output", "-o",
        help="输出文件路径 (.jsonl)",
    ),
    session: Optional[str] = typer.Option(
        None,
        "--session", "-s",
        help="指定 session ID（默认导出全部）",
    ),
) -> None:
    """导出 SFT 训练数据"""
    from ...trajectory import TrajectoryExporter

    traj_dir = _get_trajectory_dir(input_dir)
    exporter = TrajectoryExporter(traj_dir)

    sessions = [session] if session else None

    if format == "openai":
        count = exporter.export_agent_sft(output, session_ids=sessions)
        console.print(f"[green]导出 Agent SFT 数据: {count} 条 → {output}[/green]")
    elif format == "tool-sft":
        count = exporter.export_tool_sft(output, session_ids=sessions)
        console.print(f"[green]导出 Tool SFT 数据: {count} 条 → {output}[/green]")
    elif format == "tool-supervision":
        count = exporter.export_tool_supervision(output, session_ids=sessions)
        console.print(f"[green]导出工具监督数据: {count} 条 → {output}[/green]")
    else:
        console.print(
            f"[red]未知格式: {format}[/red]\n"
            "可用格式: openai, tool-sft, tool-supervision"
        )
        raise typer.Exit(1)


@export_app.command(name="stats")
def export_stats(
    input_dir: Optional[str] = typer.Option(
        None,
        "--input", "-i",
        help="轨迹数据目录",
    ),
) -> None:
    """统计已采集轨迹的数量和分布"""
    from ...trajectory import TrajectoryExporter

    traj_dir = _get_trajectory_dir(input_dir)
    exporter = TrajectoryExporter(traj_dir)

    stats = exporter.get_stats()

    if stats["total_sessions"] == 0:
        console.print("[yellow]未找到任何轨迹数据[/yellow]")
        console.print(f"轨迹目录: {traj_dir}")
        return

    console.print(f"\n[bold]轨迹数据统计[/bold]  (目录: {traj_dir})\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("指标", style="dim")
    table.add_column("数量", justify="right")

    table.add_row("Session 总数", str(stats["total_sessions"]))
    table.add_row("Agent LLM 调用次数", str(stats["total_agent_turns"]))
    table.add_row("Tool LLM 调用次数", str(stats["total_tool_turns"]))

    console.print(table)

    if stats["skill_distribution"]:
        console.print("\n[bold]Skill 分布:[/bold]")
        skill_table = Table(show_header=True, header_style="bold")
        skill_table.add_column("Skill 名称")
        skill_table.add_column("调用次数", justify="right")

        for skill, count in sorted(
            stats["skill_distribution"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            skill_table.add_row(skill, str(count))

        console.print(skill_table)

    sessions = exporter.list_sessions()
    if sessions:
        console.print(f"\n[dim]可用 sessions ({len(sessions)}): {', '.join(sessions[:10])}"
                      f"{'...' if len(sessions) > 10 else ''}[/dim]")

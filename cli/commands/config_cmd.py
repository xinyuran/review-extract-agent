"""
config 命令 — 查看/初始化配置文件
"""

from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from ..config_loader import (
    DEFAULT_OUTPUT_DIR,
    find_config_file,
    init_config_file,
    load_config,
    get_default_yaml,
    USER_CONFIG_FILE,
)

console = Console()


def config_command(
    init: bool = typer.Option(
        False,
        "--init",
        help="在用户目录初始化默认配置文件",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help="显示当前生效的配置",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="指定配置文件路径",
    ),
) -> None:
    """查看或初始化配置文件。"""

    if init:
        _handle_init()
        return

    if show:
        _handle_show(config_path)
        return

    console.print("[yellow]请使用 --init 或 --show 参数。使用 --help 查看帮助。[/yellow]")


def _handle_init() -> None:
    if USER_CONFIG_FILE.is_file():
        console.print(
            f"[yellow]配置文件已存在: {USER_CONFIG_FILE}[/yellow]"
        )
        overwrite = typer.confirm("是否覆盖？", default=False)
        if not overwrite:
            console.print("[dim]已取消[/dim]")
            return

    path = init_config_file()
    console.print(f"[green]配置文件已创建: {path}[/green]")
    console.print()
    syntax = Syntax(
        get_default_yaml(), "yaml", theme="monokai", line_numbers=True
    )
    console.print(syntax)


def _handle_show(config_path: Optional[str]) -> None:
    config_file = find_config_file(config_path)
    config = load_config(config_path)

    console.print()
    if config_file:
        console.print(f"[bold green]配置来源:[/bold green] {config_file}")
    else:
        console.print("[bold yellow]配置来源:[/bold yellow] 使用默认配置（未找到配置文件）")
    console.print()

    table = Table(title="当前生效配置", show_header=True)
    table.add_column("配置项", style="cyan")
    table.add_column("值", style="white")

    sections = [
        ("Agent LLM", [
            ("base_url", config.AGENT_LLM_BASE_URL),
            ("api_key", config.AGENT_LLM_API_KEY[:8] + "***" if len(config.AGENT_LLM_API_KEY) > 8 else config.AGENT_LLM_API_KEY),
            ("model", config.AGENT_LLM_MODEL),
            ("temperature", config.AGENT_LLM_TEMPERATURE),
            ("max_tokens", config.AGENT_LLM_MAX_TOKENS),
        ]),
        ("Tool LLM", [
            ("base_url", config.TOOL_LLM_BASE_URL),
            ("api_key", config.TOOL_LLM_API_KEY[:8] + "***" if len(config.TOOL_LLM_API_KEY) > 8 else config.TOOL_LLM_API_KEY),
            ("model", config.TOOL_LLM_MODEL),
            ("temperature", config.TOOL_LLM_TEMPERATURE),
            ("max_tokens", config.TOOL_LLM_MAX_TOKENS),
        ]),
        ("Agent 控制", [
            ("tool_calling_mode", config.AGENT_TOOL_CALLING_MODE),
            ("max_steps", config.AGENT_MAX_STEPS),
            ("timeout", config.AGENT_TIMEOUT),
            ("tool_timeout", config.TOOL_TIMEOUT),
        ]),
        ("反思器", [
            ("enabled", config.ENABLE_REFLECTION),
            ("max_rounds", config.REFLECTION_MAX_ROUNDS),
            ("score_threshold", config.REFLECTION_SCORE_THRESHOLD),
        ]),
        ("CLI", [
            ("output_dir", getattr(config, "_cli_output_dir", DEFAULT_OUTPUT_DIR)),
            ("default_mode", getattr(config, "_cli_default_mode", "agent")),
        ]),
    ]

    for section_name, items in sections:
        table.add_row(f"[bold]{section_name}[/bold]", "")
        for key, value in items:
            table.add_row(f"  {key}", str(value))

    console.print(table)

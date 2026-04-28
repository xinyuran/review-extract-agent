"""
config 命令 — 查看/初始化配置文件
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from ..config_loader import (
    DEFAULT_OUTPUT_DIR,
    _PACKAGE_ROOT,
    find_config_file,
    init_config_file,
    load_config,
    get_default_yaml,
    get_default_env,
    USER_CONFIG_FILE,
)

console = Console()


def config_command(
    init: bool = typer.Option(
        False,
        "--init",
        help="在用户目录初始化默认配置文件（YAML + .env）",
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
    env_target = _PACKAGE_ROOT / ".env"
    created_files = []

    if env_target.is_file():
        console.print(f"[yellow].env 文件已存在: {env_target}[/yellow]")
        if typer.confirm("是否覆盖 .env？", default=False):
            env_target.write_text(get_default_env(), encoding="utf-8")
            created_files.append(str(env_target))
    else:
        env_target.write_text(get_default_env(), encoding="utf-8")
        created_files.append(str(env_target))

    if USER_CONFIG_FILE.is_file():
        console.print(f"[yellow]YAML 配置文件已存在: {USER_CONFIG_FILE}[/yellow]")
        if typer.confirm("是否覆盖 YAML？", default=False):
            path = init_config_file()
            created_files.append(str(path))
    else:
        path = init_config_file()
        created_files.append(str(path))

    if created_files:
        for f in created_files:
            console.print(f"[green]已创建: {f}[/green]")
        console.print()
        console.print("[bold]推荐使用 .env 文件管理配置：[/bold]")
        console.print(f"[dim]  编辑 {env_target} 即可修改 API 地址、密钥等参数[/dim]")
    else:
        console.print("[dim]未创建任何文件[/dim]")


def _handle_show(config_path: Optional[str]) -> None:
    config = load_config(config_path)

    console.print()
    console.print(f"[bold green]配置来源:[/bold green] {config._config_source}")
    env_file = getattr(config, "_env_file", None)
    if env_file:
        console.print(f"[bold green].env 文件:[/bold green] {env_file}")
    console.print(f"[bold green]后端模式:[/bold green] {config.get_backend_label()}")
    console.print()

    table = Table(title="当前生效配置", show_header=True)
    table.add_column("配置项", style="cyan")
    table.add_column("值", style="white")

    sections = [
        ("后端", [
            ("backend_mode", config.get_backend_mode()),
        ]),
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

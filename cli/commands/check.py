"""
check 命令 — 检测配置和各组件连通性
"""

import os
from typing import Optional

import typer
from rich.console import Console

try:
    import httpx
except ImportError:
    httpx = None

from ..config_loader import load_config, find_config_file

console = Console()


def check(
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="指定配置文件路径",
    ),
) -> None:
    """检测配置文件、LLM 连通性和工具加载状态。"""

    config = load_config(config_path)
    config_file = find_config_file(config_path)

    console.print()

    _check_config(config_file)
    _check_llm("Agent LLM", config.AGENT_LLM_BASE_URL, config.AGENT_LLM_MODEL)
    _check_llm("Tool LLM", config.TOOL_LLM_BASE_URL, config.TOOL_LLM_MODEL)
    _check_redis()
    _check_tools()
    _check_fc_mode(config)

    console.print()


def _check_config(config_file) -> None:
    if config_file:
        console.print(f"[green]\\[配置文件]  {config_file}  ✓ 已加载[/green]")
    else:
        console.print("[yellow]\\[配置文件]  未找到配置文件，使用默认值[/yellow]")


def _check_llm(name: str, base_url: str, model: str) -> None:
    """测试 LLM endpoint 连通性"""
    if httpx is None:
        console.print(
            f"[yellow]\\[{name}] 需要安装 httpx 才能检测连通性[/yellow]"
        )
        return

    try:
        url = base_url.rstrip("/") + "/models"
        response = httpx.get(url, timeout=10)
        if response.status_code == 200:
            model_short = model.split("/")[-1] if "/" in model else model
            console.print(
                f"[green]\\[{name}] {base_url}  ✓ 连通 (模型: {model_short})[/green]"
            )
        else:
            console.print(
                f"[red]\\[{name}] {base_url}  ✗ HTTP {response.status_code}[/red]"
            )
    except Exception as e:
        console.print(
            f"[red]\\[{name}] {base_url}  ✗ 不可用 ({e})[/red]"
        )


def _check_redis() -> None:
    """测试 Redis 连通性"""
    try:
        import redis as redis_lib
        redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        client = redis_lib.from_url(redis_url, socket_timeout=5)
        client.ping()
        console.print(
            f"[green]\\[Redis]     {redis_url}  ✓ 连通[/green]"
        )
    except ImportError:
        console.print(
            "[yellow]\\[Redis]     redis 包未安装（异步任务不可用，不影响 CLI）[/yellow]"
        )
    except Exception:
        redis_url_display = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        console.print(
            f"[yellow]\\[Redis]     {redis_url_display}  ✗ 不可用 "
            f"(异步任务不可用，不影响 CLI)[/yellow]"
        )


def _check_tools() -> None:
    """检查工具加载"""
    try:
        from ...tools import ALL_TOOLS
        count = len(ALL_TOOLS)
        console.print(f"[green]\\[工具]      {count} 个工具已加载 ✓[/green]")
    except Exception as e:
        console.print(f"[red]\\[工具]      加载失败: {e}[/red]")


def _check_fc_mode(config) -> None:
    """显示 Function Calling 模式"""
    mode = config.AGENT_TOOL_CALLING_MODE
    console.print(f"[green]\\[Function Calling] {mode} 模式 ✓[/green]")

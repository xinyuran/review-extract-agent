"""
serve 命令 — 启动 FastAPI 服务
"""

import typer
from rich.console import Console

try:
    import uvicorn
except ImportError:
    uvicorn = None

console = Console()


def serve(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="绑定地址",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="监听端口",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="启用热重载（开发模式）",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="工作进程数",
    ),
) -> None:
    """启动 FastAPI HTTP 服务。"""

    if uvicorn is None:
        console.print("[red]错误: 需要安装 uvicorn（pip install uvicorn[standard]）[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold cyan]启动 Extract Agent API 服务[/bold cyan]")
    console.print(f"[dim]地址: http://{host}:{port}[/dim]")
    console.print(f"[dim]热重载: {'是' if reload else '否'} | 工作进程: {workers}[/dim]")
    console.print()

    uvicorn.run(
        "extract_agent.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
    )

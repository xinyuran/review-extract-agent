"""
Typer 应用入口 — 注册所有子命令
"""

import typer

from .commands.analyze import analyze
from .commands.config_cmd import config_command
from .commands.interactive import interactive
from .commands.serve import serve
from .commands.check import check
from .commands.export import export_app
from .commands.report import report_app

app = typer.Typer(
    name="extract-agent",
    help="中文电商评论智能分析 CLI 工具",
    add_completion=False,
    no_args_is_help=True,
)

app.command(
    name="analyze",
    help="单次分析模式 — 分析评论文本并输出关键词与情感",
)(analyze)

app.command(
    name="interactive",
    help="交互式 REPL 模式 — 连续分析评论",
)(interactive)

app.command(
    name="serve",
    help="启动 FastAPI HTTP 服务",
)(serve)

app.command(
    name="check",
    help="检测配置和各组件连通性",
)(check)

app.command(
    name="config",
    help="查看或初始化配置文件",
)(config_command)

app.add_typer(export_app, name="export", help="导出 SFT 训练数据")
app.add_typer(report_app, name="report", help="知识积累分析报告")

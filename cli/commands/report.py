"""
report 子命令 — 生成知识积累分析报告

支持评论者画像、商品画像、全局概览三种报告。
"""

from typing import Optional

import typer
from rich.console import Console

console = Console()

report_app = typer.Typer(
    name="report",
    help="知识积累分析报告",
    no_args_is_help=True,
)


def _get_knowledge_manager():
    from ...knowledge import KnowledgeManager
    try:
        from ..config_loader import load_config
        cfg = load_config()
        return KnowledgeManager(cfg.KNOWLEDGE_STORE_DIR)
    except Exception:
        return KnowledgeManager()


@report_app.command(name="reviewer")
def report_reviewer(
    reviewer_id: str = typer.Argument(..., help="评论者 ID"),
) -> None:
    """生成评论者分析报告"""
    from ...knowledge.reporter import KnowledgeReporter

    km = _get_knowledge_manager()
    reporter = KnowledgeReporter(km)
    reporter.report_reviewer(reviewer_id)


@report_app.command(name="product")
def report_product(
    product_id: str = typer.Argument(..., help="商品 ID"),
) -> None:
    """生成商品分析报告"""
    from ...knowledge.reporter import KnowledgeReporter

    km = _get_knowledge_manager()
    reporter = KnowledgeReporter(km)
    reporter.report_product(product_id)


@report_app.command(name="summary")
def report_summary() -> None:
    """全局统计概览"""
    from ...knowledge.reporter import KnowledgeReporter

    km = _get_knowledge_manager()
    reporter = KnowledgeReporter(km)
    reporter.report_summary()

"""
知识分析报告生成器

从 KnowledgeManager 读取画像数据并生成 Rich 终端表格报告。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .manager import KnowledgeManager
from .models import ReviewerProfile, ProductProfile

console = Console()


class KnowledgeReporter:
    """生成知识积累报告"""

    def __init__(self, knowledge_manager: KnowledgeManager):
        self._km = knowledge_manager

    def report_reviewer(self, reviewer_id: str) -> None:
        profile = self._km.get_reviewer(reviewer_id)
        if profile is None:
            console.print(f"[red]未找到评论者 {reviewer_id} 的画像数据[/red]")
            return

        console.print(Panel(
            f"[bold]评论者画像: {profile.reviewer_id}[/bold]",
            subtitle=f"累计 {profile.total_reviews} 条评论",
        ))

        info_table = Table(show_header=False, box=None)
        info_table.add_column("字段", style="dim")
        info_table.add_column("值")
        info_table.add_row("评论总数", str(profile.total_reviews))
        info_table.add_row("首次出现", profile.first_seen or "未知")
        info_table.add_row("最后更新", profile.last_updated or "未知")
        detail_level = self._km.get_output_detail_level(reviewer_id)
        info_table.add_row("输出详细级别", detail_level)
        console.print(info_table)

        console.print("\n[bold]情感分布:[/bold]")
        sent_table = Table(show_header=True, header_style="bold")
        sent_table.add_column("情感")
        sent_table.add_column("次数", justify="right")
        sent_table.add_column("占比", justify="right")

        total = max(profile.total_reviews, 1)
        for label in ("positive", "negative", "neutral"):
            count = profile.sentiment_distribution.get(label, 0)
            pct = f"{count / total * 100:.1f}%"
            color = {"positive": "green", "negative": "red", "neutral": "yellow"}.get(label, "")
            sent_table.add_row(f"[{color}]{label}[/{color}]", str(count), pct)
        console.print(sent_table)

        if profile.keyword_frequency:
            console.print("\n[bold]关键词频次 Top-20:[/bold]")
            kw_table = Table(show_header=True, header_style="bold")
            kw_table.add_column("关键词")
            kw_table.add_column("频次", justify="right")

            sorted_kws = sorted(
                profile.keyword_frequency.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:20]
            for kw, count in sorted_kws:
                kw_table.add_row(kw, str(count))
            console.print(kw_table)

        if profile.positive_tags:
            console.print(f"\n[green]正向标签:[/green] {', '.join(profile.positive_tags[:15])}")
        if profile.negative_tags:
            console.print(f"[red]负向标签:[/red] {', '.join(profile.negative_tags[:15])}")

    def report_product(self, product_id: str) -> None:
        profile = self._km.get_product(product_id)
        if profile is None:
            console.print(f"[red]未找到商品 {product_id} 的画像数据[/red]")
            return

        title = f"商品画像: {profile.product_id}"
        if profile.product_name:
            title += f" ({profile.product_name})"

        console.print(Panel(
            f"[bold]{title}[/bold]",
            subtitle=f"累计 {profile.total_reviews} 条评论",
        ))

        rates = profile.sentiment_rates
        console.print("\n[bold]情感分布:[/bold]")
        sent_table = Table(show_header=True, header_style="bold")
        sent_table.add_column("情感")
        sent_table.add_column("次数", justify="right")
        sent_table.add_column("占比", justify="right")

        for label in ("positive", "negative", "neutral"):
            count = profile.sentiment_stats.get(f"{label}_count", 0)
            rate = rates.get(f"{label}_rate", 0)
            color = {"positive": "green", "negative": "red", "neutral": "yellow"}.get(label, "")
            sent_table.add_row(
                f"[{color}]{label}[/{color}]",
                str(count),
                f"{rate * 100:.1f}%",
            )
        console.print(sent_table)

        top_kws = profile.top_keywords
        if top_kws:
            console.print("\n[bold]关键词 Top-20:[/bold]")
            kw_table = Table(show_header=True, header_style="bold")
            kw_table.add_column("关键词")
            kw_table.add_column("出现次数", justify="right")
            kw_table.add_column("平均分数", justify="right")

            for item in top_kws:
                kw_table.add_row(
                    item["keyword"],
                    str(item["count"]),
                    str(item["avg_score"]),
                )
            console.print(kw_table)

        if profile.strengths:
            console.print(f"\n[green]优势标签:[/green] {', '.join(profile.strengths)}")
        if profile.common_issues:
            console.print(f"[red]常见问题:[/red] {', '.join(profile.common_issues)}")

    def report_summary(self) -> None:
        stats = self._km.get_global_stats()

        console.print(Panel("[bold]知识积累全局概览[/bold]"))

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("指标", style="dim")
        table.add_column("数量", justify="right")

        table.add_row("评论者总数", str(stats["total_reviewers"]))
        table.add_row("商品总数", str(stats["total_products"]))
        table.add_row("追踪的评论总数", str(stats["total_reviews_tracked"]))
        console.print(table)

        reviewers = self._km.list_reviewers()
        products = self._km.list_products()

        if reviewers:
            console.print(
                f"\n[dim]评论者 ({len(reviewers)}): "
                f"{', '.join(reviewers[:10])}"
                f"{'...' if len(reviewers) > 10 else ''}[/dim]"
            )
        if products:
            console.print(
                f"[dim]商品 ({len(products)}): "
                f"{', '.join(products[:10])}"
                f"{'...' if len(products) > 10 else ''}[/dim]"
            )

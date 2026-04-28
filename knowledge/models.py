"""
知识积累数据模型

定义评论者画像（个体结构体）和商品画像（客体结构体）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ReviewerProfile:
    """
    评论者画像（个体持久化结构体）

    跨序列追踪某个评论者的行为模式：
    - 关键词频次统计
    - 情感分布
    - 正向/负向标签积累
    """

    reviewer_id: str
    total_reviews: int = 0
    keyword_frequency: Dict[str, int] = field(default_factory=dict)
    sentiment_distribution: Dict[str, int] = field(
        default_factory=lambda: {"positive": 0, "negative": 0, "neutral": 0}
    )
    positive_tags: List[str] = field(default_factory=list)
    negative_tags: List[str] = field(default_factory=list)
    first_seen: Optional[str] = None
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "total_reviews": self.total_reviews,
            "keyword_frequency": self.keyword_frequency,
            "sentiment_distribution": self.sentiment_distribution,
            "positive_tags": self.positive_tags,
            "negative_tags": self.negative_tags,
            "first_seen": self.first_seen,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReviewerProfile:
        return cls(
            reviewer_id=data["reviewer_id"],
            total_reviews=data.get("total_reviews", 0),
            keyword_frequency=data.get("keyword_frequency", {}),
            sentiment_distribution=data.get(
                "sentiment_distribution",
                {"positive": 0, "negative": 0, "neutral": 0},
            ),
            positive_tags=data.get("positive_tags", []),
            negative_tags=data.get("negative_tags", []),
            first_seen=data.get("first_seen"),
            last_updated=data.get("last_updated"),
        )


@dataclass
class ProductProfile:
    """
    商品画像（客体全局结构体）

    全局统计某个商品的评论模式：
    - 关键词 Top-N 及频次
    - 情感比例统计
    - 常见问题/优势标签
    - 关键词趋势（按月）
    """

    product_id: str
    product_name: str = ""
    total_reviews: int = 0
    keyword_counts: Dict[str, int] = field(default_factory=dict)
    keyword_scores: Dict[str, float] = field(default_factory=dict)
    sentiment_stats: Dict[str, float] = field(
        default_factory=lambda: {
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
        }
    )
    common_issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    keyword_trend: List[Dict[str, Any]] = field(default_factory=list)
    first_seen: Optional[str] = None
    last_updated: Optional[str] = None

    @property
    def top_keywords(self) -> List[Dict[str, Any]]:
        sorted_kws = sorted(
            self.keyword_counts.items(), key=lambda x: x[1], reverse=True
        )
        return [
            {
                "keyword": kw,
                "count": count,
                "avg_score": round(
                    self.keyword_scores.get(kw, 0) / max(count, 1), 2
                ),
            }
            for kw, count in sorted_kws[:20]
        ]

    @property
    def sentiment_rates(self) -> Dict[str, float]:
        total = max(self.total_reviews, 1)
        return {
            "positive_rate": round(
                self.sentiment_stats.get("positive_count", 0) / total, 3
            ),
            "negative_rate": round(
                self.sentiment_stats.get("negative_count", 0) / total, 3
            ),
            "neutral_rate": round(
                self.sentiment_stats.get("neutral_count", 0) / total, 3
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "total_reviews": self.total_reviews,
            "top_keywords": self.top_keywords,
            "keyword_counts": self.keyword_counts,
            "keyword_scores": self.keyword_scores,
            "sentiment_stats": self.sentiment_stats,
            "sentiment_rates": self.sentiment_rates,
            "common_issues": self.common_issues,
            "strengths": self.strengths,
            "keyword_trend": self.keyword_trend,
            "first_seen": self.first_seen,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProductProfile:
        return cls(
            product_id=data["product_id"],
            product_name=data.get("product_name", ""),
            total_reviews=data.get("total_reviews", 0),
            keyword_counts=data.get("keyword_counts", {}),
            keyword_scores=data.get("keyword_scores", {}),
            sentiment_stats=data.get(
                "sentiment_stats",
                {"positive_count": 0, "negative_count": 0, "neutral_count": 0},
            ),
            common_issues=data.get("common_issues", []),
            strengths=data.get("strengths", []),
            keyword_trend=data.get("keyword_trend", []),
            first_seen=data.get("first_seen"),
            last_updated=data.get("last_updated"),
        )

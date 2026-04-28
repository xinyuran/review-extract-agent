"""
知识积累管理器

管理评论者画像和商品画像的 CRUD 操作，
支持从分析结果中自动更新画像数据。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import ReviewerProfile, ProductProfile

logger = logging.getLogger(__name__)

_SENTIMENT_LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "正向": "positive",
    "负向": "negative",
    "中立": "neutral",
}

_NEGATIVE_SENTIMENT_KEYWORDS = {"差", "不好", "失望", "垃圾", "不满", "难用", "慢", "坏"}
_POSITIVE_SENTIMENT_KEYWORDS = {"好", "不错", "满意", "快", "赞", "喜欢", "推荐", "性价比"}


class KnowledgeManager:
    """
    知识积累管理器

    按 knowledge_store/{reviewers,products}/{id}.json 组织目录结构。
    """

    def __init__(self, store_dir: str = "extract_agent_output/knowledge_store"):
        self._store_dir = Path(store_dir)
        self._reviewers_dir = self._store_dir / "reviewers"
        self._products_dir = self._store_dir / "products"

        self._reviewers_dir.mkdir(parents=True, exist_ok=True)
        self._products_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 评论者画像 (Reviewer Profile)
    # ------------------------------------------------------------------

    def get_reviewer(self, reviewer_id: str) -> Optional[ReviewerProfile]:
        path = self._reviewers_dir / f"{reviewer_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ReviewerProfile.from_dict(data)
        except Exception as e:
            logger.warning("加载评论者画像失败 %s: %s", reviewer_id, e)
            return None

    def save_reviewer(self, profile: ReviewerProfile) -> None:
        path = self._reviewers_dir / f"{profile.reviewer_id}.json"
        path.write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def update_reviewer(
        self, reviewer_id: str, analysis_result: Dict[str, Any]
    ) -> ReviewerProfile:
        """从分析结果更新评论者画像"""
        profile = self.get_reviewer(reviewer_id) or ReviewerProfile(
            reviewer_id=reviewer_id,
            first_seen=datetime.now().isoformat(),
        )

        profile.total_reviews += 1
        profile.last_updated = datetime.now().isoformat()

        keywords = analysis_result.get("keywords", [])
        for kw_item in keywords:
            kw = kw_item.get("keyword", "") if isinstance(kw_item, dict) else ""
            if kw:
                profile.keyword_frequency[kw] = (
                    profile.keyword_frequency.get(kw, 0) + 1
                )

        sentiment = analysis_result.get("sentiment", {})
        if isinstance(sentiment, dict):
            label = _SENTIMENT_LABEL_MAP.get(
                sentiment.get("label", ""), "neutral"
            )
            profile.sentiment_distribution[label] = (
                profile.sentiment_distribution.get(label, 0) + 1
            )

            self._update_reviewer_tags(profile, keywords, label)

        self.save_reviewer(profile)
        return profile

    @staticmethod
    def _update_reviewer_tags(
        profile: ReviewerProfile,
        keywords: List[Dict[str, Any]],
        sentiment_label: str,
    ) -> None:
        """根据情感标签和关键词更新正向/负向标签"""
        for kw_item in keywords:
            kw = kw_item.get("keyword", "") if isinstance(kw_item, dict) else ""
            if not kw:
                continue

            if sentiment_label == "negative" or kw in _NEGATIVE_SENTIMENT_KEYWORDS:
                if kw not in profile.negative_tags:
                    profile.negative_tags.append(kw)
                    if len(profile.negative_tags) > 50:
                        profile.negative_tags = profile.negative_tags[-50:]
            elif sentiment_label == "positive" or kw in _POSITIVE_SENTIMENT_KEYWORDS:
                if kw not in profile.positive_tags:
                    profile.positive_tags.append(kw)
                    if len(profile.positive_tags) > 50:
                        profile.positive_tags = profile.positive_tags[-50:]

    # ------------------------------------------------------------------
    # 商品画像 (Product Profile)
    # ------------------------------------------------------------------

    def get_product(self, product_id: str) -> Optional[ProductProfile]:
        path = self._products_dir / f"{product_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ProductProfile.from_dict(data)
        except Exception as e:
            logger.warning("加载商品画像失败 %s: %s", product_id, e)
            return None

    def save_product(self, profile: ProductProfile) -> None:
        path = self._products_dir / f"{profile.product_id}.json"
        path.write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def update_product(
        self,
        product_id: str,
        analysis_result: Dict[str, Any],
        product_name: str = "",
    ) -> ProductProfile:
        """从分析结果更新商品画像"""
        profile = self.get_product(product_id) or ProductProfile(
            product_id=product_id,
            product_name=product_name,
            first_seen=datetime.now().isoformat(),
        )

        if product_name and not profile.product_name:
            profile.product_name = product_name

        profile.total_reviews += 1
        profile.last_updated = datetime.now().isoformat()

        keywords = analysis_result.get("keywords", [])
        for kw_item in keywords:
            kw = kw_item.get("keyword", "") if isinstance(kw_item, dict) else ""
            score = kw_item.get("score", 0.5) if isinstance(kw_item, dict) else 0.5
            if kw:
                profile.keyword_counts[kw] = profile.keyword_counts.get(kw, 0) + 1
                profile.keyword_scores[kw] = (
                    profile.keyword_scores.get(kw, 0) + float(score)
                )

        sentiment = analysis_result.get("sentiment", {})
        if isinstance(sentiment, dict):
            label = _SENTIMENT_LABEL_MAP.get(
                sentiment.get("label", ""), "neutral"
            )
            count_key = f"{label}_count"
            profile.sentiment_stats[count_key] = (
                profile.sentiment_stats.get(count_key, 0) + 1
            )

        self._update_product_issues_and_strengths(profile)
        self.save_product(profile)
        return profile

    @staticmethod
    def _update_product_issues_and_strengths(profile: ProductProfile) -> None:
        """从高频关键词中提取常见问题和优势"""
        sorted_kws = sorted(
            profile.keyword_counts.items(), key=lambda x: x[1], reverse=True
        )

        issues = []
        strengths = []
        for kw, count in sorted_kws[:30]:
            if count < 2:
                continue
            if kw in _NEGATIVE_SENTIMENT_KEYWORDS:
                if kw not in issues:
                    issues.append(kw)
            elif kw in _POSITIVE_SENTIMENT_KEYWORDS:
                if kw not in strengths:
                    strengths.append(kw)

        if issues:
            profile.common_issues = issues[:15]
        if strengths:
            profile.strengths = strengths[:15]

    # ------------------------------------------------------------------
    # 输出详细程度控制
    # ------------------------------------------------------------------

    def get_output_detail_level(self, reviewer_id: str) -> str:
        """
        根据评论者历史分析次数返回输出详细程度。

        Returns:
            "full" (< 5 次)
            "standard" (5-20 次)
            "delta" (> 20 次，仅输出增量变化)
        """
        profile = self.get_reviewer(reviewer_id)
        if profile is None:
            return "full"

        if profile.total_reviews < 5:
            return "full"
        elif profile.total_reviews <= 20:
            return "standard"
        else:
            return "delta"

    def get_delta_keywords(
        self, reviewer_id: str, current_keywords: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """返回相对于评论者历史画像的增量关键词（新出现的）"""
        profile = self.get_reviewer(reviewer_id)
        if profile is None:
            return current_keywords

        known_keywords = set(profile.keyword_frequency.keys())
        return [
            kw for kw in current_keywords
            if kw.get("keyword", "") not in known_keywords
        ]

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def list_reviewers(self) -> List[str]:
        return sorted([
            p.stem for p in self._reviewers_dir.glob("*.json")
        ])

    def list_products(self) -> List[str]:
        return sorted([
            p.stem for p in self._products_dir.glob("*.json")
        ])

    def get_global_stats(self) -> Dict[str, Any]:
        reviewers = self.list_reviewers()
        products = self.list_products()

        total_reviews = 0
        for rid in reviewers:
            profile = self.get_reviewer(rid)
            if profile:
                total_reviews += profile.total_reviews

        return {
            "total_reviewers": len(reviewers),
            "total_products": len(products),
            "total_reviews_tracked": total_reviews,
        }

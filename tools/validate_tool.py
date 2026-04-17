"""
关键词校验工具

封装 core/post_process.py 中的校验逻辑，对已提取的关键词进行
停用词过滤、时间/日期过滤、长度过滤、原文对齐校验等。
"""

import logging
import time
from typing import Any, Dict, List

from .base_tool import BaseTool, ToolResult
from ..core.post_process import (
    post_process_keywords,
    load_stopwords,
    is_time_keyword,
    is_date_keyword,
    validate_keyword_chars_in_text,
)

logger = logging.getLogger(__name__)


class ValidateTool(BaseTool):

    @property
    def name(self) -> str:
        return "validate_keywords"

    @property
    def description(self) -> str:
        return (
            "对已提取的关键词列表进行去重和质量校验。"
            "支持的校验项：重复关键词去重、停用词过滤、时间/日期关键词过滤、"
            "超长关键词过滤、原文对齐验证（关键词的每个字符是否在原文中紧凑存在）。"
            "返回校验通过的关键词和被移除的关键词及其原因。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "description": "三元组 [推理说明, 关键词, 分数]",
                    },
                    "description": "待校验的关键词列表，格式为 [[推理, 关键词, 分数], ...]",
                },
                "original_text": {
                    "type": "string",
                    "description": "原始评论文本（用于原文对齐校验）",
                },
                "max_keyword_length": {
                    "type": "integer",
                    "description": "单个关键词允许的最大字符长度",
                    "default": 6,
                },
                "filter_time": {
                    "type": "boolean",
                    "description": "是否过滤时间相关关键词",
                    "default": True,
                },
                "filter_date": {
                    "type": "boolean",
                    "description": "是否过滤日期相关关键词",
                    "default": True,
                },
                "filter_stopwords": {
                    "type": "boolean",
                    "description": "是否过滤停用词",
                    "default": True,
                },
            },
            "required": ["keywords", "original_text"],
        }

    def execute(self, **kwargs) -> ToolResult:
        keywords: List = kwargs.get("keywords", [])
        original_text: str = kwargs.get("original_text", "")
        max_keyword_length: int = kwargs.get("max_keyword_length", 6)
        filter_time: bool = kwargs.get("filter_time", True)
        filter_date: bool = kwargs.get("filter_date", True)
        do_filter_stopwords: bool = kwargs.get("filter_stopwords", True)

        start = time.time()

        if not keywords:
            return ToolResult(
                success=False,
                error="关键词列表为空",
                metadata={"elapsed_ms": 0},
            )

        try:
            removed: List[Dict[str, str]] = []
            valid: List = []
            seen_keywords: set = set()
            keyword_idx = 1

            for item in keywords:
                if not isinstance(item, list) or len(item) <= keyword_idx:
                    continue
                kw = item[keyword_idx]
                if not isinstance(kw, str) or not kw.strip():
                    removed.append({"keyword": str(kw), "reason": "空关键词或非字符串"})
                    continue

                if kw in seen_keywords:
                    removed.append({"keyword": kw, "reason": "重复关键词"})
                    continue
                seen_keywords.add(kw)

                reason = None

                if len(kw) > max_keyword_length:
                    reason = f"超长（{len(kw)} > {max_keyword_length}）"
                elif filter_time and is_time_keyword(kw):
                    reason = "时间关键词"
                elif filter_date and is_date_keyword(kw):
                    reason = "日期关键词"
                elif do_filter_stopwords:
                    stopwords = load_stopwords("stopwords.txt")
                    if kw in stopwords:
                        reason = "停用词"

                if original_text and reason is None:
                    if not validate_keyword_chars_in_text(kw, original_text, max_span_ratio=2):
                        reason = "原文对齐失败"

                if reason:
                    removed.append({"keyword": kw, "reason": reason})
                else:
                    valid.append(item)

            elapsed = round((time.time() - start) * 1000, 2)
            return ToolResult(
                success=True,
                data={
                    "valid_keywords": valid,
                    "removed_keywords": removed,
                    "valid_count": len(valid),
                    "removed_count": len(removed),
                },
                metadata={"elapsed_ms": elapsed},
            )

        except Exception as e:
            logger.exception("关键词校验工具执行异常")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"elapsed_ms": round((time.time() - start) * 1000, 2)},
            )

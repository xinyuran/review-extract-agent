"""
快速执行路径（跳过 ReAct 循环）

提供 offline 和 fast 两种模式的管线式执行：
- offline: 仅使用 jieba 提取关键词，无 LLM 调用
- fast: 依次执行预处理 → LLM 关键词提取 → 校验 → 情感分析
"""

import logging
import time
from typing import Any, Callable, Dict

from ..tools.base_tool import ToolResult

logger = logging.getLogger(__name__)


class FastPathExecutor:
    """
    管线式快速分析执行器。

    不经过 ReAct 推理循环，直接按预定义顺序调用工具。
    """

    def __init__(self, execute_tool: Callable[..., ToolResult]):
        self._execute_tool = execute_tool

    @staticmethod
    def _format_keywords(keywords) -> list:
        return [
            {"keyword": item[1], "reasoning": item[0], "score": item[2]}
            for item in keywords
            if isinstance(item, list) and len(item) >= 3
        ]

    def run_offline(self, comment: str) -> Dict[str, Any]:
        start = time.time()
        result: Dict[str, Any] = {"original_text": comment}

        prep = self._execute_tool("text_preprocess", {"text": comment})
        cleaned = prep.data["cleaned_text"] if prep.success else comment.strip()
        result["cleaned_text"] = cleaned

        jieba_result = self._execute_tool("jieba_extract", {"text": cleaned})
        keywords = jieba_result.data.get("keywords", []) if jieba_result.success else []

        if keywords:
            val_result = self._execute_tool(
                "validate_keywords", {"keywords": keywords, "original_text": cleaned}
            )
            if val_result.success:
                keywords = val_result.data.get("valid_keywords", keywords)

        result["keywords"] = self._format_keywords(keywords)
        result["sentiment"] = None
        result["analysis_complete"] = True
        result["elapsed_ms"] = round((time.time() - start) * 1000, 2)
        result["mode"] = "offline"
        return result

    def run_fast(self, comment: str) -> Dict[str, Any]:
        start = time.time()
        result: Dict[str, Any] = {"original_text": comment}

        prep = self._execute_tool("text_preprocess", {"text": comment})
        cleaned = prep.data["cleaned_text"] if prep.success else comment.strip()
        result["cleaned_text"] = cleaned

        kw_result = self._execute_tool("keyword_extract", {"text": cleaned})
        keywords = []
        if kw_result.success:
            keywords = kw_result.data.get("keywords", [])
        else:
            jieba_result = self._execute_tool("jieba_extract", {"text": cleaned})
            if jieba_result.success:
                keywords = jieba_result.data.get("keywords", [])

        if keywords:
            val_result = self._execute_tool(
                "validate_keywords", {"keywords": keywords, "original_text": cleaned}
            )
            if val_result.success:
                keywords = val_result.data.get("valid_keywords", keywords)

        result["keywords"] = self._format_keywords(keywords)

        sent_result = self._execute_tool("sentiment_analyze", {"text": cleaned})
        if sent_result.success:
            result["sentiment"] = sent_result.data
        else:
            result["sentiment"] = {"label": "unknown", "confidence": 0, "reasoning": "分析失败"}

        result["analysis_complete"] = True
        result["elapsed_ms"] = round((time.time() - start) * 1000, 2)
        result["mode"] = "fast"
        return result

    # ------------------------------------------------------------------
    # 流式快速路径
    # ------------------------------------------------------------------

    def run_offline_stream(self, comment: str):
        """流式 offline 模式，yield 步骤级事件。"""
        start = time.time()
        result: Dict[str, Any] = {"original_text": comment}

        yield {"type": "step_start", "step_name": "text_preprocess"}
        prep = self._execute_tool("text_preprocess", {"text": comment})
        cleaned = prep.data["cleaned_text"] if prep.success else comment.strip()
        result["cleaned_text"] = cleaned
        yield {"type": "step_done", "step_name": "text_preprocess", "success": prep.success}

        yield {"type": "step_start", "step_name": "jieba_extract"}
        jieba_result = self._execute_tool("jieba_extract", {"text": cleaned})
        keywords = jieba_result.data.get("keywords", []) if jieba_result.success else []
        yield {"type": "step_done", "step_name": "jieba_extract", "success": jieba_result.success}

        if keywords:
            yield {"type": "step_start", "step_name": "validate_keywords"}
            val_result = self._execute_tool(
                "validate_keywords", {"keywords": keywords, "original_text": cleaned}
            )
            if val_result.success:
                keywords = val_result.data.get("valid_keywords", keywords)
            yield {"type": "step_done", "step_name": "validate_keywords", "success": val_result.success}

        result["keywords"] = self._format_keywords(keywords)
        result["sentiment"] = None
        result["analysis_complete"] = True
        result["elapsed_ms"] = round((time.time() - start) * 1000, 2)
        result["mode"] = "offline"
        yield {"type": "result", "data": result}

    def run_fast_stream(self, comment: str):
        """流式 fast 模式，yield 步骤级事件。"""
        start = time.time()
        result: Dict[str, Any] = {"original_text": comment}

        yield {"type": "step_start", "step_name": "text_preprocess"}
        prep = self._execute_tool("text_preprocess", {"text": comment})
        cleaned = prep.data["cleaned_text"] if prep.success else comment.strip()
        result["cleaned_text"] = cleaned
        yield {"type": "step_done", "step_name": "text_preprocess", "success": prep.success}

        yield {"type": "step_start", "step_name": "keyword_extract"}
        kw_result = self._execute_tool("keyword_extract", {"text": cleaned})
        keywords = []
        if kw_result.success:
            keywords = kw_result.data.get("keywords", [])
            yield {"type": "step_done", "step_name": "keyword_extract", "success": True}
        else:
            yield {"type": "step_done", "step_name": "keyword_extract", "success": False}
            yield {"type": "step_start", "step_name": "jieba_extract"}
            jieba_result = self._execute_tool("jieba_extract", {"text": cleaned})
            if jieba_result.success:
                keywords = jieba_result.data.get("keywords", [])
            yield {"type": "step_done", "step_name": "jieba_extract", "success": jieba_result.success}

        if keywords:
            yield {"type": "step_start", "step_name": "validate_keywords"}
            val_result = self._execute_tool(
                "validate_keywords", {"keywords": keywords, "original_text": cleaned}
            )
            if val_result.success:
                keywords = val_result.data.get("valid_keywords", keywords)
            yield {"type": "step_done", "step_name": "validate_keywords", "success": val_result.success}

        result["keywords"] = self._format_keywords(keywords)

        yield {"type": "step_start", "step_name": "sentiment_analyze"}
        sent_result = self._execute_tool("sentiment_analyze", {"text": cleaned})
        if sent_result.success:
            result["sentiment"] = sent_result.data
        else:
            result["sentiment"] = {"label": "unknown", "confidence": 0, "reasoning": "分析失败"}
        yield {"type": "step_done", "step_name": "sentiment_analyze", "success": sent_result.success}

        result["analysis_complete"] = True
        result["elapsed_ms"] = round((time.time() - start) * 1000, 2)
        result["mode"] = "fast"
        yield {"type": "result", "data": result}

"""
Jieba 兜底关键词提取工具

封装 core/fallback_extractor.py，提供 TF-IDF / TextRank / 简单分词
三种提取模式，由 Agent 在 LLM 提取失败时主动调用。
"""

import logging
import time
from typing import Any, Dict

"""
在 import jieba 之前设置 logging.getLogger("jieba").setLevel(logging.WARNING)，
这样 jieba 加载词典时的 Building prefix dict...、Loading model from cache...、Prefix dict has been built successfully. 等 DEBUG 级别日志将不再显示到终端
"""
logging.getLogger("jieba").setLevel(logging.WARNING)

from .base_tool import BaseTool, ToolResult
from ..core.fallback_extractor import jieba_fallback_extract

logger = logging.getLogger(__name__)


class JiebaExtractTool(BaseTool):

    @property
    def name(self) -> str:
        return "jieba_extract"

    @property
    def description(self) -> str:
        return (
            "使用 Jieba 分词工具从中文文本中提取关键词（兜底方案）。"
            "支持三种方法：tfidf（基于 TF-IDF 权重）、textrank（基于图排序）、"
            "simple（基于词性标注提取名词和形容词）。"
            "当 LLM 关键词提取失败或结果为空时，可调用此工具作为降级方案。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "待提取关键词的文本",
                },
                "method": {
                    "type": "string",
                    "enum": ["tfidf", "textrank", "simple"],
                    "description": "提取方法：tfidf（推荐）、textrank、simple",
                    "default": "tfidf",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的关键词数量上限",
                    "default": 8,
                },
            },
            "required": ["text"],
        }

    def execute(self, **kwargs) -> ToolResult:
        text = kwargs.get("text", "")
        method = kwargs.get("method", "tfidf")
        top_k = kwargs.get("top_k", 8)

        start = time.time()

        if not text or not text.strip():
            return ToolResult(
                success=False,
                error="输入文本为空",
                metadata={"elapsed_ms": 0},
            )

        try:
            result = jieba_fallback_extract(text, method=method, topK=top_k)
            elapsed = round((time.time() - start) * 1000, 2)

            if result:
                return ToolResult(
                    success=True,
                    data={"keywords": result},
                    metadata={"elapsed_ms": elapsed, "method": method, "count": len(result)},
                )
            else:
                return ToolResult(
                    success=False,
                    error="Jieba 未能提取到任何关键词",
                    metadata={"elapsed_ms": elapsed, "method": method},
                )

        except Exception as e:
            logger.exception("Jieba 提取工具执行异常")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"elapsed_ms": round((time.time() - start) * 1000, 2)},
            )

"""
文本预处理工具

封装 core/preprocess.py，将评论文本清洗为适合 LLM 输入的格式。
"""

import logging
import time
from typing import Any, Dict

from .base_tool import BaseTool, ToolResult
from ..core.preprocess import preprocess_comment, advanced_preprocess

logger = logging.getLogger(__name__)


class PreprocessTool(BaseTool):

    @property
    def name(self) -> str:
        return "text_preprocess"

    @property
    def description(self) -> str:
        return (
            "对中文电商评论文本进行预处理清洗，包括去除 URL、emoji、乱码、"
            "特殊符号、HTML 实体，规范化空白字符等。返回清洗后的纯净文本。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "待预处理的评论原始文本",
                },
                "remove_dates": {
                    "type": "boolean",
                    "description": "是否去除日期/时间表达式",
                    "default": False,
                },
                "keep_chinese_only": {
                    "type": "boolean",
                    "description": "是否只保留中文字符",
                    "default": False,
                },
                "max_length": {
                    "type": "integer",
                    "description": "文本最大长度限制，超出部分截断",
                    "default": 512,
                },
            },
            "required": ["text"],
        }

    def execute(self, **kwargs) -> ToolResult:
        text = kwargs.get("text", "")
        remove_dates = kwargs.get("remove_dates", False)
        keep_chinese_only = kwargs.get("keep_chinese_only", False)
        max_length = kwargs.get("max_length", 512)

        start = time.time()

        if not text or not text.strip():
            return ToolResult(
                success=False,
                error="输入文本为空",
                metadata={"elapsed_ms": 0},
            )

        try:
            cleaned = preprocess_comment(
                text,
                remove_english=False,
                deduplicate_punctuation=True,
                remove_html_entities=True,
                normalize_whitespace=True,
                remove_control_chars=True,
                remove_dates_flag=remove_dates,
                keep_chinese_only_flag=keep_chinese_only,
                keep_numbers=True,
                keep_chinese_punctuation=True,
                max_length=max_length,
            )

            if not keep_chinese_only:
                cleaned = advanced_preprocess(
                    cleaned,
                    remove_urls_flag=True,
                    remove_emails_flag=True,
                    remove_phones_flag=True,
                    normalize_numbers_flag=True,
                    remove_emojis_flag=True,
                    remove_garbled_flag=True,
                    remove_special_symbols_flag=True,
                )

            elapsed = round((time.time() - start) * 1000, 2)

            return ToolResult(
                success=True,
                data={"cleaned_text": cleaned, "original_length": len(text), "cleaned_length": len(cleaned)},
                metadata={"elapsed_ms": elapsed},
            )

        except Exception as e:
            logger.exception("预处理工具执行异常")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"elapsed_ms": round((time.time() - start) * 1000, 2)},
            )

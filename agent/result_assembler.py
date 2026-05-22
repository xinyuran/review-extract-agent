"""
从 AgentMemory 工具结果中组装最终分析输出

将 ReAct 循环中积累的工具调用结果（通过 role=tool 或 user 消息中的
[工具 X 结果] 标签块）提取并汇总为统一的结构化分析结果。
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .memory import AgentMemory

logger = logging.getLogger(__name__)

_TOOL_RESULT_BLOCK_PATTERN = re.compile(
    r"\[工具\s+(\S+)\s+结果\]\n(\{.*?\})(?=\n\n|\Z)",
    re.DOTALL,
)


def assemble_result_from_memory(
    memory: AgentMemory, original_comment: str
) -> Dict[str, Any]:
    """
    从 Memory 消息历史中收集所有工具结果，组装最终分析输出。

    支持两种工具结果来源：
    1. role=tool 消息（Native Function Calling 路径）
    2. user 消息中的 [工具 X 结果] 文本块（Prompt-based 路径）
    """
    result: Dict[str, Any] = {
        "analysis_complete": False,
        "original_text": original_comment,
    }

    cleaned_text: Optional[str] = None
    keywords: List[Any] = []
    validated_keywords: Optional[List[Any]] = None
    sentiment: Optional[Dict[str, Any]] = None
    keyword_thinking: Optional[str] = None

    tool_results: List[tuple] = []

    messages = memory.to_messages()
    for msg in messages:
        role = msg.get("role", "")

        if role == "tool":
            tool_name = msg.get("name", "")
            content_str = msg.get("content", "")
            try:
                tool_output = json.loads(content_str)
                tool_results.append((tool_name, tool_output))
            except json.JSONDecodeError as e:
                logger.warning(
                    "解析工具 %s 结果失败 (JSON 无效): %s, content[:200]=%s",
                    tool_name, e, content_str[:200]
                )

        elif role == "user":
            content = msg.get("content", "")
            for block in _TOOL_RESULT_BLOCK_PATTERN.finditer(content):
                t_name = block.group(1)
                t_json_str = block.group(2).strip()
                try:
                    tool_output = json.loads(t_json_str)
                    tool_results.append((t_name, tool_output))
                except json.JSONDecodeError:
                    pass

    tool_errors: List[str] = []
    debug_logs: List[str] = []
    for tool_name, tool_output in tool_results:
        if not tool_output.get("success", False):
            err = tool_output.get("error", "未知错误")
            tool_errors.append(f"工具 {tool_name} 失败: {err}")
            meta = tool_output.get("metadata", {})
            if meta.get("debug_log"):
                debug_logs.append(meta["debug_log"])
            continue

        data = tool_output.get("data")
        if data is None:
            continue

        if tool_name == "text_preprocess":
            cleaned_text = data.get("cleaned_text")
        elif tool_name == "keyword_extract":
            kw_list = data.get("keywords", [])
            if kw_list:
                keywords = kw_list
            thinking = data.get("thinking")
            if thinking:
                keyword_thinking = thinking
        elif tool_name == "jieba_extract":
            if not keywords:
                kw_list = data.get("keywords", [])
                if kw_list:
                    keywords = kw_list
        elif tool_name == "validate_keywords":
            val_kw = data.get("valid_keywords")
            if val_kw is not None:
                validated_keywords = val_kw
        elif tool_name == "sentiment_analyze":
            sentiment = data

    if cleaned_text:
        result["cleaned_text"] = cleaned_text

    final_keywords = validated_keywords if validated_keywords is not None else keywords
    structured = [
        {"keyword": item[1], "reasoning": item[0], "score": item[2]}
        if isinstance(item, list) and len(item) >= 3
        else item
        for item in final_keywords
    ]
    seen_kw: set = set()
    deduped: list = []
    for item in structured:
        kw = item.get("keyword", "") if isinstance(item, dict) else ""
        if kw and kw not in seen_kw:
            seen_kw.add(kw)
            deduped.append(item)
    result["keywords"] = deduped

    if keyword_thinking:
        result["keyword_thinking"] = keyword_thinking

    if sentiment:
        result["sentiment"] = sentiment
    else:
        result["sentiment"] = {"label": "unknown", "confidence": 0, "reasoning": "情感分析未执行"}

    result["analysis_complete"] = bool(result["keywords"]) or bool(sentiment)

    if tool_errors:
        result["tool_errors"] = tool_errors
    if debug_logs:
        result["debug_logs"] = debug_logs

    return result

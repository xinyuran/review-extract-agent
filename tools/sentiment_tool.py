"""
情感分析工具

调用 Tool LLM（SFT+DPO+GRPO 微调模型）进行情绪分析，
使用与微调训练时一致的 Prompt 模板（core/sentiment_prompt.py）。

微调模型输出格式为三元组：
  {"sentiment": [["推理说明", "情绪类别(正向/中立/负向)", 置信概率]]}

本工具负责将其转换为 Agent 统一的结构化格式：
  {"label": "positive/negative/neutral", "confidence": 0.95, "reasoning": "..."}
"""

import json
import logging
import re
import time
from typing import Any, Dict, Optional

from openai import OpenAI

from .base_tool import BaseTool, ToolResult
from ..config import AgentConfig
from ..core.sentiment_prompt import get_sentiment_analysis_prompt

logger = logging.getLogger(__name__)

_SENTIMENT_JSON_PATTERN = re.compile(
    r"\{[^{}]*\"sentiment\"\s*:\s*\[.*?\]\s*\}", re.DOTALL
)

_LABEL_MAP = {
    "正向": "positive",
    "负向": "negative",
    "中立": "neutral",
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
}


class SentimentTool(BaseTool):

    def __init__(self, config: AgentConfig | None = None):
        self._config = config or AgentConfig()
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self._config.TOOL_LLM_BASE_URL,
                api_key=self._config.TOOL_LLM_API_KEY,
            )
        return self._client

    @property
    def name(self) -> str:
        return "sentiment_analyze"

    @property
    def description(self) -> str:
        return (
            "分析中文电商评论的情感倾向。"
            "返回情感标签（positive/negative/neutral）、置信度和推理说明。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "待分析情感的评论文本",
                },
            },
            "required": ["text"],
        }

    @staticmethod
    def _extract_json_from_response(raw: str) -> Optional[str]:
        """从模型的"推理+JSON"混合输出中提取 JSON 部分。"""
        text = raw.strip()

        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        elif text.startswith("```"):
            text = text.replace("```", "").strip()

        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        match = _SENTIMENT_JSON_PATTERN.search(raw)
        if match:
            candidate = match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        last_brace = raw.rfind("}")
        if last_brace == -1:
            return None
        for i in range(len(raw)):
            if raw[i] == "{":
                candidate = raw[i:last_brace + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

        return None

    @staticmethod
    def _parse_sentiment_response(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        将微调模型的三元组输出转换为 Agent 统一格式。

        微调模型可能输出两种格式：
          嵌套: {"sentiment": [["推理", "正向/中立/负向", 0.95]]}
          扁平: {"sentiment": ["推理", "正向/中立/负向", 0.95]}
        统一转换为: {"label": "positive", "confidence": 0.95, "reasoning": "..."}
        """
        sentiment_data = result.get("sentiment")

        if isinstance(sentiment_data, list) and len(sentiment_data) > 0:
            triple = sentiment_data[0]
            if isinstance(triple, list) and len(triple) >= 3:
                reasoning = str(triple[0])
                raw_label = str(triple[1])
                confidence = float(triple[2])
                label = _LABEL_MAP.get(raw_label, "neutral")
                return {
                    "label": label,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }

            if isinstance(triple, str) and len(sentiment_data) >= 3:
                reasoning = str(sentiment_data[0])
                raw_label = str(sentiment_data[1])
                try:
                    confidence = float(sentiment_data[2])
                except (ValueError, TypeError):
                    confidence = 0.5
                label = _LABEL_MAP.get(raw_label, "neutral")
                return {
                    "label": label,
                    "confidence": confidence,
                    "reasoning": reasoning,
                }

        if isinstance(sentiment_data, str):
            label = _LABEL_MAP.get(sentiment_data, sentiment_data)
        else:
            label = _LABEL_MAP.get(
                result.get("label", "neutral"), "neutral"
            )
        confidence = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "")

        return {
            "label": label,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    def execute(self, **kwargs) -> ToolResult:
        text = kwargs.get("text", "")
        start = time.time()

        if not text or not text.strip():
            return ToolResult(
                success=False,
                error="输入文本为空",
                metadata={"elapsed_ms": 0},
            )

        cfg = self._config

        try:
            system_prompt, user_prompt = get_sentiment_analysis_prompt(text)

            client = self._get_client()
            resp = client.chat.completions.create(
                model=cfg.TOOL_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=cfg.TOOL_LLM_RESPONSE_FORMAT,
                max_tokens=cfg.TOOL_LLM_MAX_TOKENS,
                temperature=0,
                seed=cfg.TOOL_LLM_SEED,
            )

            raw = resp.choices[0].message.content.strip()
            json_str = self._extract_json_from_response(raw)
            if json_str is None:
                raise json.JSONDecodeError("无法从模型输出中提取 JSON", raw, 0)

            result = json.loads(json_str)
            elapsed = round((time.time() - start) * 1000, 2)

            parsed = self._parse_sentiment_response(result)

            return ToolResult(
                success=True,
                data=parsed,
                metadata={"elapsed_ms": elapsed, "model": cfg.TOOL_LLM_MODEL},
            )

        except json.JSONDecodeError as e:
            logger.warning(f"情感分析 JSON 解析失败: {e}")
            return ToolResult(
                success=False,
                error=f"JSON 解析错误: {e}",
                metadata={"elapsed_ms": round((time.time() - start) * 1000, 2)},
            )
        except Exception as e:
            logger.exception("情感分析工具执行异常")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"elapsed_ms": round((time.time() - start) * 1000, 2)},
            )

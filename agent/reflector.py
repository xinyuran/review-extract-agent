"""
结果自反思与修正器（四层架构版）

通过 LLM Service + Skill 加载反思 prompt，
对 Agent ReAct 路径产出的分析结果进行质量自评估。

设计要点（解决 7B 模型反思循环不收敛的问题）：
1. 反思器只做"增量修订"而非"输出完整替换列表"，避免每轮丢词
2. Prompt 设置明确的通过标准和容忍度，防止模型无限纠结
3. 引入收敛检测，如果连续两轮发现相同问题则强制终止
4. 给足 max_tokens 避免 JSON 被截断
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from ..config import AgentConfig

if TYPE_CHECKING:
    from ..llm_service import LLMService

logger = logging.getLogger(__name__)


class ReflectionResult:
    """反思结果封装"""

    __slots__ = ("passed", "issues", "add_keywords", "remove_keywords",
                 "corrected_sentiment", "summary", "elapsed_ms")

    def __init__(
        self,
        passed: bool,
        issues: List[Dict[str, str]],
        add_keywords: Optional[List[Dict[str, Any]]] = None,
        remove_keywords: Optional[List[str]] = None,
        corrected_sentiment: Optional[Dict[str, Any]] = None,
        summary: str = "",
        elapsed_ms: float = 0,
    ):
        self.passed = passed
        self.issues = issues
        self.add_keywords = add_keywords or []
        self.remove_keywords = remove_keywords or []
        self.corrected_sentiment = corrected_sentiment
        self.summary = summary
        self.elapsed_ms = elapsed_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "add_keywords": self.add_keywords,
            "remove_keywords": self.remove_keywords,
            "corrected_sentiment": self.corrected_sentiment,
            "summary": self.summary,
            "elapsed_ms": self.elapsed_ms,
        }


class ResultReflector:
    """
    分析结果自反思器（四层架构版）

    通过 LLM Service 调用 Agent LLM，使用 reflector skill
    对已完成的分析结果进行质量审查。
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        llm_service: Optional[LLMService] = None,
    ):
        self._config = config or AgentConfig()
        self._llm_service = llm_service

    @staticmethod
    def _extract_json_from_response(raw: str) -> Optional[str]:
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

    def reflect(
        self,
        original_text: str,
        keywords: List[Dict[str, Any]],
        sentiment: Dict[str, Any],
    ) -> ReflectionResult:
        """对分析结果执行单轮反思（增量修订模式）"""
        start = time.time()

        keywords_json = json.dumps(keywords, ensure_ascii=False)
        sentiment_json = json.dumps(sentiment, ensure_ascii=False)

        try:
            if self._llm_service:
                skill = self._llm_service.load_skill(
                    "reflector",
                    text_length=str(len(original_text)),
                    original_text=original_text,
                    keyword_count=str(len(keywords)),
                    keywords_json=keywords_json,
                    sentiment_json=sentiment_json,
                )
                llm_resp = self._llm_service.call_agent(
                    messages=[
                        {"role": "system", "content": skill.system},
                        {"role": "user", "content": skill.user},
                    ],
                )
                raw = (llm_resp.content or "").strip()
            else:
                from ..core.reflector_prompt import (
                    _REFLECTION_SYSTEM_PROMPT,
                    _REFLECTION_USER_TEMPLATE,
                )
                from openai import OpenAI

                user_content = _REFLECTION_USER_TEMPLATE.format(
                    original_text=original_text,
                    text_length=len(original_text),
                    keyword_count=len(keywords),
                    keywords_json=keywords_json,
                    sentiment_json=sentiment_json,
                )
                client = OpenAI(
                    base_url=self._config.AGENT_LLM_BASE_URL,
                    api_key=self._config.AGENT_LLM_API_KEY,
                )
                resp = client.chat.completions.create(
                    model=self._config.AGENT_LLM_MODEL,
                    messages=[
                        {"role": "system", "content": _REFLECTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=2048,
                    temperature=0,
                )
                raw = resp.choices[0].message.content.strip()

            json_str = self._extract_json_from_response(raw)
            if json_str is None:
                raise json.JSONDecodeError("无法从反思输出中提取 JSON", raw, 0)

            result = json.loads(json_str)
            elapsed = round((time.time() - start) * 1000, 2)

            return ReflectionResult(
                passed=result.get("passed", True),
                issues=result.get("issues", []),
                add_keywords=result.get("add_keywords", []),
                remove_keywords=result.get("remove_keywords", []),
                corrected_sentiment=result.get("corrected_sentiment"),
                summary=result.get("summary", ""),
                elapsed_ms=elapsed,
            )

        except json.JSONDecodeError as e:
            logger.warning("反思结果 JSON 解析失败: %s", e)
            elapsed = round((time.time() - start) * 1000, 2)
            return ReflectionResult(
                passed=True,
                issues=[{"type": "parse_error", "detail": str(e), "severity": "low"}],
                summary="反思结果解析失败，保留原结果",
                elapsed_ms=elapsed,
            )
        except Exception as e:
            logger.exception("反思器执行异常")
            elapsed = round((time.time() - start) * 1000, 2)
            return ReflectionResult(
                passed=True,
                issues=[{"type": "execution_error", "detail": str(e), "severity": "low"}],
                summary=f"反思器异常: {e}，保留原结果",
                elapsed_ms=elapsed,
            )

    @staticmethod
    def _apply_delta(
        keywords: List[Dict[str, Any]],
        sentiment: Dict[str, Any],
        ref: ReflectionResult,
    ) -> tuple:
        updated_keywords = list(keywords)

        if ref.remove_keywords:
            remove_set = set(ref.remove_keywords)
            updated_keywords = [
                kw for kw in updated_keywords
                if kw.get("keyword", "") not in remove_set
            ]

        if ref.add_keywords:
            existing = {kw.get("keyword", "") for kw in updated_keywords}
            for new_kw in ref.add_keywords:
                if new_kw.get("keyword", "") not in existing:
                    updated_keywords.append(new_kw)

        updated_sentiment = sentiment
        if ref.corrected_sentiment:
            updated_sentiment = ref.corrected_sentiment

        return updated_keywords, updated_sentiment

    @staticmethod
    def _extract_issue_fingerprints(issues: List[Dict[str, str]]) -> Set[str]:
        return {
            f"{issue.get('type', '')}:{issue.get('detail', '')[:30]}"
            for issue in issues
            if issue.get("severity") == "high"
        }

    def reflect_and_correct(
        self,
        original_text: str,
        keywords: List[Dict[str, Any]],
        sentiment: Dict[str, Any],
        max_rounds: int = 2,
    ) -> Dict[str, Any]:
        current_keywords = keywords
        current_sentiment = sentiment
        history: List[Dict[str, Any]] = []
        prev_fingerprints: Set[str] = set()

        for round_idx in range(max_rounds):
            logger.info("反思第 %d/%d 轮", round_idx + 1, max_rounds)

            ref = self.reflect(original_text, current_keywords, current_sentiment)
            history.append({"round": round_idx + 1, **ref.to_dict()})

            if ref.passed:
                logger.info("反思通过 (第 %d 轮): %s", round_idx + 1, ref.summary)
                break

            curr_fingerprints = self._extract_issue_fingerprints(ref.issues)
            if curr_fingerprints and curr_fingerprints == prev_fingerprints:
                logger.warning(
                    "反思收敛检测触发 (第 %d 轮): 与上一轮发现相同问题，强制终止",
                    round_idx + 1,
                )
                history[-1]["forced_stop"] = True
                break

            prev_fingerprints = curr_fingerprints
            current_keywords, current_sentiment = self._apply_delta(
                current_keywords, current_sentiment, ref
            )

        return {
            "keywords": current_keywords,
            "sentiment": current_sentiment,
            "reflection_history": history,
            "total_rounds": len(history),
            "final_passed": history[-1]["passed"] if history else True,
        }

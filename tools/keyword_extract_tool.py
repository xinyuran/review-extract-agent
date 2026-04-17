"""
LLM 关键词提取工具（核心工具）

封装 processor.py 中的 LLM 调用与 JSON 解析逻辑，
去掉兜底链（兜底决策交由 Agent），保留重试机制。
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base_tool import BaseTool, ToolResult
from ..config import AgentConfig
from ..core.prompt_template_3 import get_keyword_extraction_prompt_3
from ..core.prompt_template_4_shortComment import get_keyword_extraction_prompt_simple
from ..core.post_process import (
    post_process_keywords,
    extract_keywords_from_json,
    normalize_keywords_data,
)

logger = logging.getLogger(__name__)

_JSON_BLOCK_PATTERN = re.compile(
    r'\{\s*"keywords"\s*:\s*\[', re.DOTALL
)


class KeywordExtractTool(BaseTool):

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
        return "keyword_extract"

    @property
    def description(self) -> str:
        return (
            "使用 LLM 从中文电商评论中提取结构化关键词。"
            "自动选择长评/短评 Prompt 模板，调用 Tool LLM 进行推理，"
            "解析 JSON 结果并进行后处理（去重、排序、过滤）。"
            "返回关键词三元组列表 [[推理说明, 关键词, 重要性分数], ...]。"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "待提取关键词的评论文本（建议已预处理）",
                },
                "max_keywords": {
                    "type": "integer",
                    "description": "最多返回的关键词数量",
                    "default": 8,
                },
            },
            "required": ["text"],
        }

    @staticmethod
    def _find_matching_brace(text: str, start: int) -> int:
        """从 start 位置的 '{' 开始，找到与之匹配的 '}'，处理嵌套和字符串引号。"""
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return i
        return -1

    @staticmethod
    def _sanitize_json(candidate: str) -> str:
        """
        修复微调模型 JSON 输出中常见的格式错误：
        1. 中文逗号 `，` → 英文逗号 `,`
        2. 中文冒号 `：` → 英文冒号 `:`（仅在 JSON 结构位置）
        3. 移除 keywords 数组中不合法的尾部元素
        """
        fixed = candidate.replace('，', ',').replace('：', ':')

        try:
            json.loads(fixed)
            return fixed
        except json.JSONDecodeError:
            pass

        kw_match = re.search(r'"keywords"\s*:\s*\[', fixed)
        if not kw_match:
            return fixed

        arr_start = kw_match.end() - 1
        depth = 0
        in_str = False
        esc = False
        last_valid_end = -1
        elem_start = -1
        elem_depth_bracket = 0

        for i in range(arr_start, len(fixed)):
            ch = fixed[i]
            if esc:
                esc = False
                continue
            if ch == '\\' and in_str:
                esc = True
                continue
            if ch == '"' and not esc:
                in_str = not in_str
                continue
            if in_str:
                continue

            if ch == '[':
                depth += 1
                if depth == 2:
                    elem_start = i
                    elem_depth_bracket = 0
            elif ch == ']':
                depth -= 1
                if depth == 1 and elem_start != -1:
                    elem_text = fixed[elem_start:i + 1]
                    try:
                        json.loads(elem_text)
                        last_valid_end = i
                    except json.JSONDecodeError:
                        pass
                    elem_start = -1
                elif depth == 0:
                    break

        if last_valid_end > 0:
            rebuilt = fixed[:last_valid_end + 1]
            rebuilt = rebuilt.rstrip()
            if rebuilt.endswith(','):
                rebuilt = rebuilt[:-1]
            rebuilt += '\n  ]\n}'
            try:
                json.loads(rebuilt)
                return rebuilt
            except json.JSONDecodeError:
                pass

        return fixed

    @classmethod
    def _try_load(cls, candidate: str) -> Optional[str]:
        """
        尝试解析 JSON 字符串。先原样解析，失败则 sanitize 修复后重试。
        成功时返回可解析的 JSON 字符串，失败返回 None。
        """
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

        sanitized = cls._sanitize_json(candidate)
        try:
            json.loads(sanitized)
            return sanitized
        except json.JSONDecodeError:
            return None

    @classmethod
    def _extract_json_and_thinking(cls, raw: str) -> tuple:
        """
        从模型输出中提取 JSON 部分和思考文本。

        策略（按优先级）：
        1. 整段直接解析为 JSON
        2. 剥离 ```json ``` 标记后解析
        3. 找到 {"keywords": [ 开头，用括号匹配定位完整 JSON
        4. 从后往前找最后一个 } 对应的 {，尝试解析
        每步都先原样解析，失败则用 _sanitize_json 修复后重试。

        Returns:
            (json_str, thinking_text) 元组。
            json_str 为 None 表示提取失败；
            thinking_text 为 JSON 之前的推理文本（可能为空字符串）。
        """
        text = raw.strip()

        result = cls._try_load(text)
        if result is not None:
            return result, ""

        cleaned = text
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        if cleaned != text:
            result = cls._try_load(cleaned)
            if result is not None:
                return result, ""

        match = _JSON_BLOCK_PATTERN.search(raw)
        if match:
            brace_start = match.start()
            brace_end = cls._find_matching_brace(raw, brace_start)
            if brace_end != -1:
                candidate = raw[brace_start:brace_end + 1]
                result = cls._try_load(candidate)
                if result is not None:
                    thinking = raw[:brace_start].strip()
                    return result, thinking

        last_brace = raw.rfind("}")
        if last_brace == -1:
            return None, raw.strip()

        for i in range(len(raw) - 1, -1, -1):
            if raw[i] == "{":
                candidate = raw[i:last_brace + 1]
                result = cls._try_load(candidate)
                if result is not None:
                    thinking = raw[:i].strip()
                    return result, thinking

        return None, raw.strip()

    @staticmethod
    def _try_repair_truncated_json(raw: str) -> Optional[str]:
        """
        当 finish_reason == "length" 导致 JSON 被截断时，
        尝试找到 {"keywords": [...]} 的起始位置，然后补全缺失的括号。
        只处理最常见的截断情况：数组元素被截断。
        """
        pattern = re.compile(r'\{\s*"keywords"\s*:\s*\[', re.DOTALL)
        match = pattern.search(raw)
        if not match:
            return None

        start = match.start()
        fragment = raw[start:]

        depth_brace = 0
        depth_bracket = 0
        in_string = False
        escape = False

        for ch in fragment:
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth_brace += 1
            elif ch == '}':
                depth_brace -= 1
            elif ch == '[':
                depth_bracket += 1
            elif ch == ']':
                depth_bracket -= 1

        last_complete_bracket = fragment.rfind(']')
        if last_complete_bracket == -1:
            last_bracket_open = fragment.rfind('[')
            if last_bracket_open != -1:
                repaired = fragment[:last_bracket_open + 1] + ']}'
                try:
                    json.loads(repaired)
                    return repaired
                except json.JSONDecodeError:
                    pass
            return None

        last_comma = fragment.rfind(',', 0, last_complete_bracket)
        if last_comma != -1:
            candidate = fragment[:last_complete_bracket + 1]
            candidate = candidate.rstrip().rstrip(',')
            candidate += ']}'
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        for pos in range(len(fragment) - 1, match.end() - match.start() - 1, -1):
            ch = fragment[pos]
            if ch == ']':
                suffix = fragment[:pos + 1] + '}'
                try:
                    json.loads(suffix)
                    return suffix
                except json.JSONDecodeError:
                    candidate = fragment[:pos + 1].rstrip().rstrip(',') + ']}'
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        continue

        repair = fragment.rstrip()
        if repair.endswith(','):
            repair = repair[:-1]
        repair += ']' * max(0, depth_bracket) + '}' * max(0, depth_brace)
        try:
            json.loads(repair)
            return repair
        except json.JSONDecodeError:
            pass

        return None

    def _save_debug_log(
        self,
        input_text: str,
        prompt_type: str,
        system_prompt: str,
        user_prompt: str,
        attempts: List[Dict[str, Any]],
        final_error: str,
    ) -> Optional[str]:
        """将 keyword_extract 的全部调试信息保存到 JSON 文件，返回文件路径。"""
        try:
            from ..cli.config_loader import DEFAULT_OUTPUT_DIR
            debug_dir = Path(DEFAULT_OUTPUT_DIR) / "debug"
        except Exception:
            debug_dir = Path("extract_agent_output") / "debug"

        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = debug_dir / f"keyword_extract_debug_{ts}.json"
            debug_data = {
                "timestamp": datetime.now().isoformat(),
                "input_text": input_text,
                "input_text_length": len(input_text),
                "prompt_type": prompt_type,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "model": self._config.TOOL_LLM_MODEL,
                "max_tokens": self._config.TOOL_LLM_MAX_TOKENS,
                "final_error": final_error,
                "total_attempts": len(attempts),
                "attempts": attempts,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
            logger.info(f"keyword_extract debug 日志已保存: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.warning(f"保存 debug 日志失败: {e}")
            return None

    def execute(self, **kwargs) -> ToolResult:
        text = kwargs.get("text", "")
        max_keywords = kwargs.get("max_keywords", self._config.N)

        start = time.time()

        if not text or not text.strip():
            return ToolResult(
                success=False,
                error="输入文本为空",
                metadata={"elapsed_ms": 0},
            )

        cfg = self._config

        if len(text.strip()) < cfg.SHORT_TEXT_LEN:
            system_prompt, user_prompt = get_keyword_extraction_prompt_simple(text)
            prompt_type = "short"
        else:
            system_prompt, user_prompt = get_keyword_extraction_prompt_3(text)
            prompt_type = "long"

        client = self._get_client()
        max_retries = cfg.MAX_RETRIES
        last_error = None
        attempt_logs: List[Dict[str, Any]] = []

        for attempt in range(max_retries + 1):
            attempt_record: Dict[str, Any] = {
                "attempt": attempt + 1,
                "status": "unknown",
            }
            try:
                use_penalty = attempt >= 2
                current_seed = cfg.TOOL_LLM_SEED + attempt if use_penalty else cfg.TOOL_LLM_SEED

                extra_kwargs: Dict[str, Any] = {
                    "model": cfg.TOOL_LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": cfg.TOOL_LLM_MAX_TOKENS,
                    "temperature": cfg.TOOL_LLM_TEMPERATURE,
                    "seed": current_seed,
                }

                if use_penalty:
                    extra_kwargs["frequency_penalty"] = cfg.TOOL_LLM_FREQUENCY_PENALTY
                    extra_kwargs["extra_body"] = {
                        "repetition_penalty": cfg.TOOL_LLM_REPETITION_PENALTY
                    }

                resp = client.chat.completions.create(**extra_kwargs)
                raw = resp.choices[0].message.content.strip()

                finish_reason = resp.choices[0].finish_reason
                usage = None
                if resp.usage:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }

                attempt_record["raw_output"] = raw
                attempt_record["raw_output_length"] = len(raw)
                attempt_record["finish_reason"] = finish_reason
                attempt_record["usage"] = usage
                attempt_record["seed"] = current_seed
                attempt_record["use_penalty"] = use_penalty

                json_str, thinking_text = self._extract_json_and_thinking(raw)

                if json_str is None and finish_reason == "length":
                    logger.info("finish_reason=length，尝试修复被截断的 JSON")
                    repaired = self._try_repair_truncated_json(raw)
                    if repaired:
                        json_str = repaired
                        thinking_text = raw[:raw.find(repaired[0])].strip() if repaired[0] in raw else ""
                        attempt_record["json_repaired"] = True
                        logger.info(f"截断 JSON 修复成功，长度={len(repaired)}")

                if json_str is None:
                    raw_preview = raw[:200] + ("..." if len(raw) > 200 else "")
                    truncation_hint = ""
                    if finish_reason == "length":
                        truncation_hint = (
                            "（模型输出被 max_tokens 截断，JSON 不完整。"
                            f"当前 TOOL_LLM_MAX_TOKENS={cfg.TOOL_LLM_MAX_TOKENS}，"
                            "建议增大该值或缩短评论文本）"
                        )
                    last_error = f"无法从模型输出中提取 JSON{truncation_hint}，输出片段: {raw_preview}"
                    attempt_record["status"] = "json_extract_failed"
                    attempt_record["error"] = f"无法从模型输出中提取 JSON{truncation_hint}"
                    attempt_logs.append(attempt_record)
                    logger.warning(
                        f"关键词提取无法提取 JSON (尝试 {attempt + 1}/{max_retries + 1})，"
                        f"输出长度={len(raw)}，finish_reason={finish_reason}，"
                        f"前200字: {raw_preview}"
                    )
                    continue

                attempt_record["extracted_json"] = json_str
                attempt_record["thinking_text_length"] = len(thinking_text)

                keywords_json = json.loads(json_str)
                keywords_data = extract_keywords_from_json(keywords_json)
                keywords_data = normalize_keywords_data(keywords_data, json_format="new")

                processed = post_process_keywords(
                    keywords_data,
                    deduplicate=cfg.DEDUPLICATE,
                    sort_by_importance=cfg.SORT_BY_IMPORTANCE,
                    filter_low_score=cfg.FILTER_LOW_SCORE,
                    score_threshold=cfg.SCORE_THRESHOLD,
                    top_n=True,
                    n=max_keywords,
                    return_full_info=True,
                    json_format="new",
                    remove_english=cfg.REMOVE_ENGLISH_IN_POSTPROCESS,
                    filter_stopwords=cfg.FILTER_STOPWORDS,
                    stopwords_exact_match=cfg.STOPWORDS_EXACT_MATCH,
                    stopwords_contain_match=cfg.STOPWORDS_CONTAIN_MATCH,
                    stopwords_file=cfg.STOPWORDS_FILE,
                    filter_time_keywords=cfg.FILTER_TIME_KEYWORDS,
                    filter_date_keywords=cfg.FILTER_DATE_KEYWORDS,
                    filter_long_keywords=cfg.FILTER_LONG_KEYWORDS,
                    max_keyword_length=cfg.MAX_KEYWORD_LENGTH,
                    backfill_topn=cfg.BACKFILL_TOPN,
                    filter_not_in_original=cfg.FILTER_KEYWORDS_NOT_IN_ORIGINAL,
                    original_text=text,
                    max_span_ratio=cfg.KEYWORD_MAX_SPAN_RATIO,
                )

                if processed:
                    elapsed = round((time.time() - start) * 1000, 2)
                    data: Dict[str, Any] = {"keywords": processed}
                    if thinking_text:
                        data["thinking"] = thinking_text
                    attempt_record["status"] = "success"
                    attempt_record["keywords_count"] = len(processed)
                    attempt_logs.append(attempt_record)
                    return ToolResult(
                        success=True,
                        data=data,
                        metadata={
                            "elapsed_ms": elapsed,
                            "prompt_type": prompt_type,
                            "attempts": attempt + 1,
                            "model": cfg.TOOL_LLM_MODEL,
                        },
                    )

                last_error = "LLM 返回结果经后处理后为空"
                attempt_record["status"] = "post_process_empty"
                attempt_record["error"] = last_error

            except json.JSONDecodeError as e:
                last_error = f"JSON 解析错误: {e}"
                attempt_record["status"] = "json_decode_error"
                attempt_record["error"] = last_error
                logger.warning(f"关键词提取 JSON 解析失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
            except Exception as e:
                last_error = str(e)
                attempt_record["status"] = "exception"
                attempt_record["error"] = last_error
                logger.warning(f"关键词提取异常 (尝试 {attempt + 1}/{max_retries + 1}): {e}")

            attempt_logs.append(attempt_record)

        elapsed = round((time.time() - start) * 1000, 2)
        final_error = f"经 {max_retries + 1} 次尝试后仍失败: {last_error}"

        debug_path = self._save_debug_log(
            input_text=text,
            prompt_type=prompt_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            attempts=attempt_logs,
            final_error=final_error,
        )

        return ToolResult(
            success=False,
            error=final_error,
            metadata={
                "elapsed_ms": elapsed,
                "attempts": max_retries + 1,
                "debug_log": debug_path,
            },
        )

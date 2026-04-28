"""
ReAct Agent 主循环（四层架构版）

架构职责解耦（四层）：
- Skill 层：skills/ 目录下的 SKILL.md 文件，定义各阶段的 prompt
- Agent 层（本模块）：ReAct 主循环、消息管理、工具分发
- LLM Service 层：统一管理 Agent LLM / Tool LLM 调用
- Tool 层：纯计算工具（preprocess / jieba / validate）

支持两种工具调用模式：
1. 原生 Function Calling（vLLM 需启动时加 --enable-auto-tool-choice --tool-call-parser hermes）
2. Prompt-based 模式（不依赖服务端 tool calling，通过解析 <tool_call> 标签实现）
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from ..config import AgentConfig
from ..tools.base_tool import BaseTool, ToolResult
from ..tools.preprocess_tool import PreprocessTool
from ..tools.keyword_extract_tool import KeywordExtractTool
from ..tools.jieba_extract_tool import JiebaExtractTool
from ..tools.validate_tool import ValidateTool
from ..tools.sentiment_tool import SentimentTool
from ..llm_service import LLMService, SkillLoader
from .memory import AgentMemory
from .reflector import ResultReflector

logger = logging.getLogger(__name__)

_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

_TOOL_RESULT_BLOCK_PATTERN = re.compile(
    r"\[工具\s+(\S+)\s+结果\]\n(\{.*?\})(?=\n\n|\Z)",
    re.DOTALL,
)

_LLM_POWERED_TOOLS = {"keyword_extract", "sentiment_analyze"}


class ReviewAnalysisAgent:
    """
    中文电商评论分析 Agent（四层架构）

    采用 ReAct 范式，通过 Agent LLM 进行规划和工具调度，
    自动完成评论的预处理、关键词提取、质量校验和情感分析。

    keyword_extract 和 sentiment_analyze 对 Agent LLM 仍暴露为"工具"，
    但其执行路径为: Agent 层 → LLM Service（加载 Skill）→ 后处理，
    而非委托给 Tool 层的 LLM 调用。
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        llm_service: LLMService | None = None,
    ):
        self.config = config or AgentConfig()
        self._backend_mode = self.config.get_backend_mode()

        self.llm_service = llm_service or LLMService(
            config=self.config,
            skill_loader=SkillLoader(),
        )

        self.tools: Dict[str, BaseTool] = {}
        self._all_tool_schemas: Dict[str, Dict[str, Any]] = {}
        self._init_tools()

        self.max_steps = self.config.AGENT_MAX_STEPS
        self.timeout = self.config.AGENT_TIMEOUT

        self._reflector: Optional[ResultReflector] = None
        if self.config.ENABLE_REFLECTION and self._backend_mode != "offline":
            self._reflector = ResultReflector(
                config=self.config, llm_service=self.llm_service
            )
            logger.info("反思器已启用 (max_rounds=%d)", self.config.REFLECTION_MAX_ROUNDS)

    def _init_tools(self) -> None:
        """注册纯计算工具 + LLM 驱动工具的 schema"""
        pure_tools: List[BaseTool] = [
            PreprocessTool(),
            JiebaExtractTool(),
            ValidateTool(),
        ]
        for tool in pure_tools:
            self.tools[tool.name] = tool

        if self._backend_mode != "offline":
            kw_tool = KeywordExtractTool(self.config)
            sent_tool = SentimentTool(self.config)
            for t in (kw_tool, sent_tool):
                self._all_tool_schemas[t.name] = t.to_openai_tool()

        for tool in self.tools.values():
            self._all_tool_schemas[tool.name] = tool.to_openai_tool()

        logger.info(
            "Agent 初始化完成 (后端=%s)，工具 schema: %s",
            self._backend_mode,
            list(self._all_tool_schemas.keys()),
        )

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        return list(self._all_tool_schemas.values())

    # ------------------------------------------------------------------
    # LLM 驱动的工具执行（keyword_extract / sentiment_analyze）
    # ------------------------------------------------------------------

    def _execute_llm_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> ToolResult:
        """通过 LLM Service + Skill 执行关键词提取或情感分析"""
        text = arguments.get("text", "")
        if not text or not text.strip():
            return ToolResult(success=False, error="输入文本为空")

        try:
            if tool_name == "keyword_extract":
                return self._execute_keyword_extract(text, arguments)
            elif tool_name == "sentiment_analyze":
                return self._execute_sentiment_analyze(text)
            else:
                return ToolResult(success=False, error=f"未知 LLM 工具: {tool_name}")
        except Exception as e:
            logger.exception("LLM 工具 %s 执行异常", tool_name)
            return ToolResult(success=False, error=str(e))

    def _execute_keyword_extract(
        self, text: str, arguments: Dict[str, Any]
    ) -> ToolResult:
        cfg = self.config
        max_keywords = arguments.get("max_keywords", cfg.N)

        skill_name = (
            "keyword_extract_short"
            if len(text.strip()) < cfg.SHORT_TEXT_LEN
            else "keyword_extract_long"
        )
        prompt_type = "short" if "short" in skill_name else "long"

        max_retries = cfg.MAX_RETRIES
        last_error = None
        start = time.time()

        for attempt in range(max_retries + 1):
            try:
                use_penalty = attempt >= 2
                extra_params: Dict[str, Any] = {
                    "seed": cfg.TOOL_LLM_SEED + attempt if use_penalty else cfg.TOOL_LLM_SEED,
                }
                if use_penalty:
                    extra_params["frequency_penalty"] = cfg.TOOL_LLM_FREQUENCY_PENALTY
                    extra_params["extra_body"] = {
                        "repetition_penalty": cfg.TOOL_LLM_REPETITION_PENALTY,
                    }

                llm_resp = self.llm_service.call_tool(
                    skill_name=skill_name,
                    variables={"comment": text},
                    extra_params=extra_params,
                )

                raw = (llm_resp.content or "").strip()
                json_str, thinking_text = KeywordExtractTool._extract_json_and_thinking(raw)

                if json_str is None and llm_resp.finish_reason == "length":
                    repaired = KeywordExtractTool._try_repair_truncated_json(raw)
                    if repaired:
                        json_str = repaired
                        thinking_text = ""

                if json_str is None:
                    last_error = f"无法从模型输出中提取 JSON (attempt {attempt + 1})"
                    logger.warning(last_error)
                    continue

                from ..core.post_process import (
                    post_process_keywords,
                    extract_keywords_from_json,
                    normalize_keywords_data,
                )

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
                    data: Dict[str, Any] = {"keywords": processed}
                    if thinking_text:
                        data["thinking"] = thinking_text
                    elapsed = round((time.time() - start) * 1000, 2)
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
            except json.JSONDecodeError as e:
                last_error = f"JSON 解析错误: {e}"
            except Exception as e:
                last_error = str(e)
                logger.warning("关键词提取异常 (attempt %d): %s", attempt + 1, e)

        elapsed = round((time.time() - start) * 1000, 2)
        return ToolResult(
            success=False,
            error=f"经 {max_retries + 1} 次尝试后仍失败: {last_error}",
            metadata={"elapsed_ms": elapsed, "attempts": max_retries + 1},
        )

    def _execute_sentiment_analyze(self, text: str) -> ToolResult:
        cfg = self.config
        start = time.time()

        try:
            extra_params: Dict[str, Any] = {
                "temperature": 0,
                "seed": cfg.TOOL_LLM_SEED,
            }
            if cfg.TOOL_LLM_RESPONSE_FORMAT:
                extra_params["response_format"] = cfg.TOOL_LLM_RESPONSE_FORMAT

            llm_resp = self.llm_service.call_tool(
                skill_name="sentiment_analyze",
                variables={"comment": text},
                extra_params=extra_params,
            )

            raw = (llm_resp.content or "").strip()
            json_str = SentimentTool._extract_json_from_response(raw)
            if json_str is None:
                raise json.JSONDecodeError("无法从模型输出中提取 JSON", raw, 0)

            result = json.loads(json_str)
            parsed = SentimentTool._parse_sentiment_response(result)
            elapsed = round((time.time() - start) * 1000, 2)

            return ToolResult(
                success=True,
                data=parsed,
                metadata={"elapsed_ms": elapsed, "model": cfg.TOOL_LLM_MODEL},
            )
        except Exception as e:
            logger.warning("情感分析失败: %s", e)
            elapsed = round((time.time() - start) * 1000, 2)
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"elapsed_ms": elapsed},
            )

    # ------------------------------------------------------------------
    # 统一工具执行入口
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        if tool_name in _LLM_POWERED_TOOLS:
            return self._execute_llm_tool(tool_name, arguments)

        tool = self.tools.get(tool_name)
        if tool is None:
            return ToolResult(success=False, error=f"未知工具: {tool_name}")

        try:
            start = time.time()
            result = tool.execute(**arguments)
            elapsed = time.time() - start

            if elapsed > self.config.TOOL_TIMEOUT:
                logger.warning(
                    "工具 %s 执行超时: %.1fs > %ds",
                    tool_name, elapsed, self.config.TOOL_TIMEOUT,
                )

            logger.info(
                "工具 [%s] 执行%s (耗时 %.2fs)",
                tool_name, "成功" if result.success else "失败", elapsed,
            )
            return result
        except Exception as e:
            logger.exception("工具 %s 执行异常", tool_name)
            return ToolResult(success=False, error=str(e))

    # ------------------------------------------------------------------
    # Prompt-based 工具调用解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls_from_text(content: str) -> List[Dict[str, Any]]:
        calls = []
        for match in _TOOL_CALL_PATTERN.finditer(content):
            try:
                payload = json.loads(match.group(1))
                name = payload.get("name", "")
                arguments = payload.get("arguments", {})
                if name:
                    calls.append({"name": name, "arguments": arguments})
            except json.JSONDecodeError:
                logger.warning("解析 <tool_call> JSON 失败: %s", match.group(1)[:200])
        return calls

    @staticmethod
    def _strip_tool_call_tags(content: str) -> str:
        return _TOOL_CALL_PATTERN.sub("", content).strip()

    # ------------------------------------------------------------------
    # 快速路径
    # ------------------------------------------------------------------

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

        result["keywords"] = [
            {"keyword": item[1], "reasoning": item[0], "score": item[2]}
            for item in keywords
            if isinstance(item, list) and len(item) >= 3
        ]
        result["sentiment"] = None
        result["analysis_complete"] = True
        result["elapsed_ms"] = round((time.time() - start) * 1000, 2)
        result["mode"] = "offline"
        return result

    def run_fast(self, comment: str) -> Dict[str, Any]:
        if self._backend_mode == "offline":
            return self.run_offline(comment)

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

        result["keywords"] = [
            {"keyword": item[1], "reasoning": item[0], "score": item[2]}
            for item in keywords
            if isinstance(item, list) and len(item) >= 3
        ]

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
    # Agent 主循环（ReAct + 反思外层循环）
    # ------------------------------------------------------------------

    def run(
        self,
        comment: str,
        use_fast_path: bool = False,
        reviewer_id: Optional[str] = None,
        product_id: Optional[str] = None,
        product_name: str = "",
    ) -> Dict[str, Any]:
        if self._backend_mode == "offline":
            result = self.run_offline(comment)
            self._update_knowledge(result, reviewer_id, product_id, product_name)
            return result

        if use_fast_path:
            result = self.run_fast(comment)
            self._update_knowledge(result, reviewer_id, product_id, product_name)
            return result

        native = self.llm_service.detect_tool_calling_mode()

        start_time = time.time()
        agent_trace: List[Dict[str, Any]] = []

        if native:
            skill = self.llm_service.load_skill("agent_system")
            memory = AgentMemory(system_prompt=skill.system)
        else:
            tool_desc = self.llm_service.build_tool_descriptions(self._all_tool_schemas)
            skill = self.llm_service.load_skill(
                "agent_system_tools", tool_descriptions=tool_desc
            )
            memory = AgentMemory(system_prompt=skill.system)

        user_skill = self.llm_service.load_skill("user_request", comment=comment)
        memory.add_user_message(user_skill.user)

        agent_summary = self._run_react_loop(memory, agent_trace, start_time, native)

        if agent_summary == self._TOKEN_LIMIT_MARKER:
            logger.warning("Agent 模式 token 超限，降级为 fast 模式执行")
            fast_result = self.run_fast(comment)
            fast_result["mode"] = "agent-native-fallback-fast"
            fast_result["warnings"] = [
                "评论文本较长，Agent 模式因 token 超限自动降级为 fast 管线模式"
            ]
            fast_result["agent_trace"] = agent_trace
            return fast_result

        result = self._assemble_result_from_memory(memory, comment)
        if agent_summary:
            result["agent_summary"] = agent_summary

        reflection_history: List[Dict[str, Any]] = []
        if self._reflector and result.get("analysis_complete"):
            result, reflection_history = self._code_level_reflection(
                result, memory, agent_trace, start_time
            )

        if reflection_history:
            result["reflection"] = {
                "total_rounds": len(reflection_history),
                "final_passed": reflection_history[-1].get("passed", True),
                "history": reflection_history,
            }

        result["agent_trace"] = agent_trace
        result["elapsed_ms"] = round((time.time() - start_time) * 1000, 2)
        result["steps"] = memory.get_step_count()
        result["mode"] = "agent-native" if native else "agent-prompt"

        if self.config.DEBUG:
            logger.debug("Memory 摘要:\n%s", memory.summarize())

        self._update_knowledge(result, reviewer_id, product_id, product_name)
        self._apply_detail_level(result, reviewer_id)

        return result

    # ------------------------------------------------------------------
    # 知识积累集成
    # ------------------------------------------------------------------

    def _update_knowledge(
        self,
        result: Dict[str, Any],
        reviewer_id: Optional[str],
        product_id: Optional[str],
        product_name: str = "",
    ) -> None:
        """分析完成后自动更新知识积累（需 ENABLE_KNOWLEDGE=true）"""
        if not getattr(self.config, "ENABLE_KNOWLEDGE", False):
            return
        if not reviewer_id and not product_id:
            return

        try:
            from ..knowledge import KnowledgeManager
            km = KnowledgeManager(self.config.KNOWLEDGE_STORE_DIR)

            if reviewer_id:
                km.update_reviewer(reviewer_id, result)
                logger.debug("已更新评论者画像: %s", reviewer_id)

            if product_id:
                km.update_product(product_id, result, product_name=product_name)
                logger.debug("已更新商品画像: %s", product_id)
        except Exception as e:
            logger.warning("知识积累更新失败: %s", e)

    def _apply_detail_level(
        self, result: Dict[str, Any], reviewer_id: Optional[str]
    ) -> None:
        """根据评论者历史调整输出详细程度（需 ENABLE_KNOWLEDGE=true）"""
        if not getattr(self.config, "ENABLE_KNOWLEDGE", False):
            return
        if not reviewer_id:
            return

        try:
            from ..knowledge import KnowledgeManager
            km = KnowledgeManager(self.config.KNOWLEDGE_STORE_DIR)
            level = km.get_output_detail_level(reviewer_id)
            result["detail_level"] = level

            if level == "delta":
                delta_kws = km.get_delta_keywords(
                    reviewer_id, result.get("keywords", [])
                )
                result["delta_keywords"] = delta_kws
                result["delta_keywords_count"] = len(delta_kws)
        except Exception as e:
            logger.warning("详细程度调整失败: %s", e)

    # ------------------------------------------------------------------
    # 内层 ReAct 循环
    # ------------------------------------------------------------------

    def _run_react_loop(
        self,
        memory: AgentMemory,
        agent_trace: List[Dict[str, Any]],
        start_time: float,
        native: bool,
        is_retry: bool = False,
    ) -> Optional[str]:
        if native:
            return self._react_native(memory, agent_trace, start_time)
        else:
            return self._react_prompt_based(memory, agent_trace, start_time)

    _TOKEN_LIMIT_MARKER = "__token_limit_exceeded__"
    _MAX_TOOL_RESULT_CHARS = 2048

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(k in msg for k in (
            "maximum context length",
            "max_model_len",
            "context length",
            "too many tokens",
            "token limit",
        ))

    @staticmethod
    def _truncate_result(result_str: str, max_chars: int) -> str:
        """智能截断工具结果，确保 JSON 有效性。
        
        对于 keyword_extract 等工具的结果，优先截断 thinking 字段而保留核心数据。
        """
        if len(result_str) <= max_chars:
            return result_str
        
        try:
            data = json.loads(result_str)
            
            if isinstance(data, dict) and "data" in data:
                inner = data.get("data", {})
                if isinstance(inner, dict) and "thinking" in inner:
                    thinking = inner.get("thinking", "")
                    if len(thinking) > 200:
                        inner["thinking"] = thinking[:200] + "...(截断)"
                        truncated = json.dumps(data, ensure_ascii=False)
                        if len(truncated) <= max_chars:
                            return truncated
                        inner["thinking"] = "(已截断)"
                        return json.dumps(data, ensure_ascii=False)
            
            truncated = json.dumps(data, ensure_ascii=False)
            if len(truncated) <= max_chars:
                return truncated
            return result_str[:max_chars - 20] + '...(截断)"}'
            
        except (json.JSONDecodeError, TypeError):
            return result_str[:max_chars - 20] + '...(截断)"}'

    def _react_native(
        self,
        memory: AgentMemory,
        agent_trace: List[Dict[str, Any]],
        start_time: float,
    ) -> Optional[str]:
        tool_defs = self._get_tool_definitions()
        agent_summary: Optional[str] = None

        for step in range(self.max_steps):
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                logger.warning("Agent 超时 (%.1fs > %ds)", elapsed, self.timeout)
                break

            step_num = len(agent_trace) + 1
            logger.info("--- Agent Step %d (native) ---", step_num)

            try:
                llm_resp = self.llm_service.call_agent(
                    messages=memory.to_messages(),
                    tools=tool_defs,
                    tool_choice="auto",
                )
            except Exception as e:
                logger.exception("Agent LLM 调用失败")
                agent_trace.append({
                    "step": step_num, "type": "error",
                    "content": f"Agent LLM 调用失败: {e}",
                })
                if self._is_context_length_error(e):
                    agent_summary = self._TOKEN_LIMIT_MARKER
                break

            thought = llm_resp.content or ""

            if not llm_resp.has_tool_calls:
                fallback_calls = self._parse_tool_calls_from_text(thought)
                if fallback_calls:
                    logger.info(
                        "Native 模式检测到 <tool_call> 标签 (step %d), "
                        "回退到 prompt-based 执行", step_num,
                    )
                    memory.add_assistant_message(content=thought)

                    tool_names = []
                    observations = []
                    for call in fallback_calls:
                        tool_name = call["name"]
                        tool_names.append(tool_name)
                        tool_result = self._execute_tool(tool_name, call["arguments"])
                        result_str = json.dumps(
                            tool_result.model_dump(), ensure_ascii=False, default=str
                        )
                        result_str = self._truncate_result(result_str, self._MAX_TOOL_RESULT_CHARS)
                        observations.append(f"[工具 {tool_name} 结果]\n{result_str}")

                    observation = "\n\n".join(observations)
                    memory.add_user_message(
                        f"以上工具已执行完毕，以下是执行结果：\n\n{observation}\n\n"
                        f"请根据以上结果继续分析，或输出你的总结。"
                    )
                    agent_trace.append({
                        "step": step_num,
                        "type": "thought_and_action",
                        "thought": self._strip_tool_call_tags(thought),
                        "actions": tool_names,
                    })
                    continue

                memory.add_assistant_message(content=thought)
                logger.info("Agent 分析完成 (step %d)", step_num)
                agent_summary = thought
                agent_trace.append({
                    "step": step_num,
                    "type": "final_summary",
                    "content": thought,
                })
                break

            tool_calls_raw = llm_resp.tool_calls
            tool_names = [tc["function"]["name"] for tc in tool_calls_raw]
            memory.add_assistant_message(content=llm_resp.content, tool_calls=tool_calls_raw)

            for tc in tool_calls_raw:
                tool_name = tc["function"]["name"]
                try:
                    arguments = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}
                    logger.warning("工具 %s 参数解析失败: %s", tool_name, tc["function"]["arguments"])

                tool_result = self._execute_tool(tool_name, arguments)
                result_str = json.dumps(tool_result.model_dump(), ensure_ascii=False, default=str)
                result_str = self._truncate_result(result_str, self._MAX_TOOL_RESULT_CHARS)
                memory.add_tool_result(
                    tool_call_id=tc["id"], tool_name=tool_name, result=result_str,
                )

            if thought:
                logger.info("Thought: %s", thought[:100] + ("..." if len(thought) > 100 else ""))

            agent_trace.append({
                "step": step_num,
                "type": "thought_and_action",
                "thought": thought,
                "actions": tool_names,
            })

        return agent_summary

    def _react_prompt_based(
        self,
        memory: AgentMemory,
        agent_trace: List[Dict[str, Any]],
        start_time: float,
    ) -> Optional[str]:
        agent_summary: Optional[str] = None

        for step in range(self.max_steps):
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                logger.warning("Agent 超时 (%.1fs > %ds)", elapsed, self.timeout)
                break

            step_num = len(agent_trace) + 1
            logger.info("--- Agent Step %d (prompt-based) ---", step_num)

            try:
                llm_resp = self.llm_service.call_agent(
                    messages=memory.to_messages(),
                )
            except Exception as e:
                logger.exception("Agent LLM 调用失败")
                agent_trace.append({
                    "step": step_num, "type": "error",
                    "content": f"Agent LLM 调用失败: {e}",
                })
                if self._is_context_length_error(e):
                    agent_summary = self._TOKEN_LIMIT_MARKER
                break

            content = llm_resp.content or ""
            parsed_calls = self._parse_tool_calls_from_text(content)

            if not parsed_calls:
                memory.add_assistant_message(content=content)
                logger.info("Agent 分析完成 (step %d)", step_num)
                agent_summary = content
                agent_trace.append({
                    "step": step_num,
                    "type": "final_summary",
                    "content": content,
                })
                break

            memory.add_assistant_message(content=content)
            thought_text = self._strip_tool_call_tags(content)

            tool_names = []
            observation_parts = []
            for call in parsed_calls:
                tool_name = call["name"]
                tool_names.append(tool_name)
                arguments = call["arguments"]

                tool_result = self._execute_tool(tool_name, arguments)
                result_str = json.dumps(tool_result.model_dump(), ensure_ascii=False, default=str)
                result_str = self._truncate_result(result_str, self._MAX_TOOL_RESULT_CHARS)
                observation_parts.append(f"[工具 {tool_name} 结果]\n{result_str}")

            observation = "\n\n".join(observation_parts)
            memory.add_user_message(
                f"以上工具已执行完毕，以下是执行结果：\n\n{observation}\n\n"
                f"请根据以上结果继续分析，或输出你的总结。"
            )
            agent_trace.append({
                "step": step_num,
                "type": "thought_and_action",
                "thought": thought_text,
                "actions": tool_names,
            })

        return agent_summary

    # ------------------------------------------------------------------
    # 代码级反思
    # ------------------------------------------------------------------

    def _get_min_keywords_for_text(self, text: str) -> int:
        length = len(text)
        if length < 20:
            return self.config.REFLECTION_MIN_KEYWORDS_SHORT
        elif length < 60:
            return self.config.REFLECTION_MIN_KEYWORDS_MEDIUM
        elif length < 120:
            return self.config.REFLECTION_MIN_KEYWORDS_LONG
        else:
            return self.config.REFLECTION_MIN_KEYWORDS_XLONG

    @staticmethod
    def _keyword_in_original(keyword: str, original_text: str) -> bool:
        return keyword in original_text

    def _filter_keywords_by_score(self, keywords: List[Dict[str, Any]]) -> tuple:
        threshold = self.config.REFLECTION_SCORE_THRESHOLD
        kept, removed = [], []
        for kw in keywords:
            (kept if kw.get("score", 1.0) >= threshold else removed).append(kw)
        return kept, removed

    def _code_level_reflection(
        self,
        result: Dict[str, Any],
        memory: AgentMemory,
        agent_trace: List[Dict[str, Any]],
        start_time: float,
    ) -> tuple:
        original_text = result.get("original_text", "")
        min_keywords = self._get_min_keywords_for_text(original_text)
        max_rounds = self.config.REFLECTION_MAX_ROUNDS
        threshold = self.config.REFLECTION_SCORE_THRESHOLD
        reflection_history: List[Dict[str, Any]] = []

        keywords = result.get("keywords", [])
        kept_keywords, removed_by_score = self._filter_keywords_by_score(keywords)

        if removed_by_score:
            logger.info(
                "Score 过滤: 移除 %d 个低分关键词 (阈值=%s)",
                len(removed_by_score), threshold,
            )

        result["keywords"] = kept_keywords

        if len(kept_keywords) >= min_keywords:
            logger.info(
                "反思通过 (代码级): %d 个关键词 >= 最低要求 %d",
                len(kept_keywords), min_keywords,
            )
            reflection_history.append({
                "passed": True,
                "type": "code_level",
                "keywords_count": len(kept_keywords),
                "min_required": min_keywords,
                "removed_by_score": [kw.get("keyword", "") for kw in removed_by_score],
                "summary": "关键词数量达标，通过代码级检查",
            })
            return result, reflection_history

        logger.info(
            "关键词不足: %d/%d，启动 LLM 反思补充",
            len(kept_keywords), min_keywords,
        )

        existing_kw_set = {kw.get("keyword", "") for kw in kept_keywords}

        for round_idx in range(max_rounds):
            round_start = time.time()

            try:
                ref = self._reflector.reflect(
                    original_text, kept_keywords, result.get("sentiment", {})
                )
            except Exception as e:
                logger.warning("反思第 %d 轮异常: %s", round_idx + 1, e)
                reflection_history.append({
                    "passed": True,
                    "type": "llm_error",
                    "summary": f"反思异常: {e}",
                    "elapsed_ms": round((time.time() - round_start) * 1000, 2),
                })
                break

            if ref.remove_keywords:
                remove_set = set(ref.remove_keywords)
                kept_keywords = [
                    kw for kw in kept_keywords
                    if kw.get("keyword", "") not in remove_set
                ]
                existing_kw_set = {kw.get("keyword", "") for kw in kept_keywords}

            new_valid_keywords = []
            rejected_keywords = []
            for add_kw in (ref.add_keywords or []):
                kw_text = add_kw.get("keyword", "")
                kw_score = add_kw.get("score", 0.0)

                if kw_text in existing_kw_set:
                    continue
                if kw_score < threshold:
                    rejected_keywords.append(f"{kw_text}(score={kw_score}<{threshold})")
                    continue
                if not self._keyword_in_original(kw_text, original_text):
                    rejected_keywords.append(f"{kw_text}(不在原文中)")
                    continue

                new_valid_keywords.append(add_kw)
                existing_kw_set.add(kw_text)

            if new_valid_keywords:
                kept_keywords.extend(new_valid_keywords)

            reflection_history.append({
                "passed": len(kept_keywords) >= min_keywords,
                "type": "llm_supplement",
                "round": round_idx + 1,
                "keywords_count": len(kept_keywords),
                "min_required": min_keywords,
                "added": [kw.get("keyword", "") for kw in new_valid_keywords],
                "rejected": rejected_keywords,
                "removed": list(ref.remove_keywords or []),
                "summary": ref.summary or "",
                "elapsed_ms": round((time.time() - round_start) * 1000, 2),
            })

            agent_trace.append({
                "step": len(agent_trace) + 1,
                "type": "reflection_supplement",
                "content": (
                    f"第 {round_idx + 1} 轮反思: "
                    f"新增={[kw.get('keyword','') for kw in new_valid_keywords]}, "
                    f"拒绝={rejected_keywords}, "
                    f"当前数量={len(kept_keywords)}/{min_keywords}"
                ),
            })

            if len(kept_keywords) >= min_keywords:
                break
            if not new_valid_keywords:
                break

        kept_keywords.sort(key=lambda kw: kw.get("score", 0), reverse=True)
        result["keywords"] = kept_keywords

        return result, reflection_history

    # ------------------------------------------------------------------
    # 从 Memory 工具结果中组装最终输出
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_result_from_memory(
        memory: AgentMemory, original_comment: str
    ) -> Dict[str, Any]:
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

    # ------------------------------------------------------------------
    # 批量分析
    # ------------------------------------------------------------------

    def run_batch(
        self,
        comments: List[str],
        use_fast_path: bool = True,
    ) -> List[Dict[str, Any]]:
        results = []
        for i, comment in enumerate(comments):
            logger.info("批量分析 [%d/%d]", i + 1, len(comments))
            result = self.run(comment, use_fast_path=use_fast_path)
            result["batch_index"] = i
            results.append(result)
        return results

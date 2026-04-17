"""
ReAct Agent 主循环

支持两种工具调用模式：
1. 原生 Function Calling（vLLM 需启动时加 --enable-auto-tool-choice --tool-call-parser hermes）
2. Prompt-based 模式（不依赖服务端 tool calling，通过解析 <tool_call> 标签实现）

首次调用时自动探测，探测失败则降级为 prompt-based 模式。

架构职责解耦：
- Agent LLM：推理规划 + 工具调度（不负责格式化输出）
- Tool LLM：实际执行关键词提取、情感分析（使用微调 Prompt）
- 代码层：从工具结果组装结构化输出，管理反思循环
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..config import AgentConfig
from ..tools.base_tool import BaseTool, ToolResult
from ..tools.preprocess_tool import PreprocessTool
from ..tools.keyword_extract_tool import KeywordExtractTool
from ..tools.jieba_extract_tool import JiebaExtractTool
from ..tools.validate_tool import ValidateTool
from ..tools.sentiment_tool import SentimentTool
from .memory import AgentMemory
from .prompts import (
    AGENT_SYSTEM_PROMPT,
    AGENT_SYSTEM_PROMPT_WITH_TOOLS,
    USER_REQUEST_TEMPLATE,
    build_tool_descriptions_for_prompt,
)
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


class ReviewAnalysisAgent:
    """
    中文电商评论分析 Agent

    采用 ReAct 范式，通过 Agent LLM 进行规划和工具调度，
    自动完成评论的预处理、关键词提取、质量校验和情感分析。
    支持原生 Function Calling 与 prompt-based 两种模式自动切换。

    反思机制集成到主循环中：
    Agent ReAct 循环完成 → 组装结果 → 反思审查 →
    若不通过则将反馈注入 Memory 重新进入 ReAct 循环。
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()

        self._agent_client = OpenAI(
            base_url=self.config.AGENT_LLM_BASE_URL,
            api_key=self.config.AGENT_LLM_API_KEY,
        )

        self.tools: Dict[str, BaseTool] = {}
        self._init_tools()

        self.max_steps = self.config.AGENT_MAX_STEPS
        self.timeout = self.config.AGENT_TIMEOUT

        self._reflector: Optional[ResultReflector] = None
        if self.config.ENABLE_REFLECTION:
            self._reflector = ResultReflector(self.config)
            logger.info("反思器已启用 (max_rounds=%d)", self.config.REFLECTION_MAX_ROUNDS)

        self._native_tool_calling: Optional[bool] = None

    def _init_tools(self) -> None:
        tool_instances: List[BaseTool] = [
            PreprocessTool(),
            KeywordExtractTool(self.config),
            JiebaExtractTool(),
            ValidateTool(),
            SentimentTool(self.config),
        ]
        for tool in tool_instances:
            self.tools[tool.name] = tool
        logger.info(f"Agent 初始化完成，已加载 {len(self.tools)} 个工具: {list(self.tools.keys())}")

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self.tools.values()]

    # ------------------------------------------------------------------
    # 原生 Function Calling 探测
    # ------------------------------------------------------------------

    def _probe_native_tool_calling(self) -> bool:
        """
        向 vLLM 发一个轻量请求探测是否支持原生 tool calling。
        使用 httpx 直接发请求并设置严格超时，避免 vLLM 处理 tool_choice
        时卡死导致整个程序挂起。
        """
        import httpx

        probe_timeout = 10  # 秒
        url = f"{self.config.AGENT_LLM_BASE_URL}/chat/completions"
        payload = {
            "model": self.config.AGENT_LLM_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "_probe",
                    "description": "probe",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
            "tool_choice": "auto",
            "max_tokens": 1,
        }

        try:
            resp = httpx.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.config.AGENT_LLM_API_KEY}"},
                timeout=probe_timeout,
            )
            if resp.status_code == 200:
                logger.info("探测成功：vLLM 支持原生 Function Calling")
                return True
            else:
                body = resp.text[:300]
                logger.info(f"探测返回 {resp.status_code}，降级为 prompt-based: {body}")
                return False
        except httpx.TimeoutException:
            logger.warning(
                f"探测 Function Calling 超时 ({probe_timeout}s)，"
                "vLLM 可能不兼容当前模型的 tool calling，降级为 prompt-based"
            )
            return False
        except Exception as e:
            logger.warning(f"探测 Function Calling 异常，降级为 prompt-based: {e}")
            return False

    def _ensure_mode_detected(self) -> None:
        if self._native_tool_calling is not None:
            return

        mode_cfg = self.config.AGENT_TOOL_CALLING_MODE.lower()
        if mode_cfg == "native":
            self._native_tool_calling = True
        elif mode_cfg == "prompt":
            self._native_tool_calling = False
        else:
            self._native_tool_calling = self._probe_native_tool_calling()

        mode_name = "原生 Function Calling" if self._native_tool_calling else "Prompt-based"
        logger.info(f"Agent 工具调用模式: {mode_name} (配置={mode_cfg})")

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        tool = self.tools.get(tool_name)
        if tool is None:
            return ToolResult(success=False, error=f"未知工具: {tool_name}")

        try:
            start = time.time()
            result = tool.execute(**arguments)
            elapsed = time.time() - start

            if elapsed > self.config.TOOL_TIMEOUT:
                logger.warning(f"工具 {tool_name} 执行超时: {elapsed:.1f}s > {self.config.TOOL_TIMEOUT}s")

            logger.info(
                f"工具 [{tool_name}] 执行{'成功' if result.success else '失败'} "
                f"(耗时 {elapsed:.2f}s)"
            )
            return result
        except Exception as e:
            logger.exception(f"工具 {tool_name} 执行异常")
            return ToolResult(success=False, error=str(e))

    # ------------------------------------------------------------------
    # Prompt-based 工具调用解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls_from_text(content: str) -> List[Dict[str, Any]]:
        """
        从模型回复文本中解析 <tool_call>...</tool_call> 标签。

        Returns:
            解析到的工具调用列表，每项包含 name 和 arguments。
            如果没有 <tool_call> 标签则返回空列表。
        """
        calls = []
        for match in _TOOL_CALL_PATTERN.finditer(content):
            try:
                payload = json.loads(match.group(1))
                name = payload.get("name", "")
                arguments = payload.get("arguments", {})
                if name:
                    calls.append({"name": name, "arguments": arguments})
            except json.JSONDecodeError:
                logger.warning(f"解析 <tool_call> JSON 失败: {match.group(1)[:200]}")
        return calls

    @staticmethod
    def _strip_tool_call_tags(content: str) -> str:
        """移除文本中的 <tool_call> 标签，保留其余思考内容"""
        return _TOOL_CALL_PATTERN.sub("", content).strip()

    # ------------------------------------------------------------------
    # 快速路径
    # ------------------------------------------------------------------

    def run_fast(self, comment: str) -> Dict[str, Any]:
        """
        快速路径：对简单的单条评论直接按固定管线执行，
        不经过 Agent LLM 规划，减少延迟和 token 消耗。
        """
        start = time.time()
        result: Dict[str, Any] = {"original_text": comment}

        prep = self._execute_tool("text_preprocess", {"text": comment})
        if prep.success:
            cleaned = prep.data["cleaned_text"]
        else:
            cleaned = comment.strip()
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
            val_result = self._execute_tool("validate_keywords", {
                "keywords": keywords,
                "original_text": cleaned,
            })
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

    def run(self, comment: str, use_fast_path: bool = False) -> Dict[str, Any]:
        if use_fast_path:
            return self.run_fast(comment)

        self._ensure_mode_detected()

        start_time = time.time()
        agent_trace: List[Dict[str, Any]] = []

        # --- 构建初始 Memory ---
        if self._native_tool_calling:
            memory = AgentMemory(system_prompt=AGENT_SYSTEM_PROMPT)
        else:
            tool_desc = build_tool_descriptions_for_prompt(self.tools)
            system_prompt = AGENT_SYSTEM_PROMPT_WITH_TOOLS.replace(
                "{tool_descriptions}", tool_desc
            )
            memory = AgentMemory(system_prompt=system_prompt)

        user_request = USER_REQUEST_TEMPLATE.format(comment=comment)
        memory.add_user_message(user_request)

        # --- 内层 ReAct 循环 ---
        agent_summary = self._run_react_loop(
            memory, agent_trace, start_time
        )

        # --- Token 超限：自动降级为 fast 模式 ---
        if agent_summary == self._TOKEN_LIMIT_MARKER:
            logger.warning("Agent 模式 token 超限，降级为 fast 模式执行")
            fast_result = self.run_fast(comment)
            fast_result["mode"] = "agent-native-fallback-fast"
            fast_result["warnings"] = [
                "评论文本较长，Agent 模式因 token 超限自动降级为 fast 管线模式"
            ]
            fast_result["agent_trace"] = agent_trace
            return fast_result

        # --- 从 Memory 组装结构化结果 ---
        result = self._assemble_result_from_memory(memory, comment)
        if agent_summary:
            result["agent_summary"] = agent_summary

        # --- 代码级反思：score 过滤 + LLM 辅助补充 + 原文校验 ---
        # 反思策略：
        #   1. 先用 score 阈值过滤掉不合格的关键词（代码级，确定性）
        #   2. 检查过滤后数量是否达标（代码级）
        #   3. 如果达标 → 直接通过
        #   4. 如果不达标 → 调用 LLM 反思器尝试补充，
        #      但补充的词必须通过原文对齐校验
        #   5. 最多重试 REFLECTION_MAX_ROUNDS 次，
        #      如果补充无效（没有新的有效关键词）→ 接受当前结果
        # TODO: 阈值和最低数量要求可根据实际效果调整
        reflection_history: List[Dict[str, Any]] = []
        if self._reflector and result.get("analysis_complete"):
            result, reflection_history = self._code_level_reflection(
                result, memory, agent_trace, start_time
            )

        # --- 最终组装 ---
        if reflection_history:
            result["reflection"] = {
                "total_rounds": len(reflection_history),
                "final_passed": reflection_history[-1].get("passed", True),
                "history": reflection_history,
            }

        result["agent_trace"] = agent_trace
        result["elapsed_ms"] = round((time.time() - start_time) * 1000, 2)
        result["steps"] = memory.get_step_count()
        result["mode"] = "agent-native" if self._native_tool_calling else "agent-prompt"

        if self.config.DEBUG:
            logger.debug(f"Memory 摘要:\n{memory.summarize()}")

        return result

    # ------------------------------------------------------------------
    # 内层 ReAct 循环
    # ------------------------------------------------------------------

    def _run_react_loop(
        self,
        memory: AgentMemory,
        agent_trace: List[Dict[str, Any]],
        start_time: float,
        is_retry: bool = False,
    ) -> Optional[str]:
        """
        执行一轮完整的 ReAct 循环（Thought → Action → Observation 重复）。
        将 Agent LLM 每步的推理内容记录到 agent_trace 中。

        Args:
            memory: Agent 工作记忆（跨反思轮次共享）
            agent_trace: 推理轨迹记录列表
            start_time: 整体开始时间（用于超时检测）
            is_retry: 是否为反思重试轮次

        Returns:
            Agent 最终的自然语言总结（可能为 None）
        """
        if self._native_tool_calling:
            return self._react_native(memory, agent_trace, start_time)
        else:
            return self._react_prompt_based(memory, agent_trace, start_time)

    _TOKEN_LIMIT_MARKER = "__token_limit_exceeded__"
    _MAX_TOOL_RESULT_CHARS = 600

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        """判断异常是否属于 token/context length 超限"""
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
        if len(result_str) <= max_chars:
            return result_str
        return result_str[:max_chars] + '..."}'

    def _react_native(
        self,
        memory: AgentMemory,
        agent_trace: List[Dict[str, Any]],
        start_time: float,
    ) -> Optional[str]:
        """原生 Function Calling 模式的 ReAct 循环"""
        tool_defs = self._get_tool_definitions()
        agent_summary: Optional[str] = None

        for step in range(self.max_steps):
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                logger.warning(f"Agent 超时 ({elapsed:.1f}s > {self.timeout}s)")
                break

            step_num = len(agent_trace) + 1
            logger.info(f"--- Agent Step {step_num} (native) ---")

            try:
                response = self._agent_client.chat.completions.create(
                    model=self.config.AGENT_LLM_MODEL,
                    messages=memory.to_messages(),
                    tools=tool_defs,
                    tool_choice="auto",
                    max_tokens=self.config.AGENT_LLM_MAX_TOKENS,
                    temperature=self.config.AGENT_LLM_TEMPERATURE,
                )
            except Exception as e:
                logger.exception("Agent LLM 调用失败")
                err_msg = f"Agent LLM 调用失败: {e}"
                agent_trace.append({
                    "step": step_num, "type": "error", "content": err_msg,
                })
                if self._is_context_length_error(e):
                    logger.warning(
                        "Token 超限，将自动降级为 fast 模式重试"
                    )
                    agent_summary = self._TOKEN_LIMIT_MARKER
                break

            message = response.choices[0].message
            thought = message.content or ""

            if not message.tool_calls:
                fallback_calls = self._parse_tool_calls_from_text(thought)
                if fallback_calls:
                    logger.info(
                        f"Native 模式检测到 <tool_call> 标签 (step {step_num}), "
                        f"回退到 prompt-based 执行"
                    )
                    memory.add_assistant_message(content=thought)

                    tool_names = []
                    observations = []
                    for call in fallback_calls:
                        tool_name = call["name"]
                        tool_names.append(tool_name)
                        arguments = call["arguments"]
                        tool_result = self._execute_tool(tool_name, arguments)
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
                logger.info(f"Agent 分析完成 (step {step_num})")
                agent_summary = thought

                agent_trace.append({
                    "step": step_num,
                    "type": "final_summary",
                    "content": thought,
                })
                break

            tool_names = [tc.function.name for tc in message.tool_calls]
            tool_calls_raw = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
            memory.add_assistant_message(content=message.content, tool_calls=tool_calls_raw)

            for tc in message.tool_calls:
                tool_name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                    logger.warning(f"工具 {tool_name} 参数解析失败: {tc.function.arguments}")

                tool_result = self._execute_tool(tool_name, arguments)
                result_str = json.dumps(tool_result.model_dump(), ensure_ascii=False, default=str)
                result_str = self._truncate_result(result_str, self._MAX_TOOL_RESULT_CHARS)
                memory.add_tool_result(tool_call_id=tc.id, tool_name=tool_name, result=result_str)

            if thought:
                logger.info(f"Thought: {thought[:100]}{'...' if len(thought) > 100 else ''}")

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
        """Prompt-based 模式的 ReAct 循环"""
        agent_summary: Optional[str] = None

        for step in range(self.max_steps):
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                logger.warning(f"Agent 超时 ({elapsed:.1f}s > {self.timeout}s)")
                break

            step_num = len(agent_trace) + 1
            logger.info(f"--- Agent Step {step_num} (prompt-based) ---")

            try:
                response = self._agent_client.chat.completions.create(
                    model=self.config.AGENT_LLM_MODEL,
                    messages=memory.to_messages(),
                    max_tokens=self.config.AGENT_LLM_MAX_TOKENS,
                    temperature=self.config.AGENT_LLM_TEMPERATURE,
                )
            except Exception as e:
                logger.exception("Agent LLM 调用失败")
                err_msg = f"Agent LLM 调用失败: {e}"
                agent_trace.append({
                    "step": step_num, "type": "error", "content": err_msg,
                })
                if self._is_context_length_error(e):
                    logger.warning("Token 超限，将自动降级为 fast 模式重试")
                    agent_summary = self._TOKEN_LIMIT_MARKER
                break

            content = response.choices[0].message.content or ""
            parsed_calls = self._parse_tool_calls_from_text(content)

            if not parsed_calls:
                memory.add_assistant_message(content=content)
                logger.info(f"Agent 分析完成 (step {step_num})")
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
                logger.info(f"Prompt-based 调用工具: {tool_name}({json.dumps(arguments, ensure_ascii=False)[:100]})")

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
    # 代码级反思：score 过滤 + LLM 辅助补充 + 原文校验 + 智能终止
    # ------------------------------------------------------------------

    def _get_min_keywords_for_text(self, text: str) -> int:
        """根据原文长度返回最低关键词数量要求"""
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
        """检查关键词是否在原文中出现（精确子串匹配）"""
        return keyword in original_text

    def _filter_keywords_by_score(
        self, keywords: List[Dict[str, Any]]
    ) -> tuple:
        """
        用 score 阈值过滤关键词，返回 (合格列表, 被过滤列表)。
        没有 score 的关键词默认保留。
        """
        threshold = self.config.REFLECTION_SCORE_THRESHOLD
        kept = []
        removed = []
        for kw in keywords:
            score = kw.get("score", 1.0)
            if score >= threshold:
                kept.append(kw)
            else:
                removed.append(kw)
        return kept, removed

    def _code_level_reflection(
        self,
        result: Dict[str, Any],
        memory: AgentMemory,
        agent_trace: List[Dict[str, Any]],
        start_time: float,
    ) -> tuple:
        """
        代码级反思主流程，返回 (updated_result, reflection_history)。

        流程：
        1. score 阈值过滤不合格关键词
        2. 代码级检查数量是否达标
        3. 如果达标 → 直接通过
        4. 如果不达标 → 调用 LLM 反思器尝试补充
        5. LLM 补充的词必须通过原文对齐校验
        6. 如果补充无效（没有新增有效关键词）→ 接受当前结果
        7. 最多重试 REFLECTION_MAX_ROUNDS 次
        """
        original_text = result.get("original_text", "")
        min_keywords = self._get_min_keywords_for_text(original_text)
        max_rounds = self.config.REFLECTION_MAX_ROUNDS
        threshold = self.config.REFLECTION_SCORE_THRESHOLD
        reflection_history: List[Dict[str, Any]] = []

        # Step 1: score 过滤
        keywords = result.get("keywords", [])
        kept_keywords, removed_by_score = self._filter_keywords_by_score(keywords)

        if removed_by_score:
            removed_names = [kw.get("keyword", "") for kw in removed_by_score]
            logger.info(
                f"Score 过滤: 移除 {len(removed_by_score)} 个低分关键词 "
                f"(阈值={threshold}): {removed_names}"
            )

        result["keywords"] = kept_keywords

        # Step 2: 检查数量是否已达标
        if len(kept_keywords) >= min_keywords:
            logger.info(
                f"反思通过 (代码级): {len(kept_keywords)} 个关键词 >= 最低要求 {min_keywords}"
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

        # Step 3: 数量不足，用 LLM 反思器尝试补充
        logger.info(
            f"关键词不足: {len(kept_keywords)}/{min_keywords}，启动 LLM 反思补充"
        )

        existing_kw_set = {kw.get("keyword", "") for kw in kept_keywords}

        for round_idx in range(max_rounds):
            round_start = time.time()

            try:
                ref = self._reflector.reflect(
                    original_text, kept_keywords, result.get("sentiment", {})
                )
            except Exception as e:
                logger.warning(f"反思第 {round_idx + 1} 轮异常: {e}")
                reflection_history.append({
                    "passed": True,
                    "type": "llm_error",
                    "summary": f"反思异常: {e}",
                    "elapsed_ms": round((time.time() - round_start) * 1000, 2),
                })
                break

            # 处理 LLM 返回的 remove_keywords：只移除存在于当前列表的
            if ref.remove_keywords:
                remove_set = set(ref.remove_keywords)
                before_count = len(kept_keywords)
                kept_keywords = [
                    kw for kw in kept_keywords
                    if kw.get("keyword", "") not in remove_set
                ]
                removed_count = before_count - len(kept_keywords)
                if removed_count > 0:
                    logger.info(f"反思第 {round_idx + 1} 轮移除 {removed_count} 个关键词")
                existing_kw_set = {kw.get("keyword", "") for kw in kept_keywords}

            # 处理 LLM 返回的 add_keywords：必须通过原文对齐 + score 阈值
            new_valid_keywords = []
            rejected_keywords = []
            for add_kw in (ref.add_keywords or []):
                kw_text = add_kw.get("keyword", "")
                kw_score = add_kw.get("score", 0.0)

                if kw_text in existing_kw_set:
                    continue

                if kw_score < threshold:
                    rejected_keywords.append(
                        f"{kw_text}(score={kw_score}<{threshold})"
                    )
                    continue

                if not self._keyword_in_original(kw_text, original_text):
                    rejected_keywords.append(f"{kw_text}(不在原文中)")
                    continue

                new_valid_keywords.append(add_kw)
                existing_kw_set.add(kw_text)

            if rejected_keywords:
                logger.info(
                    f"反思第 {round_idx + 1} 轮拒绝无效补充: {rejected_keywords}"
                )

            if new_valid_keywords:
                kept_keywords.extend(new_valid_keywords)
                new_names = [kw.get("keyword", "") for kw in new_valid_keywords]
                logger.info(
                    f"反思第 {round_idx + 1} 轮成功补充 {len(new_valid_keywords)} 个: "
                    f"{new_names}"
                )

            round_record = {
                "passed": len(kept_keywords) >= min_keywords,
                "type": "llm_supplement",
                "round": round_idx + 1,
                "keywords_count": len(kept_keywords),
                "min_required": min_keywords,
                "added": [kw.get("keyword", "") for kw in new_valid_keywords],
                "rejected": rejected_keywords,
                "removed": list(ref.remove_keywords or []),
                "llm_issues": [
                    {"type": i.get("type", ""), "detail": i.get("detail", "")}
                    for i in (ref.issues or [])
                ],
                "summary": ref.summary or "",
                "elapsed_ms": round((time.time() - round_start) * 1000, 2),
            }
            reflection_history.append(round_record)

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

            # 检查是否达标
            if len(kept_keywords) >= min_keywords:
                logger.info(
                    f"反思通过 (第 {round_idx + 1} 轮): "
                    f"{len(kept_keywords)} >= {min_keywords}"
                )
                break

            # 如果本轮没有任何有效新增，继续循环也不会有新的结果，直接终止
            if not new_valid_keywords:
                logger.info(
                    f"反思第 {round_idx + 1} 轮无有效新增，接受当前结果 "
                    f"({len(kept_keywords)} 个关键词)"
                )
                break

        # 最终排序：按 score 降序
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
        """
        遍历 Memory 中所有工具返回消息，提取各工具的结构化结果，
        组装为最终的分析输出。

        职责解耦：Agent LLM 只负责推理规划和工具调度，
        Tool LLM 负责实际提取，代码负责结果组装。

        支持两种工具结果存储方式：
        - Native 模式：role=tool 消息，name 字段标识工具
        - Prompt-based / Fallback：role=user 消息，
          格式为 "[工具 xxx 结果]\\n{json}"
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
                except json.JSONDecodeError:
                    pass

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
        # Agent LLM 构造工具参数时可能引入重复关键词，在代码层做最终去重
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
            logger.info(f"批量分析 [{i + 1}/{len(comments)}]")
            result = self.run(comment, use_fast_path=use_fast_path)
            result["batch_index"] = i
            results.append(result)
        return results

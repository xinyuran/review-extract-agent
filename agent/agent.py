"""
ReAct Agent 主循环（四层架构版）

架构职责解耦（四层）：
- Skill 层：skills/ 目录下的 SKILL.md 文件，定义各阶段的 prompt
- Agent 层（本模块）：入口协调器，串联 ReAct 循环、快速路径、反思、知识积累
- LLM Service 层：统一管理 Agent LLM / Tool LLM 调用
- Tool 层：纯计算工具（preprocess / jieba / validate）

Agent 内部组件：
- ReactLoop (react_loop.py)：ReAct 循环执行器（native / prompt-based）
- FastPathExecutor (fast_path.py)：管线式快速分析
- assemble_result_from_memory (result_assembler.py)：从 Memory 组装结果
"""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from ..api.metrics import TOOL_CALLS

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
from .react_loop import ReactLoop, TOKEN_LIMIT_MARKER
from .fast_path import FastPathExecutor
from .result_assembler import assemble_result_from_memory

logger = logging.getLogger(__name__)

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

        self._fast_path = FastPathExecutor(self._execute_tool)
        self._react_loop = ReactLoop(
            llm_service=self.llm_service,
            execute_tool=self._execute_tool,
            tool_definitions=self._get_tool_definitions(),
            max_steps=self.max_steps,
            timeout=self.timeout,
        )

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

        self._keyword_tool: Optional[KeywordExtractTool] = None
        self._sentiment_tool: Optional[SentimentTool] = None
        if self._backend_mode != "offline":
            self._keyword_tool = KeywordExtractTool(self.config, llm_service=self.llm_service)
            self._sentiment_tool = SentimentTool(self.config, llm_service=self.llm_service)
            for t in (self._keyword_tool, self._sentiment_tool):
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
        """委托给 KeywordExtractTool.execute，避免重复实现"""
        return self._keyword_tool.execute(
            text=text,
            max_keywords=arguments.get("max_keywords", self.config.N),
        )

    def _execute_sentiment_analyze(self, text: str) -> ToolResult:
        """委托给 SentimentTool.execute，避免重复实现"""
        return self._sentiment_tool.execute(text=text)

    # ------------------------------------------------------------------
    # 统一工具执行入口
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        if tool_name in _LLM_POWERED_TOOLS:
            result = self._execute_llm_tool(tool_name, arguments)
            TOOL_CALLS.labels(tool_name=tool_name, success=str(result.success)).inc()
            return result

        tool = self.tools.get(tool_name)
        if tool is None:
            TOOL_CALLS.labels(tool_name=tool_name, success="False").inc()
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
            TOOL_CALLS.labels(tool_name=tool_name, success=str(result.success)).inc()
            return result
        except Exception as e:
            logger.exception("工具 %s 执行异常", tool_name)
            TOOL_CALLS.labels(tool_name=tool_name, success="False").inc()
            return ToolResult(success=False, error=str(e))

    # ------------------------------------------------------------------
    # 快速路径（委托给 FastPathExecutor）
    # ------------------------------------------------------------------

    def run_offline(self, comment: str) -> Dict[str, Any]:
        return self._fast_path.run_offline(comment)

    def run_fast(self, comment: str) -> Dict[str, Any]:
        if self._backend_mode == "offline":
            return self._fast_path.run_offline(comment)
        return self._fast_path.run_fast(comment)

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
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        trace_id = trace_id or str(uuid.uuid4())
        logger.info("[%s] 开始分析评论 (len=%d)", trace_id, len(comment))

        if self._backend_mode == "offline":
            result = self.run_offline(comment)
            result["trace_id"] = trace_id
            self._update_knowledge(result, reviewer_id, product_id, product_name)
            return result

        if use_fast_path:
            result = self.run_fast(comment)
            result["trace_id"] = trace_id
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

        agent_summary = self._react_loop.run(memory, agent_trace, start_time, native)

        if agent_summary == TOKEN_LIMIT_MARKER:
            logger.warning("[%s] Agent 模式 token 超限，降级为 fast 模式执行", trace_id)
            fast_result = self.run_fast(comment)
            fast_result["trace_id"] = trace_id
            fast_result["mode"] = "agent-native-fallback-fast"
            fast_result["warnings"] = [
                "评论文本较长，Agent 模式因 token 超限自动降级为 fast 管线模式"
            ]
            fast_result["agent_trace"] = agent_trace
            return fast_result

        result = assemble_result_from_memory(memory, comment)
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

        result["trace_id"] = trace_id
        result["agent_trace"] = agent_trace
        result["elapsed_ms"] = round((time.time() - start_time) * 1000, 2)
        result["steps"] = memory.get_step_count()
        result["mode"] = "agent-native" if native else "agent-prompt"

        logger.info("[%s] 分析完成 (elapsed=%.0fms)", trace_id, result["elapsed_ms"])
        if self.config.DEBUG:
            logger.debug("Memory 摘要:\n%s", memory.summarize())

        self._update_knowledge(result, reviewer_id, product_id, product_name)
        self._apply_detail_level(result, reviewer_id)

        return result

    # ------------------------------------------------------------------
    # 流式分析（SSE）
    # ------------------------------------------------------------------

    def run_stream(
        self,
        comment: str,
        trace_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        product_id: Optional[str] = None,
        product_name: str = "",
    ):
        """
        流式分析评论，yield 结构化事件字典。

        仅支持 Agent (ReAct) 模式。yield 事件包括：
        - start: 分析开始
        - token: LLM 生成的 token
        - step_start / thought / tool_call / tool_result: ReAct 步骤
        - final_summary: Agent 最终总结
        - reflection_start / reflection_done: 反思阶段（启用反思时）
        - result: 组装后的完整结果
        - error: 错误信息
        - done: 流结束
        """
        trace_id = trace_id or str(uuid.uuid4())
        yield {"type": "start", "trace_id": trace_id}

        if self._backend_mode == "offline":
            for event in self._fast_path.run_offline_stream(comment):
                if event.get("type") == "result":
                    event["data"]["trace_id"] = trace_id
                yield event
            yield {"type": "done"}
            return

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

        agent_summary = None
        for event in self._react_loop.run_stream(memory, agent_trace, start_time, native):
            yield event
            if event.get("type") == "final_summary":
                agent_summary = event.get("content", "")
            if event.get("type") == "error" and event.get("content") == TOKEN_LIMIT_MARKER:
                fast_result = self.run_fast(comment)
                fast_result["trace_id"] = trace_id
                fast_result["mode"] = "agent-stream-fallback-fast"
                yield {"type": "result", "data": fast_result}
                yield {"type": "done"}
                return

        result = assemble_result_from_memory(memory, comment)
        if agent_summary:
            result["agent_summary"] = agent_summary

        reflection_history: List[Dict[str, Any]] = []
        if self._reflector and result.get("analysis_complete"):
            yield {"type": "reflection_start"}
            result, reflection_history = self._code_level_reflection(
                result, memory, agent_trace, start_time
            )
            yield {"type": "reflection_done", "rounds": len(reflection_history)}

        if reflection_history:
            result["reflection"] = {
                "total_rounds": len(reflection_history),
                "final_passed": reflection_history[-1].get("passed", True),
                "history": reflection_history,
            }

        result["trace_id"] = trace_id
        result["agent_trace"] = agent_trace
        result["elapsed_ms"] = round((time.time() - start_time) * 1000, 2)
        result["steps"] = memory.get_step_count()
        result["mode"] = "agent-stream-native" if native else "agent-stream-prompt"

        self._update_knowledge(result, reviewer_id, product_id, product_name)
        self._apply_detail_level(result, reviewer_id)

        yield {"type": "result", "data": result}
        yield {"type": "done"}

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
    # 批量分析
    # ------------------------------------------------------------------

    def run_batch(
        self,
        comments: List[str],
        use_fast_path: bool = True,
        max_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        批量分析评论。

        当 max_workers > 1 时使用线程池并发执行，否则顺序执行。
        结果按原始输入顺序排列。
        """
        if max_workers <= 1 or len(comments) <= 1:
            results = []
            for i, comment in enumerate(comments):
                logger.info("批量分析 [%d/%d]", i + 1, len(comments))
                result = self.run(comment, use_fast_path=use_fast_path)
                result["batch_index"] = i
                results.append(result)
            return results

        logger.info("批量并发分析 %d 条评论 (max_workers=%d)", len(comments), max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.run, c, use_fast_path=use_fast_path): i
                for i, c in enumerate(comments)
            }
            results = []
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    result["batch_index"] = idx
                    results.append(result)
                    logger.info("批量分析 [%d/%d] 完成", idx + 1, len(comments))
                except Exception as e:
                    logger.exception("批量分析 [%d/%d] 异常", idx + 1, len(comments))
                    results.append({
                        "batch_index": idx,
                        "analysis_complete": False,
                        "error": str(e),
                        "original_text": comments[idx],
                    })
        return sorted(results, key=lambda x: x["batch_index"])

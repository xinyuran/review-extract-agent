"""
ReAct 循环执行器

将 _react_native 和 _react_prompt_based 的公共逻辑提取为模板方法，
差异部分（LLM 调用方式、工具发现方式、结果写回方式）通过 native 参数切换。
"""

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

from .memory import AgentMemory

logger = logging.getLogger(__name__)

_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

TOKEN_LIMIT_MARKER = "__token_limit_exceeded__"
MAX_TOOL_RESULT_CHARS = 2048


def parse_tool_calls_from_text(content: str) -> List[Dict[str, Any]]:
    """从 <tool_call>...</tool_call> 标签中解析工具调用"""
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


def strip_tool_call_tags(content: str) -> str:
    return _TOOL_CALL_PATTERN.sub("", content).strip()


def is_context_length_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in (
        "maximum context length",
        "max_model_len",
        "context length",
        "too many tokens",
        "token limit",
    ))


def truncate_result(result_str: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    """智能截断工具结果，优先截断 thinking 字段而保留核心数据。"""
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


_COMPACT_THRESHOLD_CHARS = 12000


class ReactLoop:
    """
    统一的 ReAct 循环执行器。

    通过 native 参数决定使用 Native Function Calling 还是 Prompt-based 模式，
    公共逻辑（超时检测、trace 记录、工具执行、结果截断）统一处理。
    """

    def __init__(
        self,
        llm_service,
        execute_tool: Callable,
        tool_definitions: List[Dict[str, Any]],
        max_steps: int = 10,
        timeout: int = 1200,
        compact_threshold: int = _COMPACT_THRESHOLD_CHARS,
    ):
        self._llm_service = llm_service
        self._execute_tool = execute_tool
        self._tool_definitions = tool_definitions
        self._max_steps = max_steps
        self._timeout = timeout
        self._compact_threshold = compact_threshold

    def run(
        self,
        memory: AgentMemory,
        agent_trace: List[Dict[str, Any]],
        start_time: float,
        native: bool,
    ) -> Optional[str]:
        if native:
            return self._run_native(memory, agent_trace, start_time)
        else:
            return self._run_prompt_based(memory, agent_trace, start_time)

    def _execute_and_format_tools(
        self, calls: List[Dict[str, Any]]
    ) -> tuple:
        """执行一组工具调用并格式化结果，返回 (tool_names, observation_parts, tool_results)"""
        tool_names = []
        observation_parts = []
        tool_results = []
        for call in calls:
            tool_name = call["name"]
            tool_names.append(tool_name)
            arguments = call.get("arguments", {})
            tool_result = self._execute_tool(tool_name, arguments)
            tool_results.append(tool_result)
            result_str = json.dumps(
                tool_result.model_dump(), ensure_ascii=False, default=str
            )
            result_str = truncate_result(result_str)
            observation_parts.append(f"[工具 {tool_name} 结果]\n{result_str}")
        return tool_names, observation_parts, tool_results

    def _write_observation_as_user(
        self, memory: AgentMemory, observation_parts: List[str]
    ) -> None:
        observation = "\n\n".join(observation_parts)
        memory.add_user_message(
            f"以上工具已执行完毕，以下是执行结果：\n\n{observation}\n\n"
            f"请根据以上结果继续分析，或输出你的总结。"
        )

    def _run_native(
        self,
        memory: AgentMemory,
        agent_trace: List[Dict[str, Any]],
        start_time: float,
    ) -> Optional[str]:
        agent_summary: Optional[str] = None

        for step in range(self._max_steps):
            elapsed = time.time() - start_time
            if elapsed > self._timeout:
                logger.warning("Agent 超时 (%.1fs > %ds)", elapsed, self._timeout)
                break

            if memory.get_total_chars() > self._compact_threshold:
                trimmed = memory.compact()
                if trimmed:
                    logger.info("上下文裁剪: 压缩了 %d 条旧 tool_result", trimmed)

            step_num = len(agent_trace) + 1
            logger.info("--- Agent Step %d (native) ---", step_num)

            try:
                llm_resp = self._llm_service.call_agent(
                    messages=memory.to_messages(),
                    tools=self._tool_definitions,
                    tool_choice="auto",
                )
            except Exception as e:
                logger.exception("Agent LLM 调用失败")
                agent_trace.append({
                    "step": step_num, "type": "error",
                    "content": f"Agent LLM 调用失败: {e}",
                })
                if is_context_length_error(e):
                    agent_summary = TOKEN_LIMIT_MARKER
                break

            thought = llm_resp.content or ""

            if not llm_resp.has_tool_calls:
                fallback_calls = parse_tool_calls_from_text(thought)
                if fallback_calls:
                    logger.info(
                        "Native 模式检测到 <tool_call> 标签 (step %d), "
                        "回退到 prompt-based 执行", step_num,
                    )
                    memory.add_assistant_message(content=thought)
                    tool_names, observations, _ = self._execute_and_format_tools(fallback_calls)
                    self._write_observation_as_user(memory, observations)
                    agent_trace.append({
                        "step": step_num,
                        "type": "thought_and_action",
                        "thought": strip_tool_call_tags(thought),
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
                    logger.warning("工具 %s 参数解析失败: %s",
                                   tool_name, tc["function"]["arguments"])

                tool_result = self._execute_tool(tool_name, arguments)
                result_str = json.dumps(
                    tool_result.model_dump(), ensure_ascii=False, default=str
                )
                result_str = truncate_result(result_str)
                memory.add_tool_result(
                    tool_call_id=tc["id"], tool_name=tool_name, result=result_str,
                )

            if thought:
                logger.info("Thought: %s",
                            thought[:100] + ("..." if len(thought) > 100 else ""))

            agent_trace.append({
                "step": step_num,
                "type": "thought_and_action",
                "thought": thought,
                "actions": tool_names,
            })

        return agent_summary

    # ------------------------------------------------------------------
    # 流式 ReAct 循环
    # ------------------------------------------------------------------

    def run_stream(
        self,
        memory: AgentMemory,
        agent_trace: List[Dict[str, Any]],
        start_time: float,
        native: bool,
    ):
        """
        流式 ReAct 循环，yield 结构化事件字典。

        事件类型：
        - step_start: 每步开始
        - token: LLM 生成的单个 token
        - thought: 完整思考文本（步结束时）
        - tool_call: 工具调用请求
        - tool_result: 工具执行结果
        - final_summary: 最终总结
        - error: 错误信息
        """
        for step in range(self._max_steps):
            elapsed = time.time() - start_time
            if elapsed > self._timeout:
                logger.warning("Agent 超时 (%.1fs > %ds)", elapsed, self._timeout)
                yield {"type": "error", "content": "Agent 超时"}
                break

            step_num = len(agent_trace) + 1
            yield {"type": "step_start", "step": step_num, "mode": "native" if native else "prompt"}

            try:
                if native:
                    finished = yield from self._stream_native_step(
                        memory, agent_trace, step_num
                    )
                else:
                    finished = yield from self._stream_prompt_step(
                        memory, agent_trace, step_num
                    )
                if finished:
                    break
            except Exception as e:
                logger.exception("Agent LLM 调用失败")
                agent_trace.append({
                    "step": step_num, "type": "error",
                    "content": f"Agent LLM 调用失败: {e}",
                })
                yield {"type": "error", "content": str(e)}
                if is_context_length_error(e):
                    yield {"type": "error", "content": TOKEN_LIMIT_MARKER}
                break

    def _stream_native_step(
        self,
        memory: AgentMemory,
        agent_trace: List[Dict[str, Any]],
        step_num: int,
    ) -> bool:
        """流式执行 native FC 单步。返回 True 表示循环应结束。"""
        accumulated_content = ""
        tool_calls_map: Dict[int, Dict[str, Any]] = {}

        for chunk in self._llm_service.call_agent_stream(
            messages=memory.to_messages(),
            tools=self._tool_definitions,
            tool_choice="auto",
        ):
            if chunk.delta_content:
                accumulated_content += chunk.delta_content
                yield {"type": "token", "content": chunk.delta_content}

            if chunk.tool_call_index is not None:
                idx = chunk.tool_call_index
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {
                        "id": "", "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                tc = tool_calls_map[idx]
                if chunk.tool_call_id:
                    tc["id"] = chunk.tool_call_id
                if chunk.tool_call_function_name:
                    tc["function"]["name"] = chunk.tool_call_function_name
                if chunk.tool_call_function_args_delta:
                    tc["function"]["arguments"] += chunk.tool_call_function_args_delta

        if not tool_calls_map:
            fallback_calls = parse_tool_calls_from_text(accumulated_content)
            if fallback_calls:
                memory.add_assistant_message(content=accumulated_content)
                yield {"type": "thought", "content": strip_tool_call_tags(accumulated_content)}
                yield from self._stream_tool_execution(fallback_calls, agent_trace, step_num)
                tool_names = [c["name"] for c in fallback_calls]
                self._write_observation_as_user(
                    memory,
                    [f"[工具 {c['name']} 结果] (见上)" for c in fallback_calls],
                )
                agent_trace.append({
                    "step": step_num, "type": "thought_and_action",
                    "thought": strip_tool_call_tags(accumulated_content),
                    "actions": tool_names,
                })
                return False

            memory.add_assistant_message(content=accumulated_content)
            yield {"type": "final_summary", "content": accumulated_content}
            agent_trace.append({
                "step": step_num, "type": "final_summary",
                "content": accumulated_content,
            })
            return True

        tool_calls_raw = [tool_calls_map[i] for i in sorted(tool_calls_map)]
        tool_names = [tc["function"]["name"] for tc in tool_calls_raw]
        memory.add_assistant_message(content=accumulated_content, tool_calls=tool_calls_raw)

        if accumulated_content:
            yield {"type": "thought", "content": accumulated_content}

        for tc in tool_calls_raw:
            tool_name = tc["function"]["name"]
            try:
                arguments = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = {}

            yield {"type": "tool_call", "name": tool_name, "arguments": arguments}

            tool_result = self._execute_tool(tool_name, arguments)
            result_str = json.dumps(
                tool_result.model_dump(), ensure_ascii=False, default=str
            )
            result_str = truncate_result(result_str)
            memory.add_tool_result(
                tool_call_id=tc["id"], tool_name=tool_name, result=result_str,
            )
            yield {"type": "tool_result", "name": tool_name, "success": tool_result.success}

        agent_trace.append({
            "step": step_num, "type": "thought_and_action",
            "thought": accumulated_content, "actions": tool_names,
        })
        return False

    def _stream_prompt_step(
        self,
        memory: AgentMemory,
        agent_trace: List[Dict[str, Any]],
        step_num: int,
    ) -> bool:
        """流式执行 prompt-based 单步。返回 True 表示循环应结束。"""
        accumulated_content = ""

        for chunk in self._llm_service.call_agent_stream(
            messages=memory.to_messages(),
        ):
            if chunk.delta_content:
                accumulated_content += chunk.delta_content
                yield {"type": "token", "content": chunk.delta_content}

        parsed_calls = parse_tool_calls_from_text(accumulated_content)

        if not parsed_calls:
            memory.add_assistant_message(content=accumulated_content)
            yield {"type": "final_summary", "content": accumulated_content}
            agent_trace.append({
                "step": step_num, "type": "final_summary",
                "content": accumulated_content,
            })
            return True

        memory.add_assistant_message(content=accumulated_content)
        thought_text = strip_tool_call_tags(accumulated_content)
        yield {"type": "thought", "content": thought_text}

        tool_names, observation_parts, tool_results = self._execute_and_format_tools(parsed_calls)
        for call in parsed_calls:
            yield {"type": "tool_call", "name": call["name"], "arguments": call.get("arguments", {})}
        for i, name in enumerate(tool_names):
            yield {"type": "tool_result", "name": name, "success": tool_results[i].success}

        self._write_observation_as_user(memory, observation_parts)
        agent_trace.append({
            "step": step_num, "type": "thought_and_action",
            "thought": thought_text, "actions": tool_names,
        })
        return False

    def _stream_tool_execution(
        self,
        calls: List[Dict[str, Any]],
        agent_trace: List[Dict[str, Any]],
        step_num: int,
    ):
        """执行工具并 yield 事件"""
        for call in calls:
            tool_name = call["name"]
            arguments = call.get("arguments", {})
            yield {"type": "tool_call", "name": tool_name, "arguments": arguments}
            tool_result = self._execute_tool(tool_name, arguments)
            yield {"type": "tool_result", "name": tool_name, "success": tool_result.success}

    # ------------------------------------------------------------------
    # 非流式 prompt-based
    # ------------------------------------------------------------------

    def _run_prompt_based(
        self,
        memory: AgentMemory,
        agent_trace: List[Dict[str, Any]],
        start_time: float,
    ) -> Optional[str]:
        agent_summary: Optional[str] = None

        for step in range(self._max_steps):
            elapsed = time.time() - start_time
            if elapsed > self._timeout:
                logger.warning("Agent 超时 (%.1fs > %ds)", elapsed, self._timeout)
                break

            if memory.get_total_chars() > self._compact_threshold:
                trimmed = memory.compact()
                if trimmed:
                    logger.info("上下文裁剪: 压缩了 %d 条旧 tool_result", trimmed)

            step_num = len(agent_trace) + 1
            logger.info("--- Agent Step %d (prompt-based) ---", step_num)

            try:
                llm_resp = self._llm_service.call_agent(
                    messages=memory.to_messages(),
                )
            except Exception as e:
                logger.exception("Agent LLM 调用失败")
                agent_trace.append({
                    "step": step_num, "type": "error",
                    "content": f"Agent LLM 调用失败: {e}",
                })
                if is_context_length_error(e):
                    agent_summary = TOKEN_LIMIT_MARKER
                break

            content = llm_resp.content or ""
            parsed_calls = parse_tool_calls_from_text(content)

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
            thought_text = strip_tool_call_tags(content)

            tool_names, observation_parts, _ = self._execute_and_format_tools(parsed_calls)
            self._write_observation_as_user(memory, observation_parts)

            agent_trace.append({
                "step": step_num,
                "type": "thought_and_action",
                "thought": thought_text,
                "actions": tool_names,
            })

        return agent_summary

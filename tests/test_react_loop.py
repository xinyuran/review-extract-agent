"""ReactLoop 单元测试

覆盖场景：
- Native FC 模式：工具调用 + 最终总结
- Prompt-based 模式：<tool_call> 标签解析
- Token 超限降级
- 超时检测
- 流式模式事件序列
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from extract_agent.agent.react_loop import (
    ReactLoop,
    TOKEN_LIMIT_MARKER,
    parse_tool_calls_from_text,
    strip_tool_call_tags,
    truncate_result,
    is_context_length_error,
)
from extract_agent.agent.memory import AgentMemory
from extract_agent.tools.base_tool import ToolResult
from extract_agent.llm_service.models import LLMResponse, LLMStreamChunk


# ── helpers ──

def _make_tool_result(success=True, data=None):
    return ToolResult(success=success, data=data or {})


def _make_llm_response(content="", tool_calls=None):
    return LLMResponse(content=content, tool_calls=tool_calls)


def _make_react_loop(llm_service=None, execute_tool=None, max_steps=10, timeout=120):
    llm_service = llm_service or MagicMock()
    execute_tool = execute_tool or MagicMock(return_value=_make_tool_result())
    return ReactLoop(
        llm_service=llm_service,
        execute_tool=execute_tool,
        tool_definitions=[],
        max_steps=max_steps,
        timeout=timeout,
    )


# ══════════════════════════════════════════════════
# parse helpers
# ══════════════════════════════════════════════════

class TestParseToolCalls:
    def test_single_tool_call(self):
        text = '<tool_call>{"name":"text_preprocess","arguments":{"text":"hello"}}</tool_call>'
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "text_preprocess"
        assert calls[0]["arguments"]["text"] == "hello"

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name":"a","arguments":{}}</tool_call>'
            ' some text '
            '<tool_call>{"name":"b","arguments":{}}</tool_call>'
        )
        calls = parse_tool_calls_from_text(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "a"
        assert calls[1]["name"] == "b"

    def test_invalid_json(self):
        text = '<tool_call>{invalid}</tool_call>'
        calls = parse_tool_calls_from_text(text)
        assert calls == []

    def test_missing_name(self):
        text = '<tool_call>{"arguments":{}}</tool_call>'
        calls = parse_tool_calls_from_text(text)
        assert calls == []

    def test_strip_tool_call_tags(self):
        text = 'thinking <tool_call>{"name":"a","arguments":{}}</tool_call> more'
        stripped = strip_tool_call_tags(text)
        assert "<tool_call>" not in stripped
        assert "thinking" in stripped


class TestTruncateResult:
    def test_short_string_unchanged(self):
        s = '{"success": true}'
        assert truncate_result(s) == s

    def test_long_string_truncated(self):
        s = "x" * 3000
        result = truncate_result(s, max_chars=100)
        assert len(result) <= 100

    def test_thinking_field_truncated_first(self):
        data = {"data": {"thinking": "A" * 1000, "keywords": [1, 2, 3]}}
        s = json.dumps(data, ensure_ascii=False)
        result = truncate_result(s, max_chars=200)
        assert len(result) <= 200


class TestIsContextLengthError:
    @pytest.mark.parametrize("msg", [
        "maximum context length exceeded",
        "max_model_len is 8192",
        "context length error",
        "too many tokens for the model",
        "token limit exceeded",
    ])
    def test_detects_context_errors(self, msg):
        assert is_context_length_error(Exception(msg))

    def test_ignores_unrelated_errors(self):
        assert not is_context_length_error(Exception("connection refused"))


# ══════════════════════════════════════════════════
# _run_native
# ══════════════════════════════════════════════════

class TestRunNative:
    def test_final_summary_returned(self):
        """LLM 不返回 tool_calls 时，视为最终总结"""
        llm = MagicMock()
        llm.call_agent.return_value = _make_llm_response(content="分析完成")
        loop = _make_react_loop(llm_service=llm)

        memory = AgentMemory(system_prompt="sys")
        memory.add_user_message("user msg")
        trace = []
        result = loop.run(memory, trace, time.time(), native=True)

        assert result == "分析完成"
        assert trace[-1]["type"] == "final_summary"

    def test_tool_call_then_summary(self):
        """先返回工具调用，再返回总结"""
        llm = MagicMock()
        tool_calls = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "text_preprocess", "arguments": '{"text":"hi"}'},
        }]
        llm.call_agent.side_effect = [
            _make_llm_response(content="让我预处理", tool_calls=tool_calls),
            _make_llm_response(content="分析完成"),
        ]
        execute_tool = MagicMock(return_value=_make_tool_result(
            data={"cleaned_text": "hi"}
        ))
        loop = _make_react_loop(llm_service=llm, execute_tool=execute_tool)

        memory = AgentMemory(system_prompt="sys")
        memory.add_user_message("user msg")
        trace = []
        result = loop.run(memory, trace, time.time(), native=True)

        assert result == "分析完成"
        execute_tool.assert_called_once()
        assert len(trace) == 2
        assert trace[0]["type"] == "thought_and_action"
        assert trace[1]["type"] == "final_summary"

    def test_token_limit_returns_marker(self):
        """上下文超限时返回 TOKEN_LIMIT_MARKER"""
        llm = MagicMock()
        llm.call_agent.side_effect = Exception("maximum context length exceeded")
        loop = _make_react_loop(llm_service=llm)

        memory = AgentMemory(system_prompt="sys")
        memory.add_user_message("user msg")
        trace = []
        result = loop.run(memory, trace, time.time(), native=True)

        assert result == TOKEN_LIMIT_MARKER

    def test_timeout_breaks_loop(self):
        """超时时循环提前终止"""
        llm = MagicMock()
        tool_calls = [{
            "id": "call_1", "type": "function",
            "function": {"name": "text_preprocess", "arguments": '{"text":"hi"}'},
        }]
        llm.call_agent.return_value = _make_llm_response(
            content="", tool_calls=tool_calls
        )
        loop = _make_react_loop(llm_service=llm, timeout=0)

        memory = AgentMemory(system_prompt="sys")
        memory.add_user_message("user msg")
        trace = []
        start = time.time() - 1
        result = loop.run(memory, trace, start, native=True)

        assert result is None

    def test_fallback_to_prompt_based_on_tool_call_tag(self):
        """Native 模式下，LLM 返回 <tool_call> 标签时 fallback"""
        llm = MagicMock()
        tag_content = '<tool_call>{"name":"text_preprocess","arguments":{"text":"hi"}}</tool_call>'
        llm.call_agent.side_effect = [
            _make_llm_response(content=f"思考一下 {tag_content}"),
            _make_llm_response(content="完成"),
        ]
        execute_tool = MagicMock(return_value=_make_tool_result())
        loop = _make_react_loop(llm_service=llm, execute_tool=execute_tool)

        memory = AgentMemory(system_prompt="sys")
        memory.add_user_message("user msg")
        trace = []
        result = loop.run(memory, trace, time.time(), native=True)

        execute_tool.assert_called_once()
        assert result == "完成"


# ══════════════════════════════════════════════════
# _run_prompt_based
# ══════════════════════════════════════════════════

class TestRunPromptBased:
    def test_tool_call_then_summary(self):
        """Prompt-based: <tool_call> 解析 → 工具执行 → 最终总结"""
        llm = MagicMock()
        tag = '<tool_call>{"name":"text_preprocess","arguments":{"text":"hi"}}</tool_call>'
        llm.call_agent.side_effect = [
            _make_llm_response(content=f"分析评论 {tag}"),
            _make_llm_response(content="最终总结"),
        ]
        execute_tool = MagicMock(return_value=_make_tool_result())
        loop = _make_react_loop(llm_service=llm, execute_tool=execute_tool)

        memory = AgentMemory(system_prompt="sys")
        memory.add_user_message("user msg")
        trace = []
        result = loop.run(memory, trace, time.time(), native=False)

        assert result == "最终总结"
        execute_tool.assert_called_once()

    def test_no_tool_call_is_final(self):
        """无 <tool_call> 标签时视为最终总结"""
        llm = MagicMock()
        llm.call_agent.return_value = _make_llm_response(content="直接总结")
        loop = _make_react_loop(llm_service=llm)

        memory = AgentMemory(system_prompt="sys")
        memory.add_user_message("user msg")
        trace = []
        result = loop.run(memory, trace, time.time(), native=False)

        assert result == "直接总结"

    def test_token_limit_returns_marker(self):
        llm = MagicMock()
        llm.call_agent.side_effect = Exception("context length exceeded")
        loop = _make_react_loop(llm_service=llm)

        memory = AgentMemory(system_prompt="sys")
        memory.add_user_message("user msg")
        trace = []
        result = loop.run(memory, trace, time.time(), native=False)

        assert result == TOKEN_LIMIT_MARKER


# ══════════════════════════════════════════════════
# run_stream
# ══════════════════════════════════════════════════

class TestRunStream:
    def test_stream_native_final_summary(self):
        """流式 native 模式：直接出最终总结"""
        llm = MagicMock()
        llm.call_agent_stream.return_value = iter([
            LLMStreamChunk(delta_content="分析"),
            LLMStreamChunk(delta_content="完成"),
        ])
        loop = _make_react_loop(llm_service=llm)

        memory = AgentMemory(system_prompt="sys")
        memory.add_user_message("user msg")
        trace = []
        events = list(loop.run_stream(memory, trace, time.time(), native=True))

        types = [e["type"] for e in events]
        assert "step_start" in types
        assert "token" in types
        assert "final_summary" in types

    def test_stream_native_with_tool_calls(self):
        """流式 native 模式：工具调用 → 最终总结"""
        llm = MagicMock()
        llm.call_agent_stream.side_effect = [
            iter([
                LLMStreamChunk(
                    tool_call_index=0,
                    tool_call_id="call_1",
                    tool_call_function_name="text_preprocess",
                ),
                LLMStreamChunk(
                    tool_call_index=0,
                    tool_call_function_args_delta='{"text":',
                ),
                LLMStreamChunk(
                    tool_call_index=0,
                    tool_call_function_args_delta='"hi"}',
                ),
            ]),
            iter([
                LLMStreamChunk(delta_content="完成"),
            ]),
        ]
        execute_tool = MagicMock(return_value=_make_tool_result())
        loop = _make_react_loop(llm_service=llm, execute_tool=execute_tool)

        memory = AgentMemory(system_prompt="sys")
        memory.add_user_message("user msg")
        trace = []
        events = list(loop.run_stream(memory, trace, time.time(), native=True))

        types = [e["type"] for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "final_summary" in types

    def test_stream_prompt_based(self):
        """流式 prompt-based 模式"""
        llm = MagicMock()
        tag = '<tool_call>{"name":"text_preprocess","arguments":{"text":"hi"}}</tool_call>'
        llm.call_agent_stream.side_effect = [
            iter([LLMStreamChunk(delta_content=f"思考 {tag}")]),
            iter([LLMStreamChunk(delta_content="完成")]),
        ]
        execute_tool = MagicMock(return_value=_make_tool_result())
        loop = _make_react_loop(llm_service=llm, execute_tool=execute_tool)

        memory = AgentMemory(system_prompt="sys")
        memory.add_user_message("user msg")
        trace = []
        events = list(loop.run_stream(memory, trace, time.time(), native=False))

        types = [e["type"] for e in events]
        assert "thought" in types
        assert "tool_call" in types
        assert "tool_result" in types
        assert "final_summary" in types

    def test_stream_tool_result_uses_real_success(self):
        """流式 prompt-based 中 tool_result.success 使用真实值"""
        llm = MagicMock()
        tag = '<tool_call>{"name":"keyword_extract","arguments":{"text":"hi"}}</tool_call>'
        llm.call_agent_stream.side_effect = [
            iter([LLMStreamChunk(delta_content=tag)]),
            iter([LLMStreamChunk(delta_content="完成")]),
        ]
        execute_tool = MagicMock(return_value=_make_tool_result(success=False))
        loop = _make_react_loop(llm_service=llm, execute_tool=execute_tool)

        memory = AgentMemory(system_prompt="sys")
        memory.add_user_message("user msg")
        trace = []
        events = list(loop.run_stream(memory, trace, time.time(), native=False))

        tool_result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_result_events) == 1
        assert tool_result_events[0]["success"] is False

    def test_stream_token_limit(self):
        """流式模式下 token 超限"""
        llm = MagicMock()
        llm.call_agent_stream.side_effect = Exception("maximum context length")
        loop = _make_react_loop(llm_service=llm)

        memory = AgentMemory(system_prompt="sys")
        memory.add_user_message("user msg")
        trace = []
        events = list(loop.run_stream(memory, trace, time.time(), native=True))

        types = [e["type"] for e in events]
        assert "error" in types
        error_events = [e for e in events if e["type"] == "error"]
        assert any(e.get("content") == TOKEN_LIMIT_MARKER for e in error_events)

    def test_stream_timeout(self):
        """流式模式下超时"""
        llm = MagicMock()
        loop = _make_react_loop(llm_service=llm, timeout=0)

        memory = AgentMemory(system_prompt="sys")
        memory.add_user_message("user msg")
        trace = []
        start = time.time() - 1
        events = list(loop.run_stream(memory, trace, start, native=True))

        types = [e["type"] for e in events]
        assert "error" in types

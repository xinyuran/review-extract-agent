"""
OPT-01: 上下文裁剪 — 单元测试

验证 AgentMemory.compact() 方法和 ReactLoop 中的自动触发。
"""

import json
import time
from unittest.mock import MagicMock

import pytest

from extract_agent.agent.memory import AgentMemory
from extract_agent.agent.react_loop import ReactLoop, _COMPACT_THRESHOLD_CHARS


class TestMemoryCompact:

    def _build_memory_with_tools(self, n_tools: int, content_size: int = 500) -> AgentMemory:
        """构造含有 n 条 tool_result 的 Memory"""
        mem = AgentMemory(system_prompt="你是分析助手")
        mem.add_user_message("分析这条评论")

        for i in range(n_tools):
            mem.add_assistant_message(
                content=f"思考第{i}步",
                tool_calls=[{
                    "id": f"tc_{i}",
                    "type": "function",
                    "function": {"name": f"tool_{i}", "arguments": "{}"},
                }],
            )
            mem.add_tool_result(
                tool_call_id=f"tc_{i}",
                tool_name=f"tool_{i}",
                result="x" * content_size,
            )
        return mem

    def test_compact_trims_old_results(self):
        """compact 裁剪旧 tool_result，保留最近 2 条"""
        mem = self._build_memory_with_tools(5)
        trimmed = mem.compact(keep_recent=2)

        assert trimmed == 3

        tool_msgs = [m for m in mem._messages if m.get("role") == "tool"]
        compressed = [m for m in tool_msgs if m["content"] == "[历史结果已压缩]"]
        intact = [m for m in tool_msgs if m["content"] != "[历史结果已压缩]"]

        assert len(compressed) == 3
        assert len(intact) == 2

    def test_compact_preserves_message_structure(self):
        """裁剪后消息结构完整（tool_call_id 和 name 保留）"""
        mem = self._build_memory_with_tools(3)
        mem.compact(keep_recent=1)

        for m in mem._messages:
            if m.get("role") == "tool" and m["content"] == "[历史结果已压缩]":
                assert "tool_call_id" in m
                assert "name" in m
                assert m["tool_call_id"] != ""

    def test_compact_no_op_when_few_tools(self):
        """工具结果少于 keep_recent 时不裁剪"""
        mem = self._build_memory_with_tools(2)
        trimmed = mem.compact(keep_recent=2)
        assert trimmed == 0

    def test_compact_no_op_on_empty(self):
        """无 tool_result 时不裁剪"""
        mem = AgentMemory(system_prompt="test")
        mem.add_user_message("hello")
        mem.add_assistant_message(content="hi")
        trimmed = mem.compact()
        assert trimmed == 0

    def test_compact_idempotent(self):
        """多次调用 compact 是幂等的"""
        mem = self._build_memory_with_tools(5)
        t1 = mem.compact(keep_recent=2)
        t2 = mem.compact(keep_recent=2)
        assert t1 == 3
        assert t2 == 0

    def test_compact_reduces_char_count(self):
        """裁剪后总字符数明显减少"""
        mem = self._build_memory_with_tools(5, content_size=1000)
        chars_before = mem.get_total_chars()
        mem.compact(keep_recent=2)
        chars_after = mem.get_total_chars()
        assert chars_after < chars_before

    def test_get_total_chars(self):
        """get_total_chars 正确计算"""
        mem = AgentMemory(system_prompt="abc")
        mem.add_user_message("12345")
        assert mem.get_total_chars() == 3 + 5  # system + user

    def test_to_messages_after_compact(self):
        """裁剪后 to_messages() 返回有效的 OpenAI messages 格式"""
        mem = self._build_memory_with_tools(4)
        mem.compact(keep_recent=1)

        messages = mem.to_messages()

        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles

        for m in messages:
            assert "role" in m
            if m["role"] == "tool":
                assert "content" in m
                assert "tool_call_id" in m


class TestReactLoopAutoCompact:

    def test_compact_triggered_when_threshold_exceeded(self):
        """当消息总字符数超过阈值时自动触发 compact"""
        mem = AgentMemory(system_prompt="你是分析助手")
        mem.add_user_message("分析评论")

        for i in range(5):
            mem.add_assistant_message(
                content=f"思考{i}",
                tool_calls=[{
                    "id": f"tc_{i}", "type": "function",
                    "function": {"name": f"tool_{i}", "arguments": "{}"},
                }],
            )
            mem.add_tool_result(f"tc_{i}", f"tool_{i}", "x" * 3000)

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "分析完成"
        mock_resp.has_tool_calls = False
        mock_resp.tool_calls = None
        mock_llm.call_agent.return_value = mock_resp

        loop = ReactLoop(
            llm_service=mock_llm,
            execute_tool=MagicMock(),
            tool_definitions=[],
            max_steps=1,
            compact_threshold=5000,
        )

        assert mem.get_total_chars() > 5000

        loop.run(mem, [], time.time(), native=True)

        tool_msgs = [m for m in mem._messages if m.get("role") == "tool"]
        compressed = [m for m in tool_msgs if m["content"] == "[历史结果已压缩]"]
        assert len(compressed) > 0

    def test_no_compact_below_threshold(self):
        """消息总字符数低于阈值时不触发 compact"""
        mem = AgentMemory(system_prompt="你是分析助手")
        mem.add_user_message("短评论")

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "完成"
        mock_resp.has_tool_calls = False
        mock_resp.tool_calls = None
        mock_llm.call_agent.return_value = mock_resp

        loop = ReactLoop(
            llm_service=mock_llm,
            execute_tool=MagicMock(),
            tool_definitions=[],
            max_steps=1,
            compact_threshold=_COMPACT_THRESHOLD_CHARS,
        )

        loop.run(mem, [], time.time(), native=True)

        tool_msgs = [m for m in mem._messages if m.get("role") == "tool"]
        compressed = [m for m in tool_msgs if m.get("content") == "[历史结果已压缩]"]
        assert len(compressed) == 0

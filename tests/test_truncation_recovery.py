"""
OPT-07: 输出截断恢复 — 单元测试

验证当 finish_reason == "length" 时，call_agent 自动续写。
"""

from unittest.mock import MagicMock, patch, call
import pytest


def _make_mock_response(content, finish_reason="stop", tool_calls=None):
    """构造 mock OpenAI ChatCompletion response"""
    msg = MagicMock()
    msg.content = content
    msg.role = "assistant"
    msg.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    resp = MagicMock()
    resp.choices = [choice]
    resp.model = "test-model"
    resp.usage = MagicMock(
        prompt_tokens=10, completion_tokens=20, total_tokens=30
    )

    return resp


def _make_service():
    """构造使用 mock client 的 LLMService"""
    from extract_agent.llm_service.service import LLMService

    service = LLMService.__new__(LLMService)
    service.config = MagicMock()
    service.config.AGENT_LLM_MODEL = "test-model"
    service.config.AGENT_LLM_MAX_TOKENS = 1024
    service.config.AGENT_LLM_TEMPERATURE = 0.0
    service._agent_client = MagicMock()
    service._tool_client = None
    service._trajectory_recorder = None
    service._native_tool_calling = None
    service._detect_lock = MagicMock()
    return service


class TestTruncationRecovery:

    def test_no_truncation_normal_response(self):
        """finish_reason=stop 时不触发续写"""
        service = _make_service()
        service._agent_client.chat.completions.create.return_value = (
            _make_mock_response("完整输出", finish_reason="stop")
        )

        result = service.call_agent(
            messages=[{"role": "user", "content": "test"}]
        )

        assert result.content == "完整输出"
        assert service._agent_client.chat.completions.create.call_count == 1

    def test_truncation_triggers_continuation(self):
        """finish_reason=length 时触发续写，内容正确拼接"""
        service = _make_service()
        service._agent_client.chat.completions.create.side_effect = [
            _make_mock_response("前半段输出", finish_reason="length"),
            _make_mock_response("后半段输出", finish_reason="stop"),
        ]

        result = service.call_agent(
            messages=[{"role": "user", "content": "test"}]
        )

        assert result.content == "前半段输出后半段输出"
        assert service._agent_client.chat.completions.create.call_count == 2

    def test_truncation_with_tool_calls_no_continuation(self):
        """finish_reason=length 但有 tool_calls 时不续写"""
        service = _make_service()
        mock_tc = MagicMock()
        mock_tc.id = "tc_1"
        mock_tc.function.name = "test_tool"
        mock_tc.function.arguments = "{}"

        service._agent_client.chat.completions.create.return_value = (
            _make_mock_response("思考中", finish_reason="length", tool_calls=[mock_tc])
        )

        result = service.call_agent(
            messages=[{"role": "user", "content": "test"}]
        )

        assert service._agent_client.chat.completions.create.call_count == 1
        assert result.tool_calls is not None

    def test_continuation_failure_uses_original(self):
        """续写失败时使用截断的原始输出"""
        service = _make_service()
        service._agent_client.chat.completions.create.side_effect = [
            _make_mock_response("截断的输出", finish_reason="length"),
            Exception("429 Rate limit"),
            Exception("429 Rate limit"),
            Exception("429 Rate limit"),
            Exception("429 Rate limit"),
        ]

        result = service.call_agent(
            messages=[{"role": "user", "content": "test"}]
        )

        assert result.content == "截断的输出"

    def test_continuation_message_format(self):
        """验证续写请求的消息格式"""
        service = _make_service()

        responses = [
            _make_mock_response("截断", finish_reason="length"),
            _make_mock_response("续写", finish_reason="stop"),
        ]
        service._agent_client.chat.completions.create.side_effect = responses

        service.call_agent(
            messages=[{"role": "user", "content": "原始请求"}]
        )

        second_call_kwargs = service._agent_client.chat.completions.create.call_args_list[1]
        cont_messages = second_call_kwargs.kwargs.get("messages") or second_call_kwargs[1].get("messages")
        if cont_messages is None:
            cont_messages = second_call_kwargs[0][0] if second_call_kwargs[0] else None

        assert any(m.get("role") == "assistant" for m in cont_messages)
        assert any("截断" in m.get("content", "") for m in cont_messages if m.get("role") == "user")

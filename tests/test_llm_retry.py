"""
OPT-02: LLM 调用重试机制 — 单元测试

验证 with_retry 函数的行为：
- 瞬时错误触发重试
- 非瞬时错误不重试
- 重试耗尽后抛出最后异常
- 成功调用不受影响
- call_agent/call_tool 中集成了重试
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from extract_agent.llm_service.service import (
    _is_retryable,
    with_retry,
)


class TestIsRetryable:

    def test_rate_limit_is_retryable(self):
        assert _is_retryable(Exception("Rate limit exceeded (429)"))

    def test_503_is_retryable(self):
        assert _is_retryable(Exception("Service Unavailable 503"))

    def test_connection_error_is_retryable(self):
        assert _is_retryable(Exception("Connection refused"))

    def test_timeout_is_retryable(self):
        assert _is_retryable(Exception("Request timed out"))

    def test_bad_request_not_retryable(self):
        assert not _is_retryable(Exception("400 Bad Request: invalid parameters"))

    def test_auth_error_not_retryable(self):
        assert not _is_retryable(Exception("401 Unauthorized"))

    def test_context_length_not_retryable(self):
        assert not _is_retryable(Exception("maximum context length exceeded"))


class TestWithRetry:

    def test_success_no_retry(self):
        fn = MagicMock(return_value="ok")
        result = with_retry(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 1

    def test_retryable_error_retries(self):
        fn = MagicMock(side_effect=[
            Exception("429 Rate limit"),
            Exception("503 Service unavailable"),
            "ok",
        ])
        result = with_retry(fn, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 3

    def test_non_retryable_error_no_retry(self):
        fn = MagicMock(side_effect=Exception("400 Bad Request"))
        with pytest.raises(Exception, match="400 Bad Request"):
            with_retry(fn, max_retries=3, base_delay=0.01)
        assert fn.call_count == 1

    def test_max_retries_exhausted(self):
        fn = MagicMock(side_effect=Exception("429 Rate limit"))
        with pytest.raises(Exception, match="429 Rate limit"):
            with_retry(fn, max_retries=2, base_delay=0.01)
        assert fn.call_count == 3  # initial + 2 retries

    def test_delay_increases(self):
        """验证重试间隔递增（指数退避）"""
        delays = []
        original_sleep = time.sleep

        def _capture_sleep(seconds):
            delays.append(seconds)

        fn = MagicMock(side_effect=[
            Exception("429"), Exception("429"), "ok"
        ])

        with patch("extract_agent.llm_service.service.time.sleep", side_effect=_capture_sleep):
            with_retry(fn, max_retries=3, base_delay=1.0)

        assert len(delays) == 2
        assert delays[1] > delays[0]


class TestLLMServiceRetryIntegration:
    """验证 LLMService.call_agent/call_tool 中集成了 with_retry"""

    def test_call_agent_source_has_retry(self):
        import inspect
        from extract_agent.llm_service.service import LLMService
        source = inspect.getsource(LLMService.call_agent)
        assert "with_retry" in source

    def test_call_tool_source_has_retry(self):
        import inspect
        from extract_agent.llm_service.service import LLMService
        source = inspect.getsource(LLMService.call_tool_with_skill)
        assert "with_retry" in source

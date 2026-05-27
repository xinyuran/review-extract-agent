"""
OPT-04: 线程安全修复 — 单元测试

验证：
1. routes.get_agent() 多线程并发调用只创建一个实例
2. LLMService.detect_tool_calling_mode() 多线程并发调用安全
"""

import threading
from unittest.mock import MagicMock, patch

import pytest


class TestRoutesThreadSafety:
    """验证 API routes 中全局单例的线程安全"""

    def setup_method(self):
        from extract_agent.api import routes
        routes._agent = None
        routes._config = None
        routes._redis = None

    def teardown_method(self):
        from extract_agent.api import routes
        routes._agent = None
        routes._config = None
        routes._redis = None

    @patch("extract_agent.api.routes.ReviewAnalysisAgent")
    @patch("extract_agent.api.routes.AgentConfig")
    def test_get_agent_concurrent_single_instance(self, mock_config_cls, mock_agent_cls):
        """多线程并发调用 get_agent() 应只创建一个 Agent 实例"""
        from extract_agent.api import routes

        mock_config_cls.return_value = MagicMock()
        mock_agent_cls.return_value = MagicMock()

        results = []
        barrier = threading.Barrier(10)

        def _call():
            barrier.wait()
            agent = routes.get_agent()
            results.append(id(agent))

        threads = [threading.Thread(target=_call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1, f"Expected 1 unique agent, got {len(set(results))}"
        assert mock_agent_cls.call_count == 1

    @patch("extract_agent.api.routes.AgentConfig")
    def test_get_config_concurrent_single_instance(self, mock_config_cls):
        """多线程并发调用 get_config() 应只创建一个 Config 实例"""
        from extract_agent.api import routes

        mock_config_cls.return_value = MagicMock()

        results = []
        barrier = threading.Barrier(10)

        def _call():
            barrier.wait()
            config = routes.get_config()
            results.append(id(config))

        threads = [threading.Thread(target=_call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1


class TestLLMServiceThreadSafety:
    """验证 LLMService.detect_tool_calling_mode() 的线程安全"""

    def test_detect_mode_concurrent_single_probe(self):
        """多线程并发调用 detect_tool_calling_mode() 应只探测一次"""
        from extract_agent.llm_service.service import LLMService

        probe_count = 0
        original_probe = LLMService._probe_native_tool_calling

        def _mock_probe(self):
            nonlocal probe_count
            probe_count += 1
            import time
            time.sleep(0.05)
            return True

        with patch.object(LLMService, "_probe_native_tool_calling", _mock_probe):
            service = LLMService.__new__(LLMService)
            service.config = MagicMock()
            service.config.AGENT_TOOL_CALLING_MODE = "auto"
            service._native_tool_calling = None
            service._detect_lock = threading.Lock()
            service._agent_client = MagicMock()

            results = []
            barrier = threading.Barrier(10)

            def _call():
                barrier.wait()
                result = service.detect_tool_calling_mode()
                results.append(result)

            threads = [threading.Thread(target=_call) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert all(r is True for r in results)
        assert probe_count == 1, f"Expected 1 probe call, got {probe_count}"

    def test_detect_mode_cached_no_lock(self):
        """已缓存结果后不需要获取锁"""
        from extract_agent.llm_service.service import LLMService

        service = LLMService.__new__(LLMService)
        service.config = MagicMock()
        service._native_tool_calling = True
        service._detect_lock = threading.Lock()

        result = service.detect_tool_calling_mode()
        assert result is True

    def test_source_has_threading_lock(self):
        """验证 LLMService 使用了 threading.Lock"""
        import inspect
        from extract_agent.llm_service.service import LLMService
        source = inspect.getsource(LLMService.__init__)
        assert "_detect_lock" in source

    def test_routes_has_init_lock(self):
        """验证 routes 模块使用了 threading.Lock"""
        import inspect
        from extract_agent.api import routes
        source = inspect.getsource(routes.get_agent)
        assert "_init_lock" in source

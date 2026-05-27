"""
OPT-08: Reflector fallback 规范化 — 单元测试

验证 ResultReflector 不再有直接创建 OpenAI 客户端的 fallback 路径，
必须通过 LLMService 调用。
"""

import inspect

import pytest

from extract_agent.agent.reflector import ResultReflector


class TestReflectorNoFallback:

    def test_requires_llm_service(self):
        """不传 llm_service 时应抛出 ValueError"""
        with pytest.raises(ValueError, match="LLMService"):
            ResultReflector(llm_service=None)

    def test_no_direct_openai_import_in_reflect(self):
        """reflect() 方法中不应有 from openai import OpenAI"""
        source = inspect.getsource(ResultReflector.reflect)
        assert "from openai import OpenAI" not in source
        assert "OpenAI(" not in source

    def test_no_fallback_client_creation(self):
        """整个 reflector 模块中不应直接创建 OpenAI 客户端"""
        import extract_agent.agent.reflector as mod
        source = inspect.getsource(mod)
        assert "client = OpenAI(" not in source

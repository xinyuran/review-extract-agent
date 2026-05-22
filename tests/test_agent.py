"""ReviewAnalysisAgent 单元测试

覆盖场景：
- run() 主流程（mock ReactLoop + Reflector）
- run_stream() 事件序列 + 反思集成
- Token 超限降级到 fast 模式
- offline 模式走 jieba
- 批量分析 run_batch
- 工具执行入口 _execute_tool 路由
"""

import json
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from extract_agent.agent.agent import ReviewAnalysisAgent
from extract_agent.agent.react_loop import TOKEN_LIMIT_MARKER
from extract_agent.tools.base_tool import ToolResult
from extract_agent.llm_service.models import LLMResponse, SkillPrompt


# ── helpers ──

def _make_config(**overrides):
    """创建 mock config，避免真实环境变量干扰"""
    config = MagicMock()
    config.get_backend_mode.return_value = overrides.get("backend_mode", "cloud_api")
    config.AGENT_MAX_STEPS = overrides.get("max_steps", 10)
    config.AGENT_TIMEOUT = overrides.get("timeout", 120)
    config.AGENT_TOOL_CALLING_MODE = "native"
    config.ENABLE_REFLECTION = overrides.get("enable_reflection", False)
    config.REFLECTION_MAX_ROUNDS = 2
    config.REFLECTION_SCORE_THRESHOLD = 0.7
    config.REFLECTION_MIN_KEYWORDS_SHORT = 2
    config.REFLECTION_MIN_KEYWORDS_MEDIUM = 5
    config.REFLECTION_MIN_KEYWORDS_LONG = 8
    config.REFLECTION_MIN_KEYWORDS_XLONG = 10
    config.N = 8
    config.TOOL_LLM_SEED = 42
    config.TOOL_LLM_RESPONSE_FORMAT = None
    config.TOOL_TIMEOUT = 30
    config.DEBUG = False
    config.ENABLE_KNOWLEDGE = False
    config.SHORT_TEXT_LEN = 10
    config.TOOL_LLM_MODEL = "test-model"
    return config


def _make_llm_service():
    mock = MagicMock()
    mock.detect_tool_calling_mode.return_value = True
    mock.load_skill.return_value = SkillPrompt(
        name="test", description="test", target="agent_llm",
        system="system prompt", user="user prompt",
    )
    mock.build_tool_descriptions.return_value = "tool descriptions"
    return mock


def _make_agent(config=None, llm_service=None, **config_overrides):
    """创建带 mock 依赖的 agent"""
    config = config or _make_config(**config_overrides)
    llm_service = llm_service or _make_llm_service()

    with patch("extract_agent.agent.agent.PreprocessTool"), \
         patch("extract_agent.agent.agent.JiebaExtractTool"), \
         patch("extract_agent.agent.agent.ValidateTool"), \
         patch("extract_agent.agent.agent.KeywordExtractTool") as MockKW, \
         patch("extract_agent.agent.agent.SentimentTool") as MockSent:

        mock_kw = MagicMock()
        mock_kw.name = "keyword_extract"
        mock_kw.to_openai_tool.return_value = {"type": "function", "function": {"name": "keyword_extract"}}
        MockKW.return_value = mock_kw

        mock_sent = MagicMock()
        mock_sent.name = "sentiment_analyze"
        mock_sent.to_openai_tool.return_value = {"type": "function", "function": {"name": "sentiment_analyze"}}
        MockSent.return_value = mock_sent

        agent = ReviewAnalysisAgent(config=config, llm_service=llm_service)

    return agent


# ══════════════════════════════════════════════════
# _execute_tool routing
# ══════════════════════════════════════════════════

class TestExecuteTool:
    def test_routes_llm_tool(self):
        agent = _make_agent()
        agent._keyword_tool = MagicMock()
        agent._keyword_tool.execute.return_value = ToolResult(
            success=True, data={"keywords": []}
        )

        result = agent._execute_tool("keyword_extract", {"text": "好评"})

        assert result.success is True
        agent._keyword_tool.execute.assert_called_once()

    def test_routes_sentiment_tool(self):
        agent = _make_agent()
        agent._sentiment_tool = MagicMock()
        agent._sentiment_tool.execute.return_value = ToolResult(
            success=True, data={"label": "positive"}
        )

        result = agent._execute_tool("sentiment_analyze", {"text": "好评"})

        assert result.success is True
        agent._sentiment_tool.execute.assert_called_once()

    def test_routes_pure_tool(self):
        agent = _make_agent()
        mock_preprocess = MagicMock()
        mock_preprocess.execute.return_value = ToolResult(
            success=True, data={"cleaned_text": "好评"}
        )
        agent.tools["text_preprocess"] = mock_preprocess

        result = agent._execute_tool("text_preprocess", {"text": "好评"})

        assert result.success is True

    def test_unknown_tool(self):
        agent = _make_agent()
        result = agent._execute_tool("nonexistent_tool", {})
        assert result.success is False
        assert "未知工具" in result.error

    def test_empty_text_for_llm_tool(self):
        agent = _make_agent()
        result = agent._execute_tool("keyword_extract", {"text": ""})
        assert result.success is False
        assert "为空" in result.error


# ══════════════════════════════════════════════════
# run() 主流程
# ══════════════════════════════════════════════════

class TestRun:
    def test_agent_mode_basic(self):
        """Agent 模式基本流程"""
        agent = _make_agent()
        agent._react_loop = MagicMock()
        agent._react_loop.run.return_value = "分析总结"

        with patch("extract_agent.agent.agent.assemble_result_from_memory") as mock_assemble:
            mock_assemble.return_value = {
                "analysis_complete": True,
                "keywords": [{"keyword": "好", "score": 0.9}],
                "sentiment": {"label": "positive"},
                "original_text": "好评",
            }
            result = agent.run("好评")

        assert result["mode"] in ("agent-native", "agent-prompt")
        assert result["analysis_complete"] is True
        assert "trace_id" in result
        assert "elapsed_ms" in result

    def test_token_limit_fallback_to_fast(self):
        """Token 超限时降级到 fast 模式"""
        agent = _make_agent()
        agent._react_loop = MagicMock()
        agent._react_loop.run.return_value = TOKEN_LIMIT_MARKER
        agent._fast_path = MagicMock()
        agent._fast_path.run_fast.return_value = {
            "analysis_complete": True, "keywords": [], "sentiment": None,
        }

        result = agent.run("好评")

        assert result["mode"] == "agent-native-fallback-fast"
        assert "warnings" in result

    def test_fast_path_mode(self):
        """use_fast_path=True 时走快速路径"""
        agent = _make_agent()
        agent._fast_path = MagicMock()
        agent._fast_path.run_fast.return_value = {
            "analysis_complete": True, "keywords": [], "mode": "fast",
        }

        result = agent.run("好评", use_fast_path=True)

        agent._fast_path.run_fast.assert_called_once_with("好评")
        assert "trace_id" in result

    def test_with_reflection(self):
        """启用反思时执行反思逻辑"""
        agent = _make_agent(enable_reflection=True)
        agent._react_loop = MagicMock()
        agent._react_loop.run.return_value = "总结"
        agent._reflector = MagicMock()

        with patch("extract_agent.agent.agent.assemble_result_from_memory") as mock_assemble:
            mock_assemble.return_value = {
                "analysis_complete": True,
                "keywords": [{"keyword": "好", "score": 0.9}],
                "sentiment": {"label": "positive"},
                "original_text": "好评",
            }
            agent._code_level_reflection = MagicMock(return_value=(
                mock_assemble.return_value,
                [{"passed": True, "type": "code_level"}],
            ))

            result = agent.run("好评")

        assert "reflection" in result
        assert result["reflection"]["total_rounds"] == 1


# ══════════════════════════════════════════════════
# run_stream()
# ══════════════════════════════════════════════════

class TestRunStream:
    def test_event_sequence(self):
        """流式模式的完整事件序列"""
        agent = _make_agent()
        agent._react_loop = MagicMock()
        agent._react_loop.run_stream.return_value = iter([
            {"type": "step_start", "step": 1},
            {"type": "token", "content": "分析"},
            {"type": "final_summary", "content": "分析完成"},
        ])

        with patch("extract_agent.agent.agent.assemble_result_from_memory") as mock_assemble:
            mock_assemble.return_value = {
                "analysis_complete": True,
                "keywords": [],
                "sentiment": None,
                "original_text": "好评",
            }
            events = list(agent.run_stream("好评"))

        types = [e["type"] for e in events]
        assert types[0] == "start"
        assert "step_start" in types
        assert "token" in types
        assert "final_summary" in types
        assert "result" in types
        assert types[-1] == "done"

    def test_stream_with_reflection(self):
        """流式模式下包含反思事件"""
        agent = _make_agent(enable_reflection=True)
        agent._react_loop = MagicMock()
        agent._react_loop.run_stream.return_value = iter([
            {"type": "final_summary", "content": "总结"},
        ])
        agent._reflector = MagicMock()

        with patch("extract_agent.agent.agent.assemble_result_from_memory") as mock_assemble:
            mock_assemble.return_value = {
                "analysis_complete": True,
                "keywords": [{"keyword": "好", "score": 0.9}],
                "sentiment": {"label": "positive"},
                "original_text": "好评",
            }
            agent._code_level_reflection = MagicMock(return_value=(
                mock_assemble.return_value,
                [{"passed": True}],
            ))

            events = list(agent.run_stream("好评"))

        types = [e["type"] for e in events]
        assert "reflection_start" in types
        assert "reflection_done" in types
        result_event = next(e for e in events if e["type"] == "result")
        assert "reflection" in result_event["data"]

    def test_stream_token_limit_fallback(self):
        """流式模式下 token 超限降级"""
        agent = _make_agent()
        agent._react_loop = MagicMock()
        agent._react_loop.run_stream.return_value = iter([
            {"type": "error", "content": TOKEN_LIMIT_MARKER},
        ])
        agent._fast_path = MagicMock()
        agent._fast_path.run_fast.return_value = {
            "analysis_complete": True, "keywords": [], "sentiment": None,
        }

        events = list(agent.run_stream("好评"))

        types = [e["type"] for e in events]
        assert "result" in types
        assert types[-1] == "done"
        result_event = next(e for e in events if e["type"] == "result")
        assert result_event["data"]["mode"] == "agent-stream-fallback-fast"

    def test_stream_offline_mode(self):
        """offline 模式下流式直接返回结果"""
        agent = _make_agent(backend_mode="offline")

        events = list(agent.run_stream("好评"))

        types = [e["type"] for e in events]
        assert types[0] == "start"
        assert "result" in types
        assert types[-1] == "done"


# ══════════════════════════════════════════════════
# run_offline / run_fast
# ══════════════════════════════════════════════════

class TestOfflineAndFast:
    def test_offline_delegates(self):
        agent = _make_agent(backend_mode="offline")
        agent._fast_path = MagicMock()
        agent._fast_path.run_offline.return_value = {
            "analysis_complete": True, "mode": "offline",
        }

        result = agent.run_offline("好评")

        agent._fast_path.run_offline.assert_called_once_with("好评")
        assert result["mode"] == "offline"

    def test_run_fast_delegates(self):
        agent = _make_agent()
        agent._fast_path = MagicMock()
        agent._fast_path.run_fast.return_value = {
            "analysis_complete": True, "mode": "fast",
        }

        result = agent.run_fast("好评")

        agent._fast_path.run_fast.assert_called_once_with("好评")

    def test_run_fast_offline_mode(self):
        """offline 后端模式下 run_fast 走 offline 路径"""
        agent = _make_agent(backend_mode="offline")
        agent._fast_path = MagicMock()
        agent._fast_path.run_offline.return_value = {
            "analysis_complete": True, "mode": "offline",
        }

        result = agent.run_fast("好评")

        agent._fast_path.run_offline.assert_called_once()


# ══════════════════════════════════════════════════
# run_batch
# ══════════════════════════════════════════════════

class TestRunBatch:
    def test_sequential_batch(self):
        """单线程顺序批量分析"""
        agent = _make_agent()
        agent.run = MagicMock(side_effect=[
            {"analysis_complete": True, "keywords": []},
            {"analysis_complete": True, "keywords": []},
        ])

        results = agent.run_batch(["评论1", "评论2"], max_workers=1)

        assert len(results) == 2
        assert all(r.get("batch_index") is not None for r in results)

    def test_concurrent_batch(self):
        """多线程并发批量分析"""
        agent = _make_agent()
        agent.run = MagicMock(side_effect=[
            {"analysis_complete": True, "keywords": []},
            {"analysis_complete": True, "keywords": []},
            {"analysis_complete": True, "keywords": []},
        ])

        results = agent.run_batch(["a", "b", "c"], max_workers=2)

        assert len(results) == 3
        indices = [r["batch_index"] for r in results]
        assert sorted(indices) == [0, 1, 2]

    def test_batch_exception_handling(self):
        """批量中单条异常不影响其他"""
        agent = _make_agent()
        agent.run = MagicMock(side_effect=[
            {"analysis_complete": True, "keywords": []},
            Exception("LLM 崩溃"),
        ])

        results = agent.run_batch(["a", "b"], max_workers=2)

        assert len(results) == 2
        error_result = next(r for r in results if not r.get("analysis_complete", False))
        assert "error" in error_result


# ══════════════════════════════════════════════════
# _code_level_reflection
# ══════════════════════════════════════════════════

class TestCodeLevelReflection:
    def test_passes_when_enough_keywords(self):
        """关键词充足时直接通过"""
        agent = _make_agent(enable_reflection=True)
        agent._reflector = MagicMock()

        result = {
            "original_text": "好评",
            "keywords": [
                {"keyword": "好", "score": 0.9},
                {"keyword": "评", "score": 0.8},
            ],
        }
        from extract_agent.agent.memory import AgentMemory
        memory = AgentMemory(system_prompt="sys")
        trace = []

        updated, history = agent._code_level_reflection(result, memory, trace, time.time())

        assert len(history) == 1
        assert history[0]["passed"] is True
        agent._reflector.reflect.assert_not_called()

    def test_triggers_llm_reflection_when_insufficient(self):
        """关键词不足时触发 LLM 反思"""
        agent = _make_agent(enable_reflection=True)
        mock_reflector = MagicMock()
        from extract_agent.agent.reflector import ReflectionResult
        mock_reflector.reflect.return_value = ReflectionResult(
            passed=True, issues=[],
            add_keywords=[{"keyword": "质量", "score": 0.85}],
            summary="补充了一个关键词",
        )
        agent._reflector = mock_reflector

        result = {
            "original_text": "这个质量很好",
            "keywords": [],
        }
        from extract_agent.agent.memory import AgentMemory
        memory = AgentMemory(system_prompt="sys")
        trace = []

        updated, history = agent._code_level_reflection(result, memory, trace, time.time())

        mock_reflector.reflect.assert_called()
        assert any(h.get("type") == "llm_supplement" for h in history)

    def test_score_filtering(self):
        """低分关键词被过滤"""
        agent = _make_agent(enable_reflection=True)
        agent._reflector = MagicMock()

        result = {
            "original_text": "好",
            "keywords": [
                {"keyword": "好", "score": 0.9},
                {"keyword": "低分", "score": 0.3},
            ],
        }
        from extract_agent.agent.memory import AgentMemory
        memory = AgentMemory(system_prompt="sys")
        trace = []

        updated, history = agent._code_level_reflection(result, memory, trace, time.time())

        kw_names = [kw["keyword"] for kw in updated["keywords"]]
        assert "低分" not in kw_names

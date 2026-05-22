"""
FastAPI 路由单元测试

使用 TestClient + mock Agent 测试所有 API 端点，
无需真实 LLM 服务。
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _reset_module_globals():
    """每个测试前重置 routes 模块的全局状态。"""
    from extract_agent.api import routes

    routes._agent = None
    routes._config = None
    routes._redis = None
    yield
    routes._agent = None
    routes._config = None
    routes._redis = None


def _make_mock_agent():
    agent = MagicMock()
    agent.tools = ["t1", "t2", "t3"]
    agent.run.return_value = {
        "analysis_complete": True,
        "original_text": "好评",
        "cleaned_text": "好评",
        "keywords": [["好评理由", "好评", 0.9]],
        "sentiment": {"label": "positive", "confidence": 0.95, "reasoning": "正向"},
        "summary": None,
        "reflection": None,
        "elapsed_ms": 42.0,
        "mode": "fast",
        "steps": None,
        "error": None,
    }
    return agent


@pytest.fixture
def client():
    from extract_agent.api.app import app

    mock_agent = _make_mock_agent()
    mock_cfg = MagicMock()
    mock_cfg.ENABLE_REFLECTION = True

    with patch("extract_agent.api.routes.get_agent", return_value=mock_agent), \
         patch("extract_agent.api.routes.get_config", return_value=mock_cfg), \
         patch("extract_agent.api.routes.get_redis", return_value=None):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, mock_agent, mock_cfg


# ====================================================================
# GET /health
# ====================================================================

class TestHealth:
    def test_health_check(self, client):
        c, mock_agent, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["agent_ready"] is True
        assert body["tools_loaded"] == 3

    def test_health_redis_disconnected(self, client):
        c, _, _ = client
        resp = c.get("/health")
        body = resp.json()
        assert body["redis_connected"] is False


# ====================================================================
# POST /api/analyze
# ====================================================================

class TestAnalyzeSingle:
    def test_analyze_fast_mode(self, client):
        c, mock_agent, _ = client
        resp = c.post("/api/analyze", json={"text": "好评", "mode": "fast"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["analysis_complete"] is True
        assert body["mode"] == "fast"
        assert len(body["keywords"]) == 1
        assert body["keywords"][0]["keyword"] == "好评"
        mock_agent.run.assert_called_once()

    def test_analyze_agent_mode(self, client):
        c, mock_agent, _ = client
        mock_agent.run.return_value["mode"] = "agent-native"
        resp = c.post("/api/analyze", json={"text": "质量很好做工精致", "mode": "agent"})
        assert resp.status_code == 200

    def test_analyze_auto_short_text(self, client):
        c, mock_agent, _ = client
        resp = c.post("/api/analyze", json={"text": "好", "mode": "auto"})
        assert resp.status_code == 200
        call_kwargs = mock_agent.run.call_args
        assert call_kwargs[1].get("use_fast_path") is True

    def test_analyze_enable_reflection_override(self, client):
        c, mock_agent, mock_cfg = client
        mock_cfg.ENABLE_REFLECTION = True
        resp = c.post("/api/analyze", json={"text": "不错", "mode": "fast", "enable_reflection": False})
        assert resp.status_code == 200
        assert mock_cfg.ENABLE_REFLECTION is False

    def test_analyze_empty_text_rejected(self, client):
        c, _, _ = client
        resp = c.post("/api/analyze", json={"text": "", "mode": "fast"})
        assert resp.status_code == 422

    def test_analyze_cache_hit(self, client):
        c, mock_agent, _ = client
        mock_redis = AsyncMock()
        mock_redis.get_cached_result = AsyncMock(return_value={
            "analysis_complete": True,
            "original_text": "缓存",
            "cleaned_text": "缓存",
            "keywords": [],
            "sentiment": None,
            "summary": None,
            "reflection": None,
            "elapsed_ms": 1.0,
            "mode": "fast",
            "steps": None,
            "error": None,
        })
        with patch("extract_agent.api.routes.get_redis", return_value=mock_redis):
            resp = c.post("/api/analyze", json={"text": "缓存", "mode": "fast"})
            assert resp.status_code == 200
            mock_agent.run.assert_not_called()


# ====================================================================
# POST /api/analyze/stream
# ====================================================================

class TestAnalyzeStream:
    def test_stream_returns_sse(self, client):
        c, mock_agent, _ = client
        events = [
            {"type": "start", "trace_id": "t1"},
            {"type": "token", "content": "hello"},
            {"type": "result", "data": {"analysis_complete": True, "mode": "agent-stream-native"}},
            {"type": "done"},
        ]
        mock_agent.run_stream = MagicMock(return_value=iter(events))

        resp = c.post(
            "/api/analyze/stream",
            json={"text": "好评", "mode": "agent"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        lines = resp.text.strip().split("\n")
        event_types = []
        for line in lines:
            if line.startswith("event:"):
                event_types.append(line.split(":", 1)[1].strip())

        assert "start" in event_types
        assert "done" in event_types

    def test_stream_error_propagation(self, client):
        c, mock_agent, _ = client

        def _failing_stream(**kwargs):
            yield {"type": "start", "trace_id": "t-err"}
            raise RuntimeError("boom")

        mock_agent.run_stream = MagicMock(side_effect=lambda **kw: _failing_stream(**kw))

        resp = c.post(
            "/api/analyze/stream",
            json={"text": "好评", "mode": "agent"},
        )
        assert resp.status_code == 200
        assert "error" in resp.text or "boom" in resp.text


# ====================================================================
# POST /api/analyze/batch
# ====================================================================

class TestAnalyzeBatch:
    def test_batch_two_items(self, client):
        c, mock_agent, _ = client
        mock_agent.run_batch.return_value = [
            {
                "analysis_complete": True,
                "original_text": "好",
                "cleaned_text": "好",
                "keywords": [],
                "sentiment": {"label": "positive", "confidence": 0.9, "reasoning": ""},
                "elapsed_ms": 10,
                "mode": "fast",
                "batch_index": 0,
            },
            {
                "analysis_complete": False,
                "original_text": "差",
                "cleaned_text": "差",
                "keywords": [],
                "sentiment": None,
                "elapsed_ms": 5,
                "mode": "fast",
                "error": "fail",
                "batch_index": 1,
            },
        ]

        resp = c.post(
            "/api/analyze/batch",
            json={"texts": ["好", "差"], "mode": "fast"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert body["completed"] == 1
        assert body["failed"] == 1
        assert len(body["results"]) == 2

    def test_batch_empty_rejected(self, client):
        c, _, _ = client
        resp = c.post("/api/analyze/batch", json={"texts": [], "mode": "fast"})
        assert resp.status_code == 422


# ====================================================================
# _normalize_result 格式化
# ====================================================================

class TestNormalizeResult:
    def test_normalize_list_keywords(self):
        from extract_agent.api.routes import _normalize_result

        raw = {
            "analysis_complete": True,
            "original_text": "test",
            "keywords": [["推理", "关键词", 0.9]],
            "sentiment": {"label": "positive", "confidence": 0.8, "reasoning": "ok"},
            "elapsed_ms": 10,
            "mode": "fast",
        }
        result = _normalize_result(raw)
        assert result["keywords"][0]["keyword"] == "关键词"
        assert result["keywords"][0]["score"] == 0.9

    def test_normalize_dict_keywords_passthrough(self):
        from extract_agent.api.routes import _normalize_result

        raw = {
            "analysis_complete": True,
            "original_text": "test",
            "keywords": [{"keyword": "已格式化", "reasoning": "", "score": 0.5}],
            "sentiment": None,
            "elapsed_ms": 5,
            "mode": "fast",
        }
        result = _normalize_result(raw)
        assert result["keywords"][0]["keyword"] == "已格式化"

    def test_normalize_sentiment_key_rename(self):
        from extract_agent.api.routes import _normalize_result

        raw = {
            "analysis_complete": True,
            "original_text": "t",
            "keywords": [],
            "sentiment": {"sentiment": "negative", "confidence": 0.7, "reasoning": "bad"},
            "elapsed_ms": 1,
            "mode": "fast",
        }
        result = _normalize_result(raw)
        assert result["sentiment"]["label"] == "negative"

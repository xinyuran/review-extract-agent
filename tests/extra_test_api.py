"""
API 集成测试

使用 FastAPI 的 TestClient 对各接口进行端到端测试。
需要 vLLM 服务运行时才能通过（标记为 integration）。

运行方式：
    # 仅运行不需要 LLM 服务的单元测试
    pytest tests/test_api.py -m "not integration"

    # 运行所有测试（需要 vLLM 服务）
    pytest tests/test_api.py
"""

import pytest

try:
    from fastapi.testclient import TestClient
    from ..api.app import app
    from ..api.schemas import AnalysisMode
    _IMPORT_OK = True
except ImportError as _exc:
    _IMPORT_OK = False
    _IMPORT_ERR = str(_exc)

pytestmark = pytest.mark.skipif(
    not _IMPORT_OK,
    reason=f"API 依赖缺失 ({_IMPORT_ERR if not _IMPORT_OK else ''}), 跳过 API 测试",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------
# 健康检查（不依赖 LLM）
# ------------------------------------------------------------------

class TestHealth:

    def test_health_endpoint(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert isinstance(data["tools_loaded"], int)


# ------------------------------------------------------------------
# Schema 验证（不依赖 LLM）
# ------------------------------------------------------------------

class TestSchemaValidation:

    def test_empty_text_rejected(self, client: TestClient):
        resp = client.post("/api/analyze", json={"text": ""})
        assert resp.status_code == 422

    def test_text_too_long_rejected(self, client: TestClient):
        resp = client.post("/api/analyze", json={"text": "x" * 2049})
        assert resp.status_code == 422

    def test_invalid_mode_rejected(self, client: TestClient):
        resp = client.post("/api/analyze", json={"text": "测试", "mode": "invalid"})
        assert resp.status_code == 422

    def test_batch_empty_list_rejected(self, client: TestClient):
        resp = client.post("/api/analyze/batch", json={"texts": []})
        assert resp.status_code == 422


# ------------------------------------------------------------------
# 单条分析（需要 LLM 服务）
# ------------------------------------------------------------------

@pytest.mark.integration
class TestAnalyzeSingle:

    def test_fast_mode(self, client: TestClient):
        resp = client.post("/api/analyze", json={
            "text": "做工很好，面料柔软",
            "mode": "fast",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis_complete"] is True
        assert data["mode"] == "fast"
        assert len(data["keywords"]) > 0
        assert data["sentiment"]["label"] in ("positive", "negative", "neutral")

    def test_agent_mode(self, client: TestClient):
        resp = client.post("/api/analyze", json={
            "text": "这件衣服质量非常好，做工精致，穿着很舒服。",
            "mode": "agent",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis_complete"] is True
        assert data["mode"] in ("agent-native", "agent-prompt")

    def test_negative_sentiment(self, client: TestClient):
        resp = client.post("/api/analyze", json={
            "text": "垃圾产品，用了一天就坏了，退货！",
            "mode": "fast",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentiment"]["label"] == "negative"

    def test_reflection_disabled(self, client: TestClient):
        resp = client.post("/api/analyze", json={
            "text": "还可以吧",
            "mode": "fast",
            "enable_reflection": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("reflection") is None


# ------------------------------------------------------------------
# 批量分析（需要 LLM 服务）
# ------------------------------------------------------------------

@pytest.mark.integration
class TestAnalyzeBatch:

    def test_batch_analysis(self, client: TestClient):
        resp = client.post("/api/analyze/batch", json={
            "texts": [
                "做工很好，面料柔软，穿着舒适",
                "垃圾产品，用了一天就坏了",
            ],
            "mode": "fast",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["completed"] == 2
        assert len(data["results"]) == 2

        assert data["results"][0]["batch_index"] == 0
        assert data["results"][1]["batch_index"] == 1


# ------------------------------------------------------------------
# 异步任务（需要 LLM + Redis）
# ------------------------------------------------------------------

@pytest.mark.integration
class TestAsyncTask:

    def test_submit_and_query(self, client: TestClient):
        resp = client.post("/api/task/submit", json={
            "texts": ["测试评论"],
            "mode": "fast",
        })

        if resp.status_code == 503:
            pytest.skip("Redis 不可用，跳过异步任务测试")

        assert resp.status_code == 200
        data = resp.json()
        task_id = data["task_id"]
        assert data["status"] == "pending"

        import time
        for _ in range(30):
            time.sleep(1)
            status_resp = client.get(f"/api/task/{task_id}")
            assert status_resp.status_code == 200
            status_data = status_resp.json()
            if status_data["status"] in ("completed", "failed"):
                break

        assert status_data["status"] == "completed"
        assert status_data["completed"] == 1

    def test_query_nonexistent_task(self, client: TestClient):
        resp = client.get("/api/task/nonexistent-id")
        if resp.status_code == 503:
            pytest.skip("Redis 不可用")
        assert resp.status_code == 404

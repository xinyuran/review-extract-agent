"""
OPT-09: API Key 时序攻击修复 — 单元测试

验证 APIKeyMiddleware 使用 hmac.compare_digest 进行常量时间密钥比较。
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _reset_routes():
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
    agent.tools = {"a": MagicMock(), "b": MagicMock()}
    return agent


@pytest.fixture
def client_with_key():
    """启用 API Key 认证的测试客户端"""
    with patch("extract_agent.api.app._API_KEY", "test-secret-key"):
        from extract_agent.api.app import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture
def client_no_key():
    """未启用 API Key 认证的测试客户端"""
    with patch("extract_agent.api.app._API_KEY", ""):
        from extract_agent.api.app import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestAPIKeySecurity:
    """测试 API Key 认证中间件的安全性"""

    def test_correct_key_passes_bearer(self, client_with_key):
        with patch("extract_agent.api.routes.get_agent", return_value=_make_mock_agent()):
            resp = client_with_key.get(
                "/health",
                headers={"Authorization": "Bearer test-secret-key"},
            )
        assert resp.status_code == 200

    def test_correct_key_passes_x_api_key(self, client_with_key):
        with patch("extract_agent.api.routes.get_agent", return_value=_make_mock_agent()):
            resp = client_with_key.get(
                "/health",
                headers={"X-API-Key": "test-secret-key"},
            )
        assert resp.status_code == 200

    def test_wrong_key_rejected_on_api_endpoint(self, client_with_key):
        """非跳过路径（/api/*）使用错误密钥应被拒绝"""
        resp = client_with_key.post(
            "/api/analyze",
            json={"text": "test"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_no_key_rejected_on_api_endpoint(self, client_with_key):
        """非跳过路径（/api/*）未提供密钥应被拒绝"""
        resp = client_with_key.post("/api/analyze", json={"text": "test"})
        assert resp.status_code == 401

    def test_skip_paths_bypass_auth(self, client_with_key):
        """/health 和 /docs 等路径跳过认证"""
        with patch("extract_agent.api.routes.get_agent", return_value=_make_mock_agent()):
            resp = client_with_key.get("/health")
        assert resp.status_code == 200

    def test_no_api_key_configured_allows_all(self, client_no_key):
        """未配置 API Key 时所有请求放行"""
        with patch("extract_agent.api.routes.get_agent", return_value=_make_mock_agent()):
            resp = client_no_key.get("/health")
        assert resp.status_code == 200

    def test_uses_hmac_compare_digest(self):
        """验证源码中使用了 hmac.compare_digest"""
        import inspect
        from extract_agent.api.app import APIKeyMiddleware
        source = inspect.getsource(APIKeyMiddleware.dispatch)
        assert "hmac.compare_digest" in source
        assert "provided_key != _API_KEY" not in source

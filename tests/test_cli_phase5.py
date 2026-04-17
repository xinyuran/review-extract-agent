"""
第五阶段测试 — serve + check

测试内容：
1. serve 命令参数解析
2. check 命令各检测项（mock HTTP/Redis）
"""

from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from extract_agent.cli.main import app

runner = CliRunner()


class TestServeCommand:
    """测试 serve 命令"""

    def test_serve_help(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "port" in result.output.lower() or "端口" in result.output

    @patch("extract_agent.cli.commands.serve.uvicorn")
    def test_serve_default_args(self, mock_uvicorn):
        """默认参数启动"""
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once_with(
            "extract_agent.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            workers=1,
        )

    @patch("extract_agent.cli.commands.serve.uvicorn")
    def test_serve_custom_port(self, mock_uvicorn):
        result = runner.invoke(app, ["serve", "--port", "9000"])
        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once()
        call_kwargs = mock_uvicorn.run.call_args[1]
        assert call_kwargs["port"] == 9000

    @patch("extract_agent.cli.commands.serve.uvicorn")
    def test_serve_with_reload(self, mock_uvicorn):
        result = runner.invoke(app, ["serve", "--reload"])
        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once()
        call_kwargs = mock_uvicorn.run.call_args[1]
        assert call_kwargs["reload"] is True
        assert call_kwargs["workers"] == 1


class TestCheckCommand:
    """测试 check 命令"""

    def test_check_help(self):
        result = runner.invoke(app, ["check", "--help"])
        assert result.exit_code == 0
        assert "检测" in result.output or "check" in result.output.lower()

    @patch("extract_agent.cli.commands.check.httpx")
    def test_check_runs(self, mock_httpx):
        """check 命令应能正常运行（即使连接失败）"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx.get.return_value = mock_response

        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0
        assert "配置文件" in result.output or "配置" in result.output

    @patch("extract_agent.cli.commands.check.httpx")
    def test_check_llm_connected(self, mock_httpx):
        """LLM 连通时显示成功"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx.get.return_value = mock_response

        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0
        assert "连通" in result.output or "✓" in result.output

    @patch("extract_agent.cli.commands.check.httpx")
    def test_check_llm_failure(self, mock_httpx):
        """LLM 不可用时显示错误"""
        mock_httpx.get.side_effect = ConnectionError("refused")

        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0
        assert "不可用" in result.output or "✗" in result.output

    def test_check_shows_fc_mode(self):
        """应显示 Function Calling 模式"""
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0
        assert "Function Calling" in result.output

    def test_check_shows_tools(self):
        """应显示工具加载状态"""
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0
        assert "工具" in result.output

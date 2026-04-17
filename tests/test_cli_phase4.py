"""
第四阶段测试 — 交互式 REPL

测试内容：
1. REPL 内置命令解析逻辑
2. session 历史记录
3. mode 切换
4. interactive 命令 --help
"""

from unittest.mock import patch, MagicMock, PropertyMock

import pytest
from typer.testing import CliRunner

from extract_agent.cli.commands.interactive import (
    _handle_command,
    _cmd_mode,
    _cmd_full,
)
from extract_agent.cli.session import CLISession
from extract_agent.cli.main import app
from extract_agent.config import AgentConfig

runner = CliRunner()

MOCK_RESULT = {
    "original_text": "测试评论",
    "cleaned_text": "测试评论",
    "keywords": [
        {"keyword": "测试", "reasoning": "测试", "score": 0.9},
    ],
    "sentiment": {"label": "neutral", "confidence": 0.8, "reasoning": "中性"},
    "analysis_complete": True,
    "elapsed_ms": 100,
    "steps": 3,
    "mode": "fast",
}


class TestInteractiveHelp:
    """测试 interactive --help"""

    def test_interactive_help(self):
        result = runner.invoke(app, ["interactive", "--help"])
        assert result.exit_code == 0
        assert "交互" in result.output or "REPL" in result.output


class TestReplCommands:
    """测试 REPL 内置命令"""

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def _make_session(self, mock_cls):
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_cls.return_value = mock_agent
        return CLISession(config=AgentConfig(), mode="agent")

    def test_exit_command(self):
        session = self._make_session()
        assert _handle_command("/exit", session) is True

    def test_quit_command(self):
        session = self._make_session()
        assert _handle_command("/quit", session) is True

    def test_help_command(self):
        session = self._make_session()
        assert _handle_command("/help", session) is False

    def test_unknown_command(self):
        session = self._make_session()
        assert _handle_command("/unknown", session) is False

    def test_history_command_empty(self):
        session = self._make_session()
        assert _handle_command("/history", session) is False

    def test_session_command(self):
        session = self._make_session()
        assert _handle_command("/session", session) is False


class TestModeSwitch:
    """测试 /mode 命令"""

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_switch_to_fast(self, mock_cls):
        mock_cls.return_value = MagicMock()
        session = CLISession(config=AgentConfig(), mode="agent")

        _cmd_mode("fast", session)
        assert session.mode == "fast"

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_switch_to_agent(self, mock_cls):
        mock_cls.return_value = MagicMock()
        session = CLISession(config=AgentConfig(), mode="fast")

        _cmd_mode("agent", session)
        assert session.mode == "agent"

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_invalid_mode(self, mock_cls):
        mock_cls.return_value = MagicMock()
        session = CLISession(config=AgentConfig(), mode="agent")

        _cmd_mode("invalid", session)
        assert session.mode == "agent"  # 不应改变


class TestFullSwitch:
    """测试 /full 命令"""

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_full_on(self, mock_cls):
        mock_cls.return_value = MagicMock()
        session = CLISession(config=AgentConfig(), mode="fast")
        assert session.full_output is False

        _cmd_full("on", session)
        assert session.full_output is True

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_full_off(self, mock_cls):
        mock_cls.return_value = MagicMock()
        session = CLISession(config=AgentConfig(), mode="fast", full_output=True)

        _cmd_full("off", session)
        assert session.full_output is False

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_full_invalid(self, mock_cls):
        mock_cls.return_value = MagicMock()
        session = CLISession(config=AgentConfig(), mode="fast")

        _cmd_full("maybe", session)
        assert session.full_output is False  # 不应改变


class TestSessionHistory:
    """测试 session 历史记录"""

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_history_after_analyze(self, mock_cls):
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_cls.return_value = mock_agent

        session = CLISession(config=AgentConfig(), mode="fast")
        session.analyze("第一条评论")
        session.analyze("第二条评论")

        history = session.get_history_summary()
        assert len(history) == 2
        assert history[0]["index"] == 1
        assert history[1]["index"] == 2
        assert "第一条" in history[0]["text_preview"]

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_history_empty(self, mock_cls):
        mock_cls.return_value = MagicMock()
        session = CLISession(config=AgentConfig(), mode="fast")

        history = session.get_history_summary()
        assert len(history) == 0

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_session_info_reflects_count(self, mock_cls):
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_cls.return_value = mock_agent

        session = CLISession(config=AgentConfig(), mode="fast")
        session.analyze("评论")
        session.analyze("评论")

        info = session.get_session_info()
        assert info["total_analyzed"] == 2

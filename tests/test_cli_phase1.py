"""
第一阶段测试 — 基础骨架

测试内容：
1. python -m extract_agent --help 能输出帮助
2. python -m extract_agent analyze --help 能输出帮助
3. analyze 命令参数解析
4. formatter 输出（mock Agent）
"""

import json
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from extract_agent.cli.main import app
from extract_agent.cli.formatter import format_single_result, format_full_json

runner = CliRunner()

MOCK_RESULT = {
    "original_text": "这件衣服质量很好，做工精致",
    "cleaned_text": "这件衣服质量很好做工精致",
    "keywords": [
        {"keyword": "质量很好", "reasoning": "正面评价", "score": 0.95},
        {"keyword": "做工精致", "reasoning": "工艺描述", "score": 0.92},
    ],
    "sentiment": {
        "label": "positive",
        "confidence": 0.96,
        "reasoning": "整体正面评价",
    },
    "analysis_complete": True,
    "elapsed_ms": 2340.5,
    "steps": 5,
    "mode": "agent-native",
}


class TestHelpOutput:
    """测试 --help 输出"""

    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "extract-agent" in result.output or "中文电商评论" in result.output

    def test_analyze_help(self):
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "分析" in result.output or "analyze" in result.output.lower()


class TestAnalyzeCommand:
    """测试 analyze 命令参数解析与执行"""

    def test_no_text_exits_with_error(self):
        """未提供文本时应报错退出"""
        result = runner.invoke(app, ["analyze"])
        assert result.exit_code != 0

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_analyze_basic_text(self, mock_agent_cls):
        """基本文本分析"""
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_agent_cls.return_value = mock_agent

        result = runner.invoke(app, ["analyze", "这件衣服质量很好"])
        assert result.exit_code == 0
        assert "质量很好" in result.output

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_analyze_fast_mode(self, mock_agent_cls):
        """fast 模式应传递 use_fast_path=True"""
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_agent_cls.return_value = mock_agent

        result = runner.invoke(app, ["analyze", "评论", "--mode", "fast"])
        assert result.exit_code == 0
        mock_agent.run.assert_called_once_with("评论", use_fast_path=True)

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_analyze_agent_mode(self, mock_agent_cls):
        """agent 模式应传递 use_fast_path=False"""
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_agent_cls.return_value = mock_agent

        result = runner.invoke(app, ["analyze", "评论", "--mode", "agent"])
        assert result.exit_code == 0
        mock_agent.run.assert_called_once_with("评论", use_fast_path=False)

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_analyze_no_reflect(self, mock_agent_cls):
        """--no-reflect 应禁用反思"""
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_agent_cls.return_value = mock_agent

        result = runner.invoke(app, ["analyze", "评论", "--no-reflect"])
        assert result.exit_code == 0

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_analyze_full_json(self, mock_agent_cls):
        """--full 应输出完整 JSON"""
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_agent_cls.return_value = mock_agent

        result = runner.invoke(app, ["analyze", "评论", "--full"])
        assert result.exit_code == 0
        assert "original_text" in result.output
        assert "keywords" in result.output

    def test_invalid_mode(self):
        """无效模式应报错"""
        result = runner.invoke(app, ["analyze", "评论", "--mode", "invalid"])
        assert result.exit_code != 0


class TestFormatter:
    """测试 formatter 输出函数"""

    def test_format_single_result_runs(self):
        """format_single_result 不应抛出异常"""
        format_single_result(MOCK_RESULT, "测试评论")

    def test_format_full_json_runs(self):
        """format_full_json 不应抛出异常"""
        format_full_json(MOCK_RESULT)

    def test_format_single_result_empty_keywords(self):
        """空关键词列表不应崩溃"""
        result = {**MOCK_RESULT, "keywords": []}
        format_single_result(result, "测试评论")

    def test_format_single_result_missing_sentiment(self):
        """缺失 sentiment 字段不应崩溃"""
        result = {**MOCK_RESULT, "sentiment": {}}
        format_single_result(result, "测试评论")

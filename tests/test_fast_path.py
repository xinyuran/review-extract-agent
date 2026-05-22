"""FastPathExecutor 单元测试

覆盖场景：
- run_offline: 仅使用 jieba，不调用 LLM 工具
- run_fast: 按管线顺序执行（预处理 → 关键词提取 → 校验 → 情感分析）
- run_fast: 关键词提取失败时 fallback 到 jieba
- _format_keywords: 三元组格式转换
"""

from unittest.mock import MagicMock, call

import pytest

from extract_agent.agent.fast_path import FastPathExecutor
from extract_agent.tools.base_tool import ToolResult


# ── helpers ──

def _make_tool_result(success=True, data=None):
    return ToolResult(success=success, data=data or {})


def _make_executor(tool_responses=None):
    """创建带 mock execute_tool 的 executor"""
    responses = list(tool_responses or [])
    execute_tool = MagicMock(side_effect=responses)
    return FastPathExecutor(execute_tool), execute_tool


# ══════════════════════════════════════════════════
# _format_keywords
# ══════════════════════════════════════════════════

class TestFormatKeywords:
    def test_normal_triplets(self):
        keywords = [
            ["推理说明", "关键词A", 0.9],
            ["推理说明", "关键词B", 0.8],
        ]
        result = FastPathExecutor._format_keywords(keywords)
        assert len(result) == 2
        assert result[0] == {"keyword": "关键词A", "reasoning": "推理说明", "score": 0.9}

    def test_skips_invalid_items(self):
        keywords = [
            ["推理说明", "关键词A", 0.9],
            "invalid",
            ["too", "short"],
        ]
        result = FastPathExecutor._format_keywords(keywords)
        assert len(result) == 1

    def test_empty_list(self):
        assert FastPathExecutor._format_keywords([]) == []


# ══════════════════════════════════════════════════
# run_offline
# ══════════════════════════════════════════════════

class TestRunOffline:
    def test_calls_correct_tools(self):
        """offline 模式只调用 preprocess + jieba + validate"""
        executor, mock_tool = _make_executor([
            _make_tool_result(data={"cleaned_text": "清洗后文本"}),
            _make_tool_result(data={"keywords": [["理由", "关键词", 0.9]]}),
            _make_tool_result(data={"valid_keywords": [["理由", "关键词", 0.9]]}),
        ])

        result = executor.run_offline("原始评论")

        assert result["analysis_complete"] is True
        assert result["mode"] == "offline"
        assert result["sentiment"] is None
        assert len(result["keywords"]) == 1

        called_tools = [c.args[0] for c in mock_tool.call_args_list]
        assert called_tools == ["text_preprocess", "jieba_extract", "validate_keywords"]

    def test_preprocess_failure_uses_raw_text(self):
        """预处理失败时使用原始文本"""
        executor, _ = _make_executor([
            _make_tool_result(success=False),
            _make_tool_result(data={"keywords": []}),
        ])

        result = executor.run_offline("  原始  ")

        assert result["cleaned_text"] == "原始"

    def test_no_keywords_still_completes(self):
        """jieba 返回空时仍标记完成"""
        executor, _ = _make_executor([
            _make_tool_result(data={"cleaned_text": "x"}),
            _make_tool_result(data={"keywords": []}),
        ])

        result = executor.run_offline("x")

        assert result["analysis_complete"] is True
        assert result["keywords"] == []


# ══════════════════════════════════════════════════
# run_fast
# ══════════════════════════════════════════════════

class TestRunFast:
    def test_full_pipeline(self):
        """fast 模式完整管线：preprocess → keyword → validate → sentiment"""
        executor, mock_tool = _make_executor([
            _make_tool_result(data={"cleaned_text": "清洗后文本"}),
            _make_tool_result(data={"keywords": [["理由", "好", 0.95]]}),
            _make_tool_result(data={"valid_keywords": [["理由", "好", 0.95]]}),
            _make_tool_result(data={"label": "positive", "confidence": 0.9, "reasoning": "正面"}),
        ])

        result = executor.run_fast("好评好评")

        assert result["analysis_complete"] is True
        assert result["mode"] == "fast"
        assert len(result["keywords"]) == 1
        assert result["sentiment"]["label"] == "positive"

        called_tools = [c.args[0] for c in mock_tool.call_args_list]
        assert called_tools == [
            "text_preprocess", "keyword_extract", "validate_keywords", "sentiment_analyze"
        ]

    def test_keyword_extract_failure_falls_back_to_jieba(self):
        """关键词提取失败时 fallback 到 jieba"""
        executor, mock_tool = _make_executor([
            _make_tool_result(data={"cleaned_text": "文本"}),
            _make_tool_result(success=False, data={}),
            _make_tool_result(data={"keywords": [["jieba", "关键词", 0.7]]}),
            _make_tool_result(data={"valid_keywords": [["jieba", "关键词", 0.7]]}),
            _make_tool_result(data={"label": "neutral", "confidence": 0.5}),
        ])

        result = executor.run_fast("文本")

        assert len(result["keywords"]) == 1
        called_tools = [c.args[0] for c in mock_tool.call_args_list]
        assert "jieba_extract" in called_tools

    def test_sentiment_failure(self):
        """情感分析失败时返回 unknown"""
        executor, _ = _make_executor([
            _make_tool_result(data={"cleaned_text": "文本"}),
            _make_tool_result(data={"keywords": [["理由", "好", 0.9]]}),
            _make_tool_result(data={"valid_keywords": [["理由", "好", 0.9]]}),
            _make_tool_result(success=False),
        ])

        result = executor.run_fast("文本")

        assert result["sentiment"]["label"] == "unknown"

    def test_elapsed_ms_present(self):
        executor, _ = _make_executor([
            _make_tool_result(data={"cleaned_text": "x"}),
            _make_tool_result(data={"keywords": []}),
            _make_tool_result(data={"label": "neutral", "confidence": 0.5}),
        ])

        result = executor.run_fast("x")

        assert "elapsed_ms" in result
        assert isinstance(result["elapsed_ms"], float)

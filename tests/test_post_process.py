"""
core/post_process.py 单元测试

验证 post_process_keywords 和 post_process_keywords_with_config 的行为，
重点测试 config wrapper 与原版函数的一致性。
"""

import pytest

from extract_agent.config import PostprocessConfig
from extract_agent.core.post_process import (
    post_process_keywords,
    post_process_keywords_with_config,
)


SAMPLE_KEYWORDS = [
    ["商品质量很好，用户明确表达了满意", "质量好", 0.95],
    ["用户提到了快递速度", "快递快", 0.90],
    ["物流包装完好", "包装好", 0.85],
    ["价格实惠", "性价比高", 0.80],
    ["质量好的同义词", "质量好", 0.75],
]

ORIGINAL_TEXT = "这个商品质量好，快递快，包装好，性价比高"


class TestPostProcessKeywords:

    def test_empty_input(self):
        assert post_process_keywords([]) == []
        assert post_process_keywords(None) == []

    def test_deduplication(self):
        result = post_process_keywords(
            SAMPLE_KEYWORDS,
            deduplicate=True,
            sort_by_importance=True,
            top_n=False,
            return_full_info=False,
            filter_stopwords=False,
            filter_time_keywords=False,
            filter_date_keywords=False,
            filter_long_keywords=False,
            filter_not_in_original=False,
        )
        assert result.count("质量好") == 1

    def test_sort_by_importance(self):
        result = post_process_keywords(
            SAMPLE_KEYWORDS,
            deduplicate=False,
            sort_by_importance=True,
            top_n=False,
            return_full_info=False,
            filter_stopwords=False,
            filter_time_keywords=False,
            filter_date_keywords=False,
            filter_long_keywords=False,
            filter_not_in_original=False,
        )
        assert result[0] == "质量好"

    def test_top_n_limits_count(self):
        result = post_process_keywords(
            SAMPLE_KEYWORDS,
            deduplicate=True,
            sort_by_importance=True,
            top_n=True,
            n=2,
            return_full_info=False,
            filter_stopwords=False,
            filter_time_keywords=False,
            filter_date_keywords=False,
            filter_long_keywords=False,
            filter_not_in_original=False,
        )
        assert len(result) == 2

    def test_return_full_info(self):
        result = post_process_keywords(
            SAMPLE_KEYWORDS,
            deduplicate=True,
            sort_by_importance=True,
            top_n=False,
            return_full_info=True,
            filter_stopwords=False,
            filter_time_keywords=False,
            filter_date_keywords=False,
            filter_long_keywords=False,
            filter_not_in_original=False,
        )
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert len(result[0]) == 3

    def test_filter_long_keywords(self):
        data = [
            ["reason", "短词", 0.9],
            ["reason", "这是一个超级超级超级长的关键词", 0.8],
        ]
        result = post_process_keywords(
            data,
            filter_long_keywords=True,
            max_keyword_length=6,
            return_full_info=False,
            top_n=False,
            filter_stopwords=False,
            filter_time_keywords=False,
            filter_date_keywords=False,
            filter_not_in_original=False,
        )
        assert "短词" in result
        assert "这是一个超级超级超级长的关键词" not in result

    def test_filter_not_in_original(self):
        data = [
            ["reason", "质量好", 0.9],
            ["reason", "不存在词", 0.8],
        ]
        result = post_process_keywords(
            data,
            filter_not_in_original=True,
            original_text=ORIGINAL_TEXT,
            return_full_info=False,
            top_n=False,
            filter_stopwords=False,
            filter_time_keywords=False,
            filter_date_keywords=False,
            filter_long_keywords=False,
        )
        assert "质量好" in result
        assert "不存在词" not in result


class TestPostProcessKeywordsWithConfig:

    def test_config_wrapper_matches_direct_call(self):
        pp = PostprocessConfig(
            deduplicate=True,
            sort_by_importance=True,
            top_n=False,
            return_full_info=False,
            filter_stopwords=False,
            filter_time_keywords=False,
            filter_date_keywords=False,
            filter_long_keywords=False,
            filter_not_in_original=False,
        )
        direct = post_process_keywords(
            SAMPLE_KEYWORDS,
            deduplicate=True,
            sort_by_importance=True,
            top_n=False,
            return_full_info=False,
            filter_stopwords=False,
            filter_time_keywords=False,
            filter_date_keywords=False,
            filter_long_keywords=False,
            filter_not_in_original=False,
        )
        via_config = post_process_keywords_with_config(
            SAMPLE_KEYWORDS,
            config=pp,
        )
        assert direct == via_config

    def test_empty_input_with_config(self):
        pp = PostprocessConfig()
        assert post_process_keywords_with_config([], config=pp) == []

    def test_max_keywords_override(self):
        pp = PostprocessConfig(
            top_n=True,
            n=10,
            return_full_info=False,
            filter_stopwords=False,
            filter_time_keywords=False,
            filter_date_keywords=False,
            filter_long_keywords=False,
            filter_not_in_original=False,
        )
        result = post_process_keywords_with_config(
            SAMPLE_KEYWORDS,
            config=pp,
            max_keywords=2,
        )
        assert len(result) == 2

    def test_original_text_passed_through(self):
        pp = PostprocessConfig(
            filter_not_in_original=True,
            return_full_info=False,
            top_n=False,
            filter_stopwords=False,
            filter_time_keywords=False,
            filter_date_keywords=False,
            filter_long_keywords=False,
        )
        data = [
            ["reason", "质量好", 0.9],
            ["reason", "完全不存在", 0.8],
        ]
        result = post_process_keywords_with_config(
            data,
            config=pp,
            original_text=ORIGINAL_TEXT,
        )
        assert "质量好" in result
        assert "完全不存在" not in result

    def test_accepts_agent_config(self):
        from extract_agent.config import AgentConfig
        cfg = AgentConfig()
        cfg.postprocess.filter_stopwords = False
        cfg.postprocess.filter_time_keywords = False
        cfg.postprocess.filter_date_keywords = False
        cfg.postprocess.filter_long_keywords = False
        cfg.postprocess.filter_not_in_original = False
        cfg.postprocess.top_n = False
        cfg.postprocess.return_full_info = False

        result = post_process_keywords_with_config(
            SAMPLE_KEYWORDS,
            config=cfg,
        )
        assert isinstance(result, list)
        assert len(result) > 0

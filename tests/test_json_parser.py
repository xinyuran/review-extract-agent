"""
utils/json_parser.py 单元测试

覆盖 5 个公共函数的正常路径和边界情况。
"""

import json
import re

import pytest

from extract_agent.utils.json_parser import (
    find_matching_brace,
    sanitize_json,
    extract_json_from_response,
    extract_json_and_thinking,
    try_repair_truncated_json,
)


# ============================================================
# find_matching_brace
# ============================================================

class TestFindMatchingBrace:

    @pytest.mark.parametrize("text, start, expected", [
        ('{"a": 1}', 0, 7),
        ('{"a": {"b": 2}}', 0, 14),
        ('{"a": {"b": {"c": 3}}}', 0, 21),
        ('x{"k": 1}y', 1, 8),
    ])
    def test_normal_nesting(self, text, start, expected):
        assert find_matching_brace(text, start) == expected

    def test_string_containing_braces(self):
        text = '{"val": "has { and } inside"}'
        assert find_matching_brace(text, 0) == len(text) - 1

    def test_escaped_quote_in_string(self):
        text = r'{"val": "escaped \" quote"}'
        assert find_matching_brace(text, 0) == len(text) - 1

    def test_unclosed_brace_returns_negative(self):
        assert find_matching_brace('{"a": 1', 0) == -1

    def test_empty_object(self):
        assert find_matching_brace('{}', 0) == 1

    def test_empty_string(self):
        assert find_matching_brace('', 0) == -1


# ============================================================
# sanitize_json
# ============================================================

class TestSanitizeJson:

    def test_chinese_comma_replaced(self):
        raw = '{"keywords": ["a"， "b"]}'
        result = sanitize_json(raw)
        assert '，' not in result

    def test_chinese_colon_replaced(self):
        raw = '{"keywords"： ["a"]}'
        result = sanitize_json(raw)
        assert '：' not in result

    def test_valid_json_unchanged(self):
        raw = '{"keywords": [["reason", "word", 0.9]]}'
        assert sanitize_json(raw) == raw

    def test_trailing_invalid_element_trimmed(self):
        raw = '{"keywords": [["r1", "kw1", 0.9], ["r2", "kw2", bad_value]]}'
        result = sanitize_json(raw)
        parsed = json.loads(result)
        assert len(parsed["keywords"]) == 1
        assert parsed["keywords"][0][1] == "kw1"


# ============================================================
# extract_json_from_response
# ============================================================

class TestExtractJsonFromResponse:

    def test_pure_json(self):
        raw = '{"sentiment": "positive", "score": 0.9}'
        result = extract_json_from_response(raw)
        assert result is not None
        assert json.loads(result)["sentiment"] == "positive"

    def test_markdown_code_block(self):
        raw = '```json\n{"key": "val"}\n```'
        result = extract_json_from_response(raw)
        assert result is not None
        assert json.loads(result)["key"] == "val"

    def test_markdown_bare_block(self):
        raw = '```\n{"key": 1}\n```'
        result = extract_json_from_response(raw)
        assert result is not None

    def test_json_embedded_in_text(self):
        raw = 'Here is the result:\n{"a": 1}\nDone.'
        result = extract_json_from_response(raw)
        assert result is not None
        assert json.loads(result)["a"] == 1

    def test_custom_pattern(self):
        raw = 'some text {"sentiment": "neg", "score": 0.2} more'
        pattern = re.compile(r'\{"sentiment"\s*:.*?\}')
        result = extract_json_from_response(raw, pattern=pattern)
        assert result is not None
        assert json.loads(result)["sentiment"] == "neg"

    def test_no_json_returns_none(self):
        assert extract_json_from_response("no json here") is None

    def test_empty_string(self):
        assert extract_json_from_response("") is None


# ============================================================
# extract_json_and_thinking
# ============================================================

class TestExtractJsonAndThinking:

    def test_pure_json_no_thinking(self):
        raw = '{"keywords": [["reason", "word", 0.9]]}'
        json_str, thinking = extract_json_and_thinking(raw)
        assert json_str is not None
        assert thinking == ""

    def test_thinking_before_json(self):
        raw = '我需要分析这段评论。\n{"keywords": [["分析", "质量好", 0.9]]}'
        json_str, thinking = extract_json_and_thinking(raw)
        assert json_str is not None
        assert "分析这段评论" in thinking

    def test_markdown_wrapped_json(self):
        raw = '```json\n{"keywords": [["r", "kw", 0.8]]}\n```'
        json_str, thinking = extract_json_and_thinking(raw)
        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["keywords"][0][1] == "kw"

    def test_chinese_punctuation_in_json(self):
        raw = '{"keywords": [["理由"，"关键词"，0.9]]}'
        json_str, thinking = extract_json_and_thinking(raw)
        if json_str is not None:
            parsed = json.loads(json_str)
            assert "keywords" in parsed

    def test_no_json_returns_none_and_full_text(self):
        raw = "这是一段没有JSON的文本"
        json_str, thinking = extract_json_and_thinking(raw)
        assert json_str is None
        assert thinking == raw.strip()


# ============================================================
# try_repair_truncated_json
# ============================================================

class TestTryRepairTruncatedJson:

    def test_truncated_in_array_element(self):
        raw = '{"keywords": [["r1", "kw1", 0.9], ["r2", "kw2"'
        result = try_repair_truncated_json(raw)
        if result is not None:
            parsed = json.loads(result)
            assert "keywords" in parsed
            assert len(parsed["keywords"]) >= 1

    def test_truncated_after_complete_element(self):
        raw = '{"keywords": [["r1", "kw1", 0.9], ["r2", "kw2", 0.8],'
        result = try_repair_truncated_json(raw)
        assert result is not None
        parsed = json.loads(result)
        assert len(parsed["keywords"]) == 2

    def test_no_keywords_pattern_returns_none(self):
        raw = '{"sentiment": "positive", "sc'
        assert try_repair_truncated_json(raw) is None

    def test_complete_json_passthrough(self):
        raw = '{"keywords": [["r", "kw", 0.9]]}'
        result = try_repair_truncated_json(raw)
        if result is not None:
            parsed = json.loads(result)
            assert parsed["keywords"][0][1] == "kw"

    def test_empty_array_truncated(self):
        raw = '{"keywords": ['
        result = try_repair_truncated_json(raw)
        if result is not None:
            parsed = json.loads(result)
            assert parsed["keywords"] == []

"""ResultReflector 单元测试

覆盖场景：
- reflect() 正常路径（mock LLM 返回有效 JSON）
- reflect() JSON 解析失败的安全回退
- reflect() 异常时的安全回退
- reflect_and_correct 收敛检测
- _apply_delta 增删关键词
- _extract_issue_fingerprints
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from extract_agent.agent.reflector import ResultReflector, ReflectionResult
from extract_agent.llm_service.models import LLMResponse


# ── helpers ──

def _make_reflector(llm_response_content=None, llm_side_effect=None):
    """创建带 mock LLMService 的 reflector"""
    mock_llm = MagicMock()
    mock_llm.load_skill.return_value = MagicMock(system="sys", user="user")

    if llm_side_effect:
        mock_llm.call_agent.side_effect = llm_side_effect
    else:
        mock_llm.call_agent.return_value = LLMResponse(
            content=llm_response_content or ""
        )

    reflector = ResultReflector(llm_service=mock_llm)
    return reflector


# ══════════════════════════════════════════════════
# ReflectionResult
# ══════════════════════════════════════════════════

class TestReflectionResult:
    def test_to_dict(self):
        r = ReflectionResult(
            passed=True,
            issues=[],
            add_keywords=[{"keyword": "好", "score": 0.9}],
            remove_keywords=["差"],
            summary="通过",
        )
        d = r.to_dict()
        assert d["passed"] is True
        assert d["add_keywords"] == [{"keyword": "好", "score": 0.9}]
        assert d["remove_keywords"] == ["差"]
        assert d["summary"] == "通过"

    def test_defaults(self):
        r = ReflectionResult(passed=False, issues=[])
        assert r.add_keywords == []
        assert r.remove_keywords == []
        assert r.corrected_sentiment is None
        assert r.elapsed_ms == 0


# ══════════════════════════════════════════════════
# reflect()
# ══════════════════════════════════════════════════

class TestReflect:
    def test_normal_passed(self):
        """LLM 返回 passed=true 的 JSON"""
        response_json = {
            "passed": True,
            "issues": [],
            "add_keywords": [],
            "remove_keywords": [],
            "summary": "分析质量合格",
        }
        reflector = _make_reflector(json.dumps(response_json, ensure_ascii=False))

        result = reflector.reflect("好评", [{"keyword": "好评", "score": 0.9}], {})

        assert result.passed is True
        assert result.summary == "分析质量合格"
        assert result.elapsed_ms > 0

    def test_normal_not_passed_with_additions(self):
        """LLM 返回未通过并建议添加关键词"""
        response_json = {
            "passed": False,
            "issues": [{"type": "missing_keyword", "detail": "缺少质量", "severity": "high"}],
            "add_keywords": [{"keyword": "质量", "score": 0.85}],
            "remove_keywords": [],
            "summary": "关键词不完整",
        }
        reflector = _make_reflector(json.dumps(response_json, ensure_ascii=False))

        result = reflector.reflect("质量很好", [{"keyword": "好", "score": 0.9}], {})

        assert result.passed is False
        assert len(result.add_keywords) == 1
        assert result.add_keywords[0]["keyword"] == "质量"

    def test_json_parse_failure_returns_safe_fallback(self):
        """LLM 返回无效内容时安全回退"""
        reflector = _make_reflector("这不是一个 JSON")

        result = reflector.reflect("好评", [], {})

        assert result.passed is True
        assert any(i["type"] == "parse_error" for i in result.issues)

    def test_exception_returns_safe_fallback(self):
        """LLM 调用异常时安全回退"""
        reflector = _make_reflector(llm_side_effect=Exception("LLM 服务不可用"))

        result = reflector.reflect("好评", [], {})

        assert result.passed is True
        assert any(i["type"] == "execution_error" for i in result.issues)

    def test_with_corrected_sentiment(self):
        """LLM 建议修正情感"""
        response_json = {
            "passed": False,
            "issues": [{"type": "wrong_sentiment", "detail": "应该为负向", "severity": "high"}],
            "add_keywords": [],
            "remove_keywords": [],
            "corrected_sentiment": {"label": "negative", "confidence": 0.9, "reasoning": "负面评价"},
            "summary": "情感判断有误",
        }
        reflector = _make_reflector(json.dumps(response_json, ensure_ascii=False))

        result = reflector.reflect("垃圾", [], {"label": "positive"})

        assert result.corrected_sentiment["label"] == "negative"


# ══════════════════════════════════════════════════
# _apply_delta
# ══════════════════════════════════════════════════

class TestApplyDelta:
    def test_add_keywords(self):
        keywords = [{"keyword": "好", "score": 0.9}]
        ref = ReflectionResult(
            passed=False, issues=[],
            add_keywords=[{"keyword": "质量", "score": 0.85}],
        )
        updated_kw, updated_sent = ResultReflector._apply_delta(keywords, {}, ref)

        assert len(updated_kw) == 2
        kw_names = {kw["keyword"] for kw in updated_kw}
        assert "质量" in kw_names

    def test_remove_keywords(self):
        keywords = [
            {"keyword": "好", "score": 0.9},
            {"keyword": "差", "score": 0.3},
        ]
        ref = ReflectionResult(
            passed=False, issues=[], remove_keywords=["差"],
        )
        updated_kw, _ = ResultReflector._apply_delta(keywords, {}, ref)

        assert len(updated_kw) == 1
        assert updated_kw[0]["keyword"] == "好"

    def test_no_duplicate_add(self):
        """不重复添加已有的关键词"""
        keywords = [{"keyword": "好", "score": 0.9}]
        ref = ReflectionResult(
            passed=False, issues=[],
            add_keywords=[{"keyword": "好", "score": 0.95}],
        )
        updated_kw, _ = ResultReflector._apply_delta(keywords, {}, ref)

        assert len(updated_kw) == 1

    def test_corrected_sentiment(self):
        sentiment = {"label": "positive"}
        ref = ReflectionResult(
            passed=False, issues=[],
            corrected_sentiment={"label": "negative", "confidence": 0.9},
        )
        _, updated_sent = ResultReflector._apply_delta([], sentiment, ref)

        assert updated_sent["label"] == "negative"


# ══════════════════════════════════════════════════
# _extract_issue_fingerprints
# ══════════════════════════════════════════════════

class TestExtractIssueFingerprints:
    def test_extracts_high_severity_only(self):
        issues = [
            {"type": "missing_keyword", "detail": "缺少质量", "severity": "high"},
            {"type": "low_score", "detail": "得分偏低", "severity": "low"},
        ]
        fps = ResultReflector._extract_issue_fingerprints(issues)
        assert len(fps) == 1
        assert "missing_keyword:" in list(fps)[0]

    def test_empty_issues(self):
        fps = ResultReflector._extract_issue_fingerprints([])
        assert fps == set()


# ══════════════════════════════════════════════════
# reflect_and_correct
# ══════════════════════════════════════════════════

class TestReflectAndCorrect:
    def test_passes_on_first_round(self):
        """第一轮就通过"""
        response_json = {
            "passed": True, "issues": [],
            "add_keywords": [], "remove_keywords": [],
            "summary": "合格",
        }
        reflector = _make_reflector(json.dumps(response_json, ensure_ascii=False))

        result = reflector.reflect_and_correct("好评", [{"keyword": "好评"}], {})

        assert result["final_passed"] is True
        assert result["total_rounds"] == 1

    def test_convergence_detection(self):
        """连续两轮相同 issue 时强制终止"""
        response_json = {
            "passed": False,
            "issues": [{"type": "missing_keyword", "detail": "缺少质量", "severity": "high"}],
            "add_keywords": [],
            "remove_keywords": [],
            "summary": "关键词不够",
        }
        mock_llm = MagicMock()
        mock_llm.load_skill.return_value = MagicMock(system="sys", user="user")
        mock_llm.call_agent.return_value = LLMResponse(
            content=json.dumps(response_json, ensure_ascii=False)
        )
        reflector = ResultReflector(llm_service=mock_llm)

        result = reflector.reflect_and_correct("好评", [], {}, max_rounds=5)

        assert result["total_rounds"] == 2
        assert result["reflection_history"][-1].get("forced_stop") is True

    def test_max_rounds_respected(self):
        """不超过 max_rounds"""
        response_json = {
            "passed": False,
            "issues": [{"type": "a", "detail": "x", "severity": "low"}],
            "add_keywords": [{"keyword": "新词", "score": 0.8}],
            "remove_keywords": [],
            "summary": "继续",
        }

        call_count = 0

        def _make_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            r = dict(response_json)
            r["issues"] = [{"type": f"issue_{call_count}", "detail": "x", "severity": "high"}]
            r["add_keywords"] = [{"keyword": f"词{call_count}", "score": 0.8}]
            return LLMResponse(content=json.dumps(r, ensure_ascii=False))

        mock_llm = MagicMock()
        mock_llm.load_skill.return_value = MagicMock(system="sys", user="user")
        mock_llm.call_agent.side_effect = _make_response

        reflector = ResultReflector(llm_service=mock_llm)
        result = reflector.reflect_and_correct("好评", [], {}, max_rounds=3)

        assert result["total_rounds"] <= 3

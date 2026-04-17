"""
第三阶段测试 — 文件输入 + 批量分析

测试内容：
1. file_reader 对 txt/csv/json 的解析
2. 空文件、格式错误等边界情况
3. analyze -f 命令的集成
4. 批量表格输出格式
"""

import csv
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from extract_agent.cli.file_reader import (
    read_comments_from_file,
    FileReadError,
)
from extract_agent.cli.formatter import format_batch_table
from extract_agent.cli.main import app

runner = CliRunner()

MOCK_RESULT = {
    "original_text": "测试评论",
    "cleaned_text": "测试评论",
    "keywords": [
        {"keyword": "测试", "reasoning": "测试", "score": 0.9},
    ],
    "sentiment": {"label": "positive", "confidence": 0.9, "reasoning": "正面"},
    "analysis_complete": True,
    "elapsed_ms": 500,
    "steps": 3,
    "mode": "fast",
}


class TestFileReaderTxt:
    """测试 txt 文件读取"""

    def test_read_txt(self, tmp_path):
        f = tmp_path / "comments.txt"
        f.write_text("第一条评论\n第二条评论\n第三条\n", encoding="utf-8")
        result = read_comments_from_file(str(f))
        assert result == ["第一条评论", "第二条评论", "第三条"]

    def test_read_txt_skips_empty_lines(self, tmp_path):
        f = tmp_path / "comments.txt"
        f.write_text("评论1\n\n\n评论2\n  \n评论3\n", encoding="utf-8")
        result = read_comments_from_file(str(f))
        assert result == ["评论1", "评论2", "评论3"]

    def test_read_txt_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        with pytest.raises(FileReadError, match="文件为空"):
            read_comments_from_file(str(f))


class TestFileReaderCsv:
    """测试 csv 文件读取"""

    def test_read_csv(self, tmp_path):
        f = tmp_path / "comments.csv"
        with open(f, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["id", "review_text"])
            writer.writeheader()
            writer.writerow({"id": "1", "review_text": "好评"})
            writer.writerow({"id": "2", "review_text": "差评"})
        result = read_comments_from_file(str(f), column="review_text")
        assert result == ["好评", "差评"]

    def test_read_csv_no_column_param(self, tmp_path):
        f = tmp_path / "comments.csv"
        f.write_text("id,text\n1,好\n", encoding="utf-8")
        with pytest.raises(FileReadError, match="--column"):
            read_comments_from_file(str(f))

    def test_read_csv_wrong_column(self, tmp_path):
        f = tmp_path / "comments.csv"
        f.write_text("id,text\n1,好\n", encoding="utf-8")
        with pytest.raises(FileReadError, match="不存在列"):
            read_comments_from_file(str(f), column="nonexistent")

    def test_read_csv_empty_data(self, tmp_path):
        f = tmp_path / "comments.csv"
        f.write_text("id,text\n1,\n2,  \n", encoding="utf-8")
        with pytest.raises(FileReadError, match="没有有效数据"):
            read_comments_from_file(str(f), column="text")


class TestFileReaderJson:
    """测试 json 文件读取"""

    def test_read_json(self, tmp_path):
        f = tmp_path / "comments.json"
        data = [
            {"text": "好评", "id": 1},
            {"text": "差评", "id": 2},
        ]
        f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        result = read_comments_from_file(str(f), field="text")
        assert result == ["好评", "差评"]

    def test_read_json_no_field_param(self, tmp_path):
        f = tmp_path / "comments.json"
        f.write_text("[{}]", encoding="utf-8")
        with pytest.raises(FileReadError, match="--field"):
            read_comments_from_file(str(f))

    def test_read_json_not_array(self, tmp_path):
        f = tmp_path / "comments.json"
        f.write_text('{"key": "value"}', encoding="utf-8")
        with pytest.raises(FileReadError, match="数组格式"):
            read_comments_from_file(str(f), field="key")

    def test_read_json_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(FileReadError, match="JSON 解析失败"):
            read_comments_from_file(str(f), field="text")

    def test_read_json_empty_data(self, tmp_path):
        f = tmp_path / "comments.json"
        f.write_text('[{"text": ""}, {"text": "  "}]', encoding="utf-8")
        with pytest.raises(FileReadError, match="没有有效数据"):
            read_comments_from_file(str(f), field="text")


class TestFileReaderGeneral:
    """测试通用场景"""

    def test_nonexistent_file(self):
        with pytest.raises(FileReadError, match="文件不存在"):
            read_comments_from_file("/nonexistent/file.txt")

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "data.xml"
        f.write_text("<root/>", encoding="utf-8")
        with pytest.raises(FileReadError, match="不支持的文件格式"):
            read_comments_from_file(str(f))


class TestAnalyzeFromFile:
    """测试 analyze -f 命令集成"""

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_analyze_txt_file(self, mock_cls, tmp_path):
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_cls.return_value = mock_agent

        f = tmp_path / "comments.txt"
        f.write_text("评论一\n评论二\n", encoding="utf-8")

        result = runner.invoke(app, ["analyze", "-f", str(f), "--mode", "fast"])
        assert result.exit_code == 0
        assert "2 条评论" in result.output
        assert "批量分析结果" in result.output

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_analyze_csv_file(self, mock_cls, tmp_path):
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_cls.return_value = mock_agent

        f = tmp_path / "comments.csv"
        with open(f, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["id", "text"])
            writer.writeheader()
            writer.writerow({"id": "1", "text": "好评"})
        result = runner.invoke(
            app, ["analyze", "-f", str(f), "--column", "text", "--mode", "fast"]
        )
        assert result.exit_code == 0

    def test_analyze_no_text_no_file(self):
        result = runner.invoke(app, ["analyze"])
        assert result.exit_code != 0


class TestBatchFormatter:
    """测试批量结果表格"""

    def test_format_batch_table_runs(self):
        results = [MOCK_RESULT, MOCK_RESULT]
        texts = ["评论一", "评论二"]
        format_batch_table(results, texts)

    def test_format_batch_table_empty(self):
        format_batch_table([], [])

"""
第二阶段测试 — 配置文件 + Session

测试内容：
1. config_loader 各优先级逻辑
2. YAML -> AgentConfig 映射
3. session_id 生成格式
4. --full 目录创建和 JSON 文件写入
5. config --init / --show 命令
"""

import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml
from typer.testing import CliRunner

from extract_agent.cli.config_loader import (
    find_config_file,
    load_config,
    init_config_file,
    get_default_yaml,
    _apply_yaml_to_config,
    _load_yaml,
    USER_CONFIG_FILE,
)
from extract_agent.cli.session import CLISession
from extract_agent.cli.main import app
from extract_agent.config import AgentConfig

runner = CliRunner()

MOCK_RESULT = {
    "original_text": "测试评论",
    "cleaned_text": "测试评论",
    "keywords": [
        {"keyword": "测试", "reasoning": "测试用", "score": 0.9},
    ],
    "sentiment": {"label": "neutral", "confidence": 0.8, "reasoning": "中性"},
    "analysis_complete": True,
    "elapsed_ms": 100,
    "steps": 3,
    "mode": "fast",
}


class TestFindConfigFile:
    """测试 find_config_file 优先级"""

    def test_cli_path_takes_priority(self, tmp_path):
        config_file = tmp_path / "my_config.yaml"
        config_file.write_text("agent_llm:\n  base_url: test\n", encoding="utf-8")
        result = find_config_file(str(config_file))
        assert result == config_file

    def test_cli_path_nonexistent_returns_none(self):
        result = find_config_file("/nonexistent/path/config.yaml")
        assert result is None

    def test_local_config_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        local_cfg = tmp_path / ".extract-agent.yaml"
        local_cfg.write_text("agent_llm:\n  base_url: local\n", encoding="utf-8")
        result = find_config_file()
        assert result == local_cfg

    def test_no_config_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch(
            "extract_agent.cli.config_loader.USER_CONFIG_FILE",
            tmp_path / "nonexistent" / "config.yaml",
        ):
            result = find_config_file()
        assert result is None


class TestLoadConfig:
    """测试 YAML -> AgentConfig 映射"""

    def test_load_with_yaml_file(self, tmp_path):
        yaml_content = {
            "agent_llm": {
                "base_url": "http://test:8001/v1",
                "temperature": 0.5,
                "max_tokens": 1000,
            },
            "agent": {
                "max_steps": 20,
            },
            "reflection": {
                "enabled": False,
            },
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(yaml_content, allow_unicode=True), encoding="utf-8"
        )
        config = load_config(str(config_file))

        assert config.AGENT_LLM_TEMPERATURE == 0.5
        assert config.AGENT_MAX_STEPS == 20
        assert config.ENABLE_REFLECTION is False

    def test_env_var_overrides_yaml(self, tmp_path, monkeypatch):
        """环境变量应覆盖 YAML 配置"""
        monkeypatch.setenv("AGENT_LLM_BASE_URL", "http://env-override:9999/v1")

        yaml_content = {"agent_llm": {"base_url": "http://yaml-value:8001/v1"}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(yaml_content, allow_unicode=True), encoding="utf-8"
        )

        config = load_config(str(config_file))
        assert config.AGENT_LLM_BASE_URL == "http://env-override:9999/v1"

    def test_load_without_config_file(self, tmp_path, monkeypatch):
        """无配置文件时使用默认值"""
        monkeypatch.chdir(tmp_path)
        with patch(
            "extract_agent.cli.config_loader.USER_CONFIG_FILE",
            tmp_path / "nonexistent" / "config.yaml",
        ):
            config = load_config()
        assert config._config_source == "默认配置"

    def test_cli_section_parsed(self, tmp_path):
        yaml_content = {
            "cli": {
                "output_dir": "/custom/output",
                "default_mode": "fast",
            },
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(yaml_content, allow_unicode=True), encoding="utf-8"
        )
        config = load_config(str(config_file))
        assert config._cli_output_dir == "/custom/output"
        assert config._cli_default_mode == "fast"


class TestApplyYamlToConfig:
    """测试 _apply_yaml_to_config 类型转换"""

    def test_int_conversion(self):
        config = AgentConfig()
        _apply_yaml_to_config(config, {"agent": {"max_steps": "25"}})
        assert config.AGENT_MAX_STEPS == 25

    def test_float_conversion(self):
        config = AgentConfig()
        _apply_yaml_to_config(config, {"agent_llm": {"temperature": "0.7"}})
        assert config.AGENT_LLM_TEMPERATURE == 0.7

    def test_bool_conversion(self):
        config = AgentConfig()
        _apply_yaml_to_config(config, {"reflection": {"enabled": "false"}})
        assert config.ENABLE_REFLECTION is False


class TestInitConfigFile:
    """测试 config --init"""

    def test_init_creates_file(self, tmp_path):
        fake_config_dir = tmp_path / ".extract-agent"
        fake_config_file = fake_config_dir / "config.yaml"

        with patch("extract_agent.cli.config_loader.USER_CONFIG_DIR", fake_config_dir), \
             patch("extract_agent.cli.config_loader.USER_CONFIG_FILE", fake_config_file):
            path = init_config_file()

        assert path.is_file()
        content = path.read_text(encoding="utf-8")
        assert "agent_llm" in content
        assert "tool_llm" in content

    def test_default_yaml_is_valid(self):
        """默认 YAML 模板应该能被正确解析"""
        template = get_default_yaml()
        data = yaml.safe_load(template)
        assert isinstance(data, dict)
        assert "agent_llm" in data
        assert "tool_llm" in data
        assert "cli" in data


class TestCLISession:
    """测试 CLISession"""

    def test_session_id_format(self):
        config = AgentConfig()
        with patch("extract_agent.cli.session.ReviewAnalysisAgent"):
            session = CLISession(config=config, mode="fast")
        assert re.match(r"^[0-9a-f]{8}$", session.session_id)

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_analyze_increments_counter(self, mock_cls):
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_cls.return_value = mock_agent

        config = AgentConfig()
        session = CLISession(config=config, mode="fast")

        session.analyze("评论1")
        assert session.result_counter == 1
        session.analyze("评论2")
        assert session.result_counter == 2

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_full_output_creates_directory(self, mock_cls, tmp_path):
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_cls.return_value = mock_agent

        config = AgentConfig()
        session = CLISession(
            config=config,
            mode="fast",
            full_output=True,
            output_root=str(tmp_path / "output"),
        )
        assert session.output_dir is not None
        assert session.output_dir.is_dir()

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_full_output_saves_json(self, mock_cls, tmp_path):
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_cls.return_value = mock_agent

        config = AgentConfig()
        session = CLISession(
            config=config,
            mode="fast",
            full_output=True,
            output_root=str(tmp_path / "output"),
        )
        session.analyze("测试")

        result_file = session.output_dir / "result_001.json"
        assert result_file.is_file()
        data = json.loads(result_file.read_text(encoding="utf-8"))
        assert data["original_text"] == "测试评论"

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_close_writes_meta(self, mock_cls, tmp_path):
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_cls.return_value = mock_agent

        config = AgentConfig()
        session = CLISession(
            config=config,
            mode="fast",
            full_output=True,
            output_root=str(tmp_path / "output"),
        )
        session.analyze("测试")
        session.close()

        meta_file = session.output_dir / "session_meta.json"
        assert meta_file.is_file()
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        assert meta["session_id"] == session.session_id
        assert meta["total_analyzed"] == 1

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_session_history(self, mock_cls):
        mock_agent = MagicMock()
        mock_agent.run.return_value = MOCK_RESULT
        mock_cls.return_value = mock_agent

        config = AgentConfig()
        session = CLISession(config=config, mode="fast")
        session.analyze("第一条")
        session.analyze("第二条")

        assert len(session.history) == 2
        summary = session.get_history_summary()
        assert len(summary) == 2
        assert summary[0]["index"] == 1

    @patch("extract_agent.cli.session.ReviewAnalysisAgent")
    def test_session_info(self, mock_cls):
        mock_agent = MagicMock()
        mock_cls.return_value = mock_agent

        config = AgentConfig()
        session = CLISession(config=config, mode="fast")
        info = session.get_session_info()
        assert info["session_id"] == session.session_id
        assert info["mode"] == "fast"
        assert info["total_analyzed"] == 0


class TestConfigCommand:
    """测试 config 命令"""

    def test_config_show(self):
        result = runner.invoke(app, ["config", "--show"])
        assert result.exit_code == 0
        assert "配置来源" in result.output

    def test_config_no_args(self):
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "--init" in result.output or "--show" in result.output

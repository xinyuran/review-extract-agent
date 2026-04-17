"""
YAML 配置文件加载 — 查找、解析并映射到 AgentConfig

优先级：环境变量 > YAML 配置文件 > AgentConfig 代码默认值
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..config import AgentConfig

CONFIG_FILE_NAME = "config.yaml"
LOCAL_CONFIG_FILE = ".extract-agent.yaml"
USER_CONFIG_DIR = Path.home() / ".extract-agent"
USER_CONFIG_FILE = USER_CONFIG_DIR / CONFIG_FILE_NAME

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = str(_PACKAGE_ROOT / "extract_agent_output")

YAML_TO_CONFIG_MAP: Dict[str, Dict[str, str]] = {
    "agent_llm": {
        "base_url": "AGENT_LLM_BASE_URL",
        "api_key": "AGENT_LLM_API_KEY",
        "model": "AGENT_LLM_MODEL",
        "temperature": "AGENT_LLM_TEMPERATURE",
        "max_tokens": "AGENT_LLM_MAX_TOKENS",
    },
    "tool_llm": {
        "base_url": "TOOL_LLM_BASE_URL",
        "api_key": "TOOL_LLM_API_KEY",
        "model": "TOOL_LLM_MODEL",
        "temperature": "TOOL_LLM_TEMPERATURE",
        "max_tokens": "TOOL_LLM_MAX_TOKENS",
        "seed": "TOOL_LLM_SEED",
        "frequency_penalty": "TOOL_LLM_FREQUENCY_PENALTY",
        "repetition_penalty": "TOOL_LLM_REPETITION_PENALTY",
    },
    "agent": {
        "tool_calling_mode": "AGENT_TOOL_CALLING_MODE",
        "max_steps": "AGENT_MAX_STEPS",
        "timeout": "AGENT_TIMEOUT",
        "tool_timeout": "TOOL_TIMEOUT",
    },
    "reflection": {
        "enabled": "ENABLE_REFLECTION",
        "max_rounds": "REFLECTION_MAX_ROUNDS",
        "score_threshold": "REFLECTION_SCORE_THRESHOLD",
    },
}

ENV_BACKED_KEYS = {
    "AGENT_LLM_BASE_URL", "AGENT_LLM_API_KEY", "AGENT_LLM_MODEL",
    "AGENT_LLM_TEMPERATURE", "AGENT_LLM_MAX_TOKENS",
    "TOOL_LLM_BASE_URL", "TOOL_LLM_API_KEY", "TOOL_LLM_MODEL",
    "TOOL_LLM_TEMPERATURE", "TOOL_LLM_MAX_TOKENS",
    "AGENT_TOOL_CALLING_MODE", "AGENT_MAX_STEPS", "AGENT_TIMEOUT", "TOOL_TIMEOUT",
    "ENABLE_REFLECTION", "REFLECTION_MAX_ROUNDS", "REFLECTION_SCORE_THRESHOLD",
    "DEBUG",
}

DEFAULT_YAML_TEMPLATE = """\
# Extract Agent 配置文件
# 优先级：环境变量 > 本文件 > 代码默认值

# Agent LLM 配置（负责规划和推理）
agent_llm:
  base_url: "http://192.168.12.42:8001/v1"
  api_key: "dummy"
  model: "Qwen2.5-7B-Instruct"
  temperature: 0
  max_tokens: 800

# Tool LLM 配置（关键词提取、情感分析等工具使用）
tool_llm:
  base_url: "http://192.168.12.42:8002/v1"
  api_key: "dummy"
  model: "finetuned-model"
  temperature: 0
  max_tokens: 4096

# Agent 控制参数
agent:
  tool_calling_mode: "native"   # native / prompt / auto
  max_steps: 10
  timeout: 120
  tool_timeout: 30

# 反思器配置
reflection:
  enabled: true
  max_rounds: 5
  score_threshold: 0.7

# CLI 配置
cli:
  output_dir: ""  # 留空则保存到 extract_agent/extract_agent_output/
  default_mode: "agent"

# Redis（仅 serve 命令使用）
redis:
  url: "redis://127.0.0.1:6379/0"
"""


def find_config_file(cli_path: Optional[str] = None) -> Optional[Path]:
    """按优先级查找配置文件，返回 Path 或 None"""
    if cli_path:
        p = Path(cli_path)
        if p.is_file():
            return p
        return None

    local = Path.cwd() / LOCAL_CONFIG_FILE
    if local.is_file():
        return local

    if USER_CONFIG_FILE.is_file():
        return USER_CONFIG_FILE

    return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _apply_yaml_to_config(
    config: AgentConfig, yaml_data: Dict[str, Any]
) -> None:
    """将 YAML 中的值映射到 AgentConfig 实例属性"""
    for section, mapping in YAML_TO_CONFIG_MAP.items():
        section_data = yaml_data.get(section)
        if not isinstance(section_data, dict):
            continue
        for yaml_key, config_attr in mapping.items():
            if yaml_key not in section_data:
                continue
            value = section_data[yaml_key]
            if config_attr in ENV_BACKED_KEYS and os.getenv(config_attr) is not None:
                continue
            _set_config_attr(config, config_attr, value)


def _set_config_attr(config: AgentConfig, attr: str, value: Any) -> None:
    """按照 AgentConfig 中属性的类型做类型转换后赋值"""
    current = getattr(config, attr, None)
    if current is not None:
        try:
            if isinstance(current, bool):
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
                else:
                    value = bool(value)
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            else:
                value = str(value)
        except (ValueError, TypeError):
            return
    setattr(config, attr, value)


def _apply_env_overrides(config: AgentConfig) -> None:
    """
    运行时重读环境变量，覆盖实例属性。

    AgentConfig 的类属性在模块导入时就已求值（os.getenv 只执行一次），
    后续通过 monkeypatch / os.environ 修改环境变量不会反映到新实例上。
    此函数在 load_config 末尾显式重读，确保「环境变量 > YAML > 默认值」。
    """
    for _section, mapping in YAML_TO_CONFIG_MAP.items():
        for _yaml_key, config_attr in mapping.items():
            env_val = os.getenv(config_attr)
            if env_val is not None:
                _set_config_attr(config, config_attr, env_val)

    debug_val = os.getenv("DEBUG")
    if debug_val is not None:
        config.DEBUG = debug_val.lower() in ("true", "1", "yes")


def load_config(cli_path: Optional[str] = None) -> AgentConfig:
    """
    加载配置并返回 AgentConfig 实例。

    优先级：环境变量 > YAML > 默认值
    """
    config = AgentConfig()

    config_file = find_config_file(cli_path)
    if config_file:
        yaml_data = _load_yaml(config_file)
        _apply_yaml_to_config(config, yaml_data)
        config._config_source = str(config_file)
    else:
        config._config_source = "默认配置"

    _apply_env_overrides(config)

    cli_section = {}
    if config_file:
        yaml_data = _load_yaml(config_file)
        cli_section = yaml_data.get("cli", {})
    config._cli_output_dir = cli_section.get("output_dir", DEFAULT_OUTPUT_DIR)
    config._cli_default_mode = cli_section.get("default_mode", "agent")

    return config


def get_default_yaml() -> str:
    """返回默认的 YAML 配置模板"""
    return DEFAULT_YAML_TEMPLATE


def init_config_file() -> Path:
    """在用户目录创建默认配置文件，返回文件路径"""
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    USER_CONFIG_FILE.write_text(DEFAULT_YAML_TEMPLATE, encoding="utf-8")
    return USER_CONFIG_FILE

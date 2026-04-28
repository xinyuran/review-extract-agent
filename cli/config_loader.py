"""
配置文件加载 — 支持 .env / YAML，映射到 AgentConfig

优先级（高 → 低）：
  1. .env 文件中的变量（注入到 os.environ）
  2. 系统环境变量（已存在的不会被 .env 覆盖）
  3. YAML 配置文件
  4. AgentConfig 代码默认值
"""

import os
import re
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

# .env 文件查找路径（按优先级）
_ENV_FILE_LOCATIONS = [
    _PACKAGE_ROOT / ".env",                  # extract_agent/.env
    _PACKAGE_ROOT.parent / ".env",           # Structured-LLM-Extraction-Framework/.env
    Path.cwd() / ".env",                     # 当前工作目录/.env
]

_ENV_LINE_PATTERN = re.compile(
    r"""^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$"""
)


def _load_env_file(env_path: Path) -> Dict[str, str]:
    """
    解析 .env 文件，返回 {KEY: VALUE} 字典。
    支持：# 注释、空行、可选的 export 前缀、单/双引号值。
    不覆盖已存在的系统环境变量。
    """
    result: Dict[str, str] = {}
    if not env_path.is_file():
        return result

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = _ENV_LINE_PATTERN.match(line)
            if not match:
                continue
            key = match.group(1)
            value = match.group(2).strip()
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            inline_comment = value.find(" #")
            if inline_comment > 0 and not (value.startswith('"') or value.startswith("'")):
                value = value[:inline_comment].rstrip()
            result[key] = value

    return result


def _apply_env_file() -> Optional[Path]:
    """
    查找并加载 .env 文件，将其中的变量注入 os.environ（不覆盖已有值）。
    返回加载的文件路径，未找到则返回 None。
    """
    for env_path in _ENV_FILE_LOCATIONS:
        try:
            resolved = env_path.resolve()
        except OSError:
            continue
        if resolved.is_file():
            env_vars = _load_env_file(resolved)
            for key, value in env_vars.items():
                if key not in os.environ:
                    os.environ[key] = value
            return resolved
    return None

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
        "backend_mode": "BACKEND_MODE",
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
    "BACKEND_MODE",
    "AGENT_TOOL_CALLING_MODE", "AGENT_MAX_STEPS", "AGENT_TIMEOUT", "TOOL_TIMEOUT",
    "ENABLE_REFLECTION", "REFLECTION_MAX_ROUNDS", "REFLECTION_SCORE_THRESHOLD",
    "DEBUG",
}

DEFAULT_YAML_TEMPLATE = """\
# Extract Agent 配置文件
# 优先级：环境变量 > 本文件 > 代码默认值

# ============================================================
# 后端模式（三选一）：
#   auto        - 根据 base_url 自动判断（默认）
#   cloud_api   - 使用云服务商 API（OpenAI / DeepSeek / 通义千问等）
#   local_model - 使用本地部署模型（vLLM / Ollama 等）
#   offline     - 不使用任何 LLM，仅 jieba 提取关键词，无情感分析
# ============================================================

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

# ---- 云服务 API 配置示例 ----
# 使用 OpenAI:
#   agent_llm:
#     base_url: "https://api.openai.com/v1"
#     api_key: "sk-..."
#     model: "gpt-4o-mini"
#   tool_llm:
#     base_url: "https://api.openai.com/v1"
#     api_key: "sk-..."
#     model: "gpt-4o-mini"
#
# 使用 DeepSeek:
#   agent_llm:
#     base_url: "https://api.deepseek.com/v1"
#     api_key: "sk-..."
#     model: "deepseek-chat"

# Agent 控制参数
agent:
  backend_mode: "auto"          # auto / cloud_api / local_model / offline
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

    enable_traj = os.getenv("ENABLE_TRAJECTORY")
    if enable_traj is not None:
        config.ENABLE_TRAJECTORY = enable_traj.lower() in ("true", "1", "yes")

    traj_dir = os.getenv("TRAJECTORY_OUTPUT_DIR")
    if traj_dir is not None:
        config.TRAJECTORY_OUTPUT_DIR = _resolve_output_path(traj_dir)

    traj_think = os.getenv("TRAJECTORY_INCLUDE_THINKING")
    if traj_think is not None:
        config.TRAJECTORY_INCLUDE_THINKING = traj_think.lower() in ("true", "1", "yes")

    enable_know = os.getenv("ENABLE_KNOWLEDGE")
    if enable_know is not None:
        config.ENABLE_KNOWLEDGE = enable_know.lower() in ("true", "1", "yes")

    know_dir = os.getenv("KNOWLEDGE_STORE_DIR")
    if know_dir is not None:
        config.KNOWLEDGE_STORE_DIR = _resolve_output_path(know_dir)

    skills_dir = os.getenv("SKILLS_DIR")
    if skills_dir is not None:
        config.SKILLS_DIR = skills_dir


def _resolve_output_path(path_str: str) -> str:
    """
    将输出路径解析为绝对路径。

    如果是相对路径，则相对于 extract_agent 包目录解析，
    确保无论从哪个工作目录运行，输出都在 extract_agent/ 下。
    """
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(_PACKAGE_ROOT / path_str)


def load_config(cli_path: Optional[str] = None) -> AgentConfig:
    """
    加载配置并返回 AgentConfig 实例。

    优先级（高 → 低）：
      1. .env 文件 → 注入 os.environ（不覆盖已有环境变量）
      2. 系统环境变量
      3. YAML 配置文件
      4. AgentConfig 代码默认值
    """
    env_file = _apply_env_file()

    config = AgentConfig()

    config_file = find_config_file(cli_path)
    if config_file:
        yaml_data = _load_yaml(config_file)
        _apply_yaml_to_config(config, yaml_data)
        config._config_source = str(config_file)
    else:
        config._config_source = "默认配置"

    _apply_env_overrides(config)

    sources = []
    if env_file:
        sources.append(f".env ({env_file})")
    if config_file:
        sources.append(f"YAML ({config_file})")
    if sources:
        config._config_source = " + ".join(sources)
    elif config._config_source == "默认配置":
        pass

    config._env_file = str(env_file) if env_file else None

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


def get_default_env() -> str:
    """返回默认的 .env 模板内容"""
    env_example = _PACKAGE_ROOT / ".env.example"
    if env_example.is_file():
        return env_example.read_text(encoding="utf-8")
    return _DEFAULT_ENV_CONTENT


def init_config_file() -> Path:
    """在用户目录创建默认配置文件，返回文件路径"""
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    USER_CONFIG_FILE.write_text(DEFAULT_YAML_TEMPLATE, encoding="utf-8")
    return USER_CONFIG_FILE


_DEFAULT_ENV_CONTENT = """\
# Extract Agent 环境配置
# 详细说明参见 .env.example

BACKEND_MODE=auto
AGENT_LLM_BASE_URL=http://192.168.12.42:8001/v1
AGENT_LLM_API_KEY=dummy
AGENT_LLM_MODEL=Qwen2.5-7B-Instruct
AGENT_LLM_MAX_TOKENS=4096
TOOL_LLM_BASE_URL=http://192.168.12.42:8002/v1
TOOL_LLM_API_KEY=dummy
TOOL_LLM_MODEL=finetuned-model
TOOL_LLM_MAX_TOKENS=4096
AGENT_TOOL_CALLING_MODE=native
ENABLE_REFLECTION=true
DEBUG=false
"""

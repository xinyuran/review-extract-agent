"""
AI Agent 全局配置

支持三种后端模式：
- cloud_api：对接云服务商 API（OpenAI / Anthropic / DeepSeek 等 OpenAI 兼容接口）
- local_model：本地部署模型（vLLM / Ollama 等）
- offline：无 LLM 可用，仅使用 jieba 提取关键词，不做情感分析

支持双 LLM endpoint：
- Agent LLM：负责 Agent 的规划、推理、工具选择（需要 Function Calling 能力）
- Tool LLM：工具内部使用的 LLM（关键词提取、情感分析等）
两者默认指向同一服务，可通过环境变量分别配置。

配置分组为 dataclass 子类（LLMConfig / PreprocessConfig / PostprocessConfig / ReflectionConfig
/ TrajectoryConfig / KnowledgeConfig），保留对旧式扁平属性名的完全兼容。

环境变量通过 python-dotenv 自动加载 .env 文件（若存在），在 dataclass __post_init__
中读取，确保实例化时获取最新值而非 import-time 绑定。
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_CLOUD_API_HOSTS = (
    "api.openai.com",
    "api.anthropic.com",
    "api.deepseek.com",
    "dashscope.aliyuncs.com",
    "api.moonshot.cn",
    "api.siliconflow.cn",
    "api.zhipuai.cn",
    "open.bigmodel.cn",
)


def _detect_backend(base_url: str) -> str:
    """
    根据 base_url 自动判断后端类型。

    Returns:
        "cloud_api" | "local_model" | "offline"
    """
    if not base_url or base_url.strip().lower() in ("", "none", "offline"):
        return "offline"

    url_lower = base_url.lower()
    for host in _CLOUD_API_HOSTS:
        if host in url_lower:
            return "cloud_api"

    if "localhost" in url_lower or "127.0.0.1" in url_lower or "192.168." in url_lower or "10." in url_lower:
        return "local_model"

    if url_lower.startswith("http://") or url_lower.startswith("https://"):
        return "cloud_api"

    return "local_model"


# ---------------------------------------------------------------------------
# 子配置 dataclass
# ---------------------------------------------------------------------------

def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass
class LLMConfig:
    """Agent LLM + Tool LLM 连接和推理参数"""
    agent_base_url: str = field(default_factory=lambda: _env("AGENT_LLM_BASE_URL", "http://localhost:8001/v1"))
    agent_api_key: str = field(default_factory=lambda: _env("AGENT_LLM_API_KEY", "dummy"))
    agent_model: str = field(default_factory=lambda: _env("AGENT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    agent_temperature: float = field(default_factory=lambda: float(_env("AGENT_LLM_TEMPERATURE", "0")))
    agent_max_tokens: int = field(default_factory=lambda: int(_env("AGENT_LLM_MAX_TOKENS", "4096")))

    tool_base_url: str = field(default_factory=lambda: _env("TOOL_LLM_BASE_URL", _env("AGENT_LLM_BASE_URL", "http://localhost:8002/v1")))
    tool_api_key: str = field(default_factory=lambda: _env("TOOL_LLM_API_KEY", _env("AGENT_LLM_API_KEY", "dummy")))
    tool_model: str = field(default_factory=lambda: _env("TOOL_LLM_MODEL", _env("AGENT_LLM_MODEL", "finetuned-keyword-extract-model")))
    tool_temperature: float = field(default_factory=lambda: float(_env("TOOL_LLM_TEMPERATURE", "0")))
    tool_max_tokens: int = field(default_factory=lambda: int(_env("TOOL_LLM_MAX_TOKENS", "4096")))
    tool_seed: int = field(default_factory=lambda: int(_env("TOOL_LLM_SEED", "42")))
    tool_frequency_penalty: float = field(default_factory=lambda: float(_env("TOOL_LLM_FREQUENCY_PENALTY", "0.3")))
    tool_repetition_penalty: float = field(default_factory=lambda: float(_env("TOOL_LLM_REPETITION_PENALTY", "1.1")))
    tool_response_format: Dict[str, str] = field(default_factory=lambda: {"type": "json_object"})


@dataclass
class PreprocessConfig:
    """评论文本预处理开关"""
    enable: bool = True
    remove_dates: bool = False
    keep_chinese_only: bool = False
    keep_numbers: bool = True
    keep_chinese_punctuation: bool = True
    remove_english: bool = False
    deduplicate_punctuation: bool = True
    remove_html_entities: bool = True
    normalize_whitespace: bool = True
    remove_control_chars: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phones: bool = True
    normalize_numbers: bool = True
    remove_emojis: bool = True
    remove_garbled: bool = True
    remove_special_symbols: bool = True


@dataclass
class PostprocessConfig:
    """关键词后处理参数"""
    deduplicate: bool = True
    sort_by_importance: bool = True
    filter_low_score: bool = False
    score_threshold: float = 0.5
    top_n: bool = True
    n: int = 8
    remove_english: bool = False
    return_full_info: bool = True

    filter_stopwords: bool = True
    stopwords_exact_match: bool = True
    stopwords_contain_match: bool = False
    stopwords_file: str = "stopwords.txt"

    filter_time_keywords: bool = True
    filter_date_keywords: bool = True
    filter_long_keywords: bool = True
    max_keyword_length: int = 6
    backfill_topn: bool = True
    filter_not_in_original: bool = True
    max_span_ratio: int = 2


@dataclass
class ReflectionConfig:
    """反思器参数"""
    enable: bool = field(default_factory=lambda: _env("ENABLE_REFLECTION", "True").lower() == "true")
    max_rounds: int = field(default_factory=lambda: int(_env("REFLECTION_MAX_ROUNDS", "5")))
    score_threshold: float = field(default_factory=lambda: float(_env("REFLECTION_SCORE_THRESHOLD", "0.7")))
    min_keywords_short: int = 2
    min_keywords_medium: int = 5
    min_keywords_long: int = 8
    min_keywords_xlong: int = 10


@dataclass
class TrajectoryConfig:
    """轨迹采集参数"""
    enable: bool = field(default_factory=lambda: _env("ENABLE_TRAJECTORY", "False").lower() == "true")
    output_dir: str = field(default_factory=lambda: _env("TRAJECTORY_OUTPUT_DIR", "extract_agent_output/trajectory"))
    include_thinking: bool = field(default_factory=lambda: _env("TRAJECTORY_INCLUDE_THINKING", "True").lower() == "true")


@dataclass
class KnowledgeConfig:
    """知识积累参数"""
    enable: bool = field(default_factory=lambda: _env("ENABLE_KNOWLEDGE", "False").lower() == "true")
    store_dir: str = field(default_factory=lambda: _env("KNOWLEDGE_STORE_DIR", "extract_agent_output/knowledge_store"))


# ---------------------------------------------------------------------------
# 旧属性名 -> (子配置名, 新属性名) 映射
# ---------------------------------------------------------------------------

_ATTR_MAP: Dict[str, tuple] = {
    # LLM
    "AGENT_LLM_BASE_URL": ("llm", "agent_base_url"),
    "AGENT_LLM_API_KEY": ("llm", "agent_api_key"),
    "AGENT_LLM_MODEL": ("llm", "agent_model"),
    "AGENT_LLM_TEMPERATURE": ("llm", "agent_temperature"),
    "AGENT_LLM_MAX_TOKENS": ("llm", "agent_max_tokens"),
    "TOOL_LLM_BASE_URL": ("llm", "tool_base_url"),
    "TOOL_LLM_API_KEY": ("llm", "tool_api_key"),
    "TOOL_LLM_MODEL": ("llm", "tool_model"),
    "TOOL_LLM_TEMPERATURE": ("llm", "tool_temperature"),
    "TOOL_LLM_MAX_TOKENS": ("llm", "tool_max_tokens"),
    "TOOL_LLM_SEED": ("llm", "tool_seed"),
    "TOOL_LLM_FREQUENCY_PENALTY": ("llm", "tool_frequency_penalty"),
    "TOOL_LLM_REPETITION_PENALTY": ("llm", "tool_repetition_penalty"),
    "TOOL_LLM_RESPONSE_FORMAT": ("llm", "tool_response_format"),
    # Preprocess
    "ENABLE_PREPROCESS": ("preprocess", "enable"),
    "REMOVE_DATES": ("preprocess", "remove_dates"),
    "KEEP_CHINESE_ONLY": ("preprocess", "keep_chinese_only"),
    "KEEP_NUMBERS": ("preprocess", "keep_numbers"),
    "KEEP_CHINESE_PUNCTUATION": ("preprocess", "keep_chinese_punctuation"),
    "REMOVE_ENGLISH": ("preprocess", "remove_english"),
    "DEDUPLICATE_PUNCTUATION": ("preprocess", "deduplicate_punctuation"),
    "REMOVE_HTML_ENTITIES": ("preprocess", "remove_html_entities"),
    "NORMALIZE_WHITESPACE": ("preprocess", "normalize_whitespace"),
    "REMOVE_CONTROL_CHARS": ("preprocess", "remove_control_chars"),
    "REMOVE_URLS": ("preprocess", "remove_urls"),
    "REMOVE_EMAILS": ("preprocess", "remove_emails"),
    "REMOVE_PHONES": ("preprocess", "remove_phones"),
    "NORMALIZE_NUMBERS_FLAG": ("preprocess", "normalize_numbers"),
    "REMOVE_EMOJIS": ("preprocess", "remove_emojis"),
    "REMOVE_GARBLED": ("preprocess", "remove_garbled"),
    "REMOVE_SPECIAL_SYMBOLS": ("preprocess", "remove_special_symbols"),
    # Postprocess
    "DEDUPLICATE": ("postprocess", "deduplicate"),
    "SORT_BY_IMPORTANCE": ("postprocess", "sort_by_importance"),
    "FILTER_LOW_SCORE": ("postprocess", "filter_low_score"),
    "SCORE_THRESHOLD": ("postprocess", "score_threshold"),
    "TOP_N": ("postprocess", "top_n"),
    "N": ("postprocess", "n"),
    "REMOVE_ENGLISH_IN_POSTPROCESS": ("postprocess", "remove_english"),
    "RETURN_FULL_INFO": ("postprocess", "return_full_info"),
    "FILTER_STOPWORDS": ("postprocess", "filter_stopwords"),
    "STOPWORDS_EXACT_MATCH": ("postprocess", "stopwords_exact_match"),
    "STOPWORDS_CONTAIN_MATCH": ("postprocess", "stopwords_contain_match"),
    "STOPWORDS_FILE": ("postprocess", "stopwords_file"),
    "FILTER_TIME_KEYWORDS": ("postprocess", "filter_time_keywords"),
    "FILTER_DATE_KEYWORDS": ("postprocess", "filter_date_keywords"),
    "FILTER_LONG_KEYWORDS": ("postprocess", "filter_long_keywords"),
    "MAX_KEYWORD_LENGTH": ("postprocess", "max_keyword_length"),
    "BACKFILL_TOPN": ("postprocess", "backfill_topn"),
    "FILTER_KEYWORDS_NOT_IN_ORIGINAL": ("postprocess", "filter_not_in_original"),
    "KEYWORD_MAX_SPAN_RATIO": ("postprocess", "max_span_ratio"),
    # Reflection
    "ENABLE_REFLECTION": ("reflection", "enable"),
    "REFLECTION_MAX_ROUNDS": ("reflection", "max_rounds"),
    "REFLECTION_SCORE_THRESHOLD": ("reflection", "score_threshold"),
    "REFLECTION_MIN_KEYWORDS_SHORT": ("reflection", "min_keywords_short"),
    "REFLECTION_MIN_KEYWORDS_MEDIUM": ("reflection", "min_keywords_medium"),
    "REFLECTION_MIN_KEYWORDS_LONG": ("reflection", "min_keywords_long"),
    "REFLECTION_MIN_KEYWORDS_XLONG": ("reflection", "min_keywords_xlong"),
    # Trajectory
    "ENABLE_TRAJECTORY": ("trajectory", "enable"),
    "TRAJECTORY_OUTPUT_DIR": ("trajectory", "output_dir"),
    "TRAJECTORY_INCLUDE_THINKING": ("trajectory", "include_thinking"),
    # Knowledge
    "ENABLE_KNOWLEDGE": ("knowledge", "enable"),
    "KNOWLEDGE_STORE_DIR": ("knowledge", "store_dir"),
}


class AgentConfig:
    """
    全局配置入口。

    内部由分组 dataclass 组成（llm / preprocess / postprocess / reflection 等），
    同时通过 __getattr__ / __setattr__ 保持对旧式扁平属性名（如 AGENT_LLM_BASE_URL）
    的完全兼容。
    """

    def __init__(self) -> None:
        object.__setattr__(self, "llm", LLMConfig())
        object.__setattr__(self, "preprocess", PreprocessConfig())
        object.__setattr__(self, "postprocess", PostprocessConfig())
        object.__setattr__(self, "reflection", ReflectionConfig())
        object.__setattr__(self, "trajectory", TrajectoryConfig())
        object.__setattr__(self, "knowledge", KnowledgeConfig())

        object.__setattr__(self, "BACKEND_MODE",
                           _env("BACKEND_MODE", "auto").strip().lower())
        object.__setattr__(self, "AGENT_TOOL_CALLING_MODE",
                           _env("AGENT_TOOL_CALLING_MODE", "native").strip())
        object.__setattr__(self, "AGENT_MAX_STEPS",
                           int(_env("AGENT_MAX_STEPS", "10")))
        object.__setattr__(self, "AGENT_TIMEOUT",
                           int(_env("AGENT_TIMEOUT", "1200")))
        object.__setattr__(self, "TOOL_TIMEOUT",
                           int(_env("TOOL_TIMEOUT", "1200")))
        object.__setattr__(self, "MAX_COMMENT_LENGTH", 512)
        object.__setattr__(self, "SHORT_TEXT_LEN", 10)
        object.__setattr__(self, "MAX_RETRIES", 3)
        object.__setattr__(self, "JIEBA_METHOD", "tfidf")
        object.__setattr__(self, "JIEBA_TOP_K", 8)
        object.__setattr__(self, "SKILLS_DIR", _env("SKILLS_DIR", ""))
        object.__setattr__(self, "DEBUG",
                           _env("DEBUG", "False").lower() == "true")

    # ------ backward-compatible attribute delegation ------

    def __getattr__(self, name: str) -> Any:
        mapping = _ATTR_MAP.get(name)
        if mapping:
            group_name, attr = mapping
            group = object.__getattribute__(self, group_name)
            return getattr(group, attr)
        raise AttributeError(f"AgentConfig has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        mapping = _ATTR_MAP.get(name)
        if mapping:
            group_name, attr = mapping
            group = object.__getattribute__(self, group_name)
            setattr(group, attr, value)
        else:
            object.__setattr__(self, name, value)

    # ------ business methods ------

    def get_backend_mode(self) -> str:
        if self.BACKEND_MODE not in ("auto", ""):
            return self.BACKEND_MODE
        return _detect_backend(self.AGENT_LLM_BASE_URL)

    def get_backend_label(self) -> str:
        mode = self.get_backend_mode()
        if mode == "cloud_api":
            return f"[cloud] 云服务 API ({self.AGENT_LLM_BASE_URL})"
        elif mode == "local_model":
            return f"[local] 本地部署模型 ({self.AGENT_LLM_BASE_URL})"
        else:
            return "[offline] 离线模式 (jieba-tfidf 关键词提取, 无情感分析)"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """从环境变量构造配置（与默认构造等价，显式语义入口）。"""
        return cls()

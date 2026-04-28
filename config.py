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
"""

import os

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


class AgentConfig:

    # ==================== Agent LLM 配置 ====================
    AGENT_LLM_BASE_URL = os.getenv(
        "AGENT_LLM_BASE_URL",
        "http://192.168.12.42:8001/v1"
    )
    AGENT_LLM_API_KEY = os.getenv("AGENT_LLM_API_KEY", "dummy")
    AGENT_LLM_MODEL = os.getenv(
        "AGENT_LLM_MODEL",
        "/data/home/ranxinyu/common_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    )
    AGENT_LLM_TEMPERATURE = float(os.getenv("AGENT_LLM_TEMPERATURE", "0"))
    AGENT_LLM_MAX_TOKENS = int(os.getenv("AGENT_LLM_MAX_TOKENS", "4096"))

    # ==================== Tool LLM 配置 ====================
    TOOL_LLM_BASE_URL = os.getenv(
        "TOOL_LLM_BASE_URL",
        os.getenv("AGENT_LLM_BASE_URL", "http://192.168.12.42:8002/v1")
    )
    TOOL_LLM_API_KEY = os.getenv(
        "TOOL_LLM_API_KEY",
        os.getenv("AGENT_LLM_API_KEY", "dummy")
    )
    TOOL_LLM_MODEL = os.getenv(
        "TOOL_LLM_MODEL",
        os.getenv("AGENT_LLM_MODEL", "/data/home/ranxinyu/project_rxy/llm_keyword_sft/output/grpo/v2-20260105-090134/checkpoint-924-merged")
    )
    TOOL_LLM_TEMPERATURE = float(os.getenv("TOOL_LLM_TEMPERATURE", "0"))
    TOOL_LLM_MAX_TOKENS = int(os.getenv("TOOL_LLM_MAX_TOKENS", "4096"))

    # ==================== Tool LLM 推理参数 ====================
    TOOL_LLM_SEED = int(os.getenv("TOOL_LLM_SEED", "42"))
    TOOL_LLM_FREQUENCY_PENALTY = float(os.getenv("TOOL_LLM_FREQUENCY_PENALTY", "0.3"))
    TOOL_LLM_REPETITION_PENALTY = float(os.getenv("TOOL_LLM_REPETITION_PENALTY", "1.1"))
    TOOL_LLM_RESPONSE_FORMAT = {"type": "json_object"}

    # ==================== 后端模式 ====================
    # 显式指定: cloud_api / local_model / offline / auto(自动检测)
    BACKEND_MODE = os.getenv("BACKEND_MODE", "auto").strip().lower()

    # ==================== Agent 控制参数 ====================
    AGENT_TOOL_CALLING_MODE = os.getenv("AGENT_TOOL_CALLING_MODE", "native").strip()
    AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "10"))
    AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "1200"))
    TOOL_TIMEOUT = int(os.getenv("TOOL_TIMEOUT", "1200"))

    def get_backend_mode(self) -> str:
        """
        返回当前后端模式。

        若 BACKEND_MODE 为 auto，则根据 AGENT_LLM_BASE_URL 自动检测。
        """
        if self.BACKEND_MODE not in ("auto", ""):
            return self.BACKEND_MODE
        return _detect_backend(self.AGENT_LLM_BASE_URL)

    def get_backend_label(self) -> str:
        """返回人类可读的后端描述。"""
        mode = self.get_backend_mode()
        if mode == "cloud_api":
            return f"[cloud] 云服务 API ({self.AGENT_LLM_BASE_URL})"
        elif mode == "local_model":
            return f"[local] 本地部署模型 ({self.AGENT_LLM_BASE_URL})"
        else:
            return "[offline] 离线模式 (jieba-tfidf 关键词提取, 无情感分析)"

    # ==================== 预处理配置 ====================
    ENABLE_PREPROCESS = True
    REMOVE_DATES = False
    KEEP_CHINESE_ONLY = False
    KEEP_NUMBERS = True
    KEEP_CHINESE_PUNCTUATION = True
    REMOVE_ENGLISH = False
    DEDUPLICATE_PUNCTUATION = True
    REMOVE_HTML_ENTITIES = True
    NORMALIZE_WHITESPACE = True
    REMOVE_CONTROL_CHARS = True
    REMOVE_URLS = True
    REMOVE_EMAILS = True
    REMOVE_PHONES = True
    NORMALIZE_NUMBERS_FLAG = True
    REMOVE_EMOJIS = True
    REMOVE_GARBLED = True
    REMOVE_SPECIAL_SYMBOLS = True

    # ==================== 后处理配置 ====================
    DEDUPLICATE = True
    SORT_BY_IMPORTANCE = True
    FILTER_LOW_SCORE = False
    SCORE_THRESHOLD = 0.5
    TOP_N = True
    N = 8
    REMOVE_ENGLISH_IN_POSTPROCESS = False
    RETURN_FULL_INFO = True

    FILTER_STOPWORDS = True
    STOPWORDS_EXACT_MATCH = True
    STOPWORDS_CONTAIN_MATCH = False
    STOPWORDS_FILE = "stopwords.txt"

    FILTER_TIME_KEYWORDS = True
    FILTER_DATE_KEYWORDS = True
    FILTER_LONG_KEYWORDS = True
    MAX_KEYWORD_LENGTH = 6
    BACKFILL_TOPN = True
    FILTER_KEYWORDS_NOT_IN_ORIGINAL = True
    KEYWORD_MAX_SPAN_RATIO = 2

    # ==================== 文本处理配置 ====================
    MAX_COMMENT_LENGTH = 512
    SHORT_TEXT_LEN = 10
    MAX_RETRIES = 3

    # ==================== Jieba 兜底配置 ====================
    JIEBA_METHOD = "tfidf"
    JIEBA_TOP_K = 8

    # ==================== 反思器配置 ====================
    ENABLE_REFLECTION = os.getenv("ENABLE_REFLECTION", "True").lower() == "true"
    REFLECTION_MAX_ROUNDS = int(os.getenv("REFLECTION_MAX_ROUNDS", "5"))
    REFLECTION_SCORE_THRESHOLD = float(os.getenv("REFLECTION_SCORE_THRESHOLD", "0.7"))
    # 各长度区间的最低关键词数量要求
    REFLECTION_MIN_KEYWORDS_SHORT = 2      # 原文 < 20 字
    REFLECTION_MIN_KEYWORDS_MEDIUM = 5     # 20 <= 原文 < 60 字
    REFLECTION_MIN_KEYWORDS_LONG = 8       # 60 <= 原文 < 120 字
    REFLECTION_MIN_KEYWORDS_XLONG = 10     # 原文 >= 120 字

    # ==================== Skill 层配置 ====================
    SKILLS_DIR = os.getenv("SKILLS_DIR", "")

    # ==================== 轨迹采集配置 (Phase 2) ====================
    ENABLE_TRAJECTORY = os.getenv("ENABLE_TRAJECTORY", "False").lower() == "true"
    TRAJECTORY_OUTPUT_DIR = os.getenv("TRAJECTORY_OUTPUT_DIR", "extract_agent_output/trajectory")
    TRAJECTORY_INCLUDE_THINKING = os.getenv("TRAJECTORY_INCLUDE_THINKING", "True").lower() == "true"

    # ==================== 知识积累配置 (Phase 3) ====================
    ENABLE_KNOWLEDGE = os.getenv("ENABLE_KNOWLEDGE", "False").lower() == "true"
    KNOWLEDGE_STORE_DIR = os.getenv("KNOWLEDGE_STORE_DIR", "extract_agent_output/knowledge_store")

    # ==================== 调试配置 ====================
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

"""
AI Agent 全局配置

支持双 LLM endpoint：
- Agent LLM：负责 Agent 的规划、推理、工具选择（需要 Function Calling 能力）
- Tool LLM：工具内部使用的 LLM（关键词提取、情感分析等）
两者默认指向同一服务，可通过环境变量分别配置。
"""

import os


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

    # ==================== Agent 控制参数 ====================
    # 工具调用模式：native（原生 Function Calling，推荐）、prompt（prompt-based）、auto（探测）
    # native 模式下 vLLM 需启动时加 --enable-auto-tool-choice --tool-call-parser hermes
    # Hermes 解析器会保留 <tool_call> 标签前的文本到 message.content（即 Thought）
    AGENT_TOOL_CALLING_MODE = os.getenv("AGENT_TOOL_CALLING_MODE", "native").strip()
    AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "10"))
    AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "1200"))
    TOOL_TIMEOUT = int(os.getenv("TOOL_TIMEOUT", "1200"))

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

    # ==================== 调试配置 ====================
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

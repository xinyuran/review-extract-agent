"""
请求 / 响应数据模型

基于 Pydantic v2 定义所有 API 接口的输入输出 schema，
同时用于 FastAPI 自动文档生成和请求校验。
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ===== 通用枚举 =====

class AnalysisMode(str, Enum):
    AGENT = "agent"
    FAST = "fast"
    AUTO = "auto"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ===== 嵌套模型 =====

class KeywordItem(BaseModel):
    keyword: str = Field(description="提取的关键词")
    reasoning: str = Field(default="", description="提取推理说明")
    score: float = Field(default=0.0, description="重要性评分 0~1")


class SentimentResult(BaseModel):
    label: str = Field(description="情感标签: positive/negative/neutral")
    confidence: float = Field(default=0.0, description="置信度 0~1")
    reasoning: str = Field(default="", description="判断依据")


class ReflectionInfo(BaseModel):
    total_rounds: int = Field(default=0, description="反思轮数")
    final_passed: bool = Field(default=True, description="最终是否通过")
    history: Optional[List[Dict[str, Any]]] = Field(default=None, description="反思历史")
    error: Optional[str] = Field(default=None, description="反思异常信息")


# ===== 单条分析 =====

class AnalyzeSingleRequest(BaseModel):
    text: str = Field(description="待分析的评论文本", min_length=1, max_length=2048)
    mode: AnalysisMode = Field(default=AnalysisMode.FAST, description="分析模式")
    enable_reflection: Optional[bool] = Field(
        default=None, description="是否启用反思（None 则使用全局配置）"
    )


class AnalyzeSingleResponse(BaseModel):
    analysis_complete: bool = Field(description="分析是否完成")
    original_text: str = Field(description="原始评论文本")
    cleaned_text: Optional[str] = Field(default=None, description="清洗后的文本")
    keywords: List[KeywordItem] = Field(default_factory=list, description="关键词列表")
    sentiment: Optional[SentimentResult] = Field(default=None, description="情感分析结果")
    summary: Optional[str] = Field(default=None, description="分析摘要")
    reflection: Optional[ReflectionInfo] = Field(default=None, description="反思信息")
    elapsed_ms: float = Field(description="总耗时 (毫秒)")
    mode: str = Field(description="实际使用的分析模式")
    steps: Optional[int] = Field(default=None, description="Agent 步数")
    error: Optional[str] = Field(default=None, description="错误信息")


# ===== 批量分析 =====

class AnalyzeBatchRequest(BaseModel):
    texts: List[str] = Field(
        description="待分析的评论文本列表",
        min_length=1,
        max_length=100,
    )
    mode: AnalysisMode = Field(default=AnalysisMode.FAST, description="分析模式")
    enable_reflection: Optional[bool] = Field(default=None)


class BatchItemResult(AnalyzeSingleResponse):
    batch_index: int = Field(description="在批次中的索引")


class AnalyzeBatchResponse(BaseModel):
    total: int = Field(description="总评论数")
    completed: int = Field(description="成功完成的数量")
    failed: int = Field(description="失败的数量")
    results: List[BatchItemResult] = Field(description="各条目结果")
    total_elapsed_ms: float = Field(description="批量处理总耗时 (毫秒)")


# ===== 异步任务 =====

class AsyncTaskSubmitRequest(BaseModel):
    texts: List[str] = Field(description="待分析的评论文本列表", min_length=1)
    mode: AnalysisMode = Field(default=AnalysisMode.FAST)
    enable_reflection: Optional[bool] = Field(default=None)


class AsyncTaskSubmitResponse(BaseModel):
    task_id: str = Field(description="异步任务 ID")
    status: TaskStatus = Field(description="任务状态")
    total: int = Field(description="总评论数")
    message: str = Field(default="任务已提交")


class AsyncTaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    total: int = Field(default=0)
    completed: int = Field(default=0)
    failed: int = Field(default=0)
    results: Optional[List[BatchItemResult]] = Field(default=None)
    total_elapsed_ms: Optional[float] = Field(default=None)
    error: Optional[str] = Field(default=None)


# ===== 健康检查 =====

class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    agent_ready: bool = Field(description="Agent 是否就绪")
    redis_connected: bool = Field(description="Redis 是否连接")
    tools_loaded: int = Field(description="已加载工具数量")

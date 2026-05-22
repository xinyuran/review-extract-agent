"""
FastAPI 路由定义

接口清单：
- POST /api/analyze         单条评论分析
- POST /api/analyze/stream  单条评论流式分析 (SSE)
- POST /api/analyze/batch   批量评论分析（同步）
- POST /api/task/submit     异步任务提交
- GET  /api/task/{task_id}  异步任务状态查询
- GET  /health              健康检查
"""

import asyncio
import hashlib
import json as _json
import logging
import queue
import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from ..agent.agent import ReviewAnalysisAgent
from ..config import AgentConfig
from .redis_client import RedisManager
from .schemas import (
    AnalyzeBatchRequest,
    AnalyzeBatchResponse,
    AnalyzeSingleRequest,
    AnalyzeSingleResponse,
    AsyncTaskStatusResponse,
    AsyncTaskSubmitRequest,
    AsyncTaskSubmitResponse,
    BatchItemResult,
    HealthResponse,
    TaskStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_agent: ReviewAnalysisAgent | None = None
_config: AgentConfig | None = None
_redis: RedisManager | None = None


def get_agent() -> ReviewAnalysisAgent:
    global _agent, _config
    if _agent is None:
        _config = AgentConfig()
        _agent = ReviewAnalysisAgent(_config)
    return _agent


def get_config() -> AgentConfig:
    global _config
    if _config is None:
        _config = AgentConfig()
    return _config


def set_redis(redis_manager: RedisManager) -> None:
    global _redis
    _redis = redis_manager


def get_redis() -> RedisManager | None:
    return _redis


def _make_cache_key(text: str, mode: str) -> str:
    return hashlib.md5(f"{mode}:{text}".encode()).hexdigest()


def _normalize_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    """将 Agent 内部返回格式规范化为 API 响应格式"""
    keywords = raw.get("keywords", [])
    if keywords and isinstance(keywords[0], list):
        keywords = [
            {"keyword": item[1], "reasoning": item[0], "score": item[2]}
            for item in keywords
            if isinstance(item, list) and len(item) >= 3
        ]

    sentiment = raw.get("sentiment")
    if sentiment and "label" not in sentiment and "sentiment" in sentiment:
        sentiment["label"] = sentiment.pop("sentiment")

    return {
        "analysis_complete": raw.get("analysis_complete", False),
        "original_text": raw.get("original_text", ""),
        "cleaned_text": raw.get("cleaned_text"),
        "keywords": keywords,
        "sentiment": sentiment,
        "summary": raw.get("summary"),
        "reflection": raw.get("reflection"),
        "elapsed_ms": raw.get("elapsed_ms", 0),
        "mode": raw.get("mode", "unknown"),
        "steps": raw.get("steps"),
        "error": raw.get("error"),
    }


# ===== 单条分析 =====

@router.post("/api/analyze", response_model=AnalyzeSingleResponse)
async def analyze_single(req: AnalyzeSingleRequest, request: Request):
    """分析单条中文电商评论"""
    agent = get_agent()
    config = get_config()
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))

    redis = get_redis()
    cache_key = _make_cache_key(req.text, req.mode.value)
    if redis:
        cached = await redis.get_cached_result(cache_key)
        if cached:
            logger.info("[%s] 命中缓存: %s...", trace_id, cache_key[:12])
            return AnalyzeSingleResponse(**cached)

    if req.enable_reflection is not None:
        config.ENABLE_REFLECTION = req.enable_reflection

    use_fast = req.mode == "fast" or (req.mode == "auto" and len(req.text) < 30)
    raw_result = await asyncio.to_thread(
        agent.run, req.text, use_fast_path=use_fast, trace_id=trace_id
    )
    normalized = _normalize_result(raw_result)

    if redis:
        await redis.cache_result(cache_key, normalized)

    return AnalyzeSingleResponse(**normalized)


# ===== 流式分析 (SSE) =====

@router.post("/api/analyze/stream")
async def analyze_stream(req: AnalyzeSingleRequest, request: Request):
    """
    流式分析单条评论，返回 Server-Sent Events (SSE)。

    每个事件格式：
        event: <type>
        data: <json>

    事件类型包括：start, token, step_start, thought, tool_call, tool_result,
    final_summary, result, error, done。
    """
    agent = get_agent()
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))

    def _run_generator():
        try:
            for event in agent.run_stream(
                comment=req.text,
                trace_id=trace_id,
                reviewer_id=req.reviewer_id,
                product_id=req.product_id,
                product_name=req.product_name,
            ):
                event_queue.put(event)
        except Exception as e:
            event_queue.put({"type": "error", "content": str(e)})
        finally:
            event_queue.put(None)

    event_queue: queue.Queue = queue.Queue()

    async def _sse_generator():
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, _run_generator)

        while True:
            event = await asyncio.to_thread(event_queue.get)
            if event is None:
                break
            event_type = event.get("type", "message")
            data = _json.dumps(event, ensure_ascii=False, default=str)
            yield {"event": event_type, "data": data}

        if future.done() and future.exception():
            exc = future.exception()
            logger.error("SSE generator 线程异常: %s", exc)
            error_data = _json.dumps(
                {"type": "error", "content": str(exc)}, ensure_ascii=False
            )
            yield {"event": "error", "data": error_data}

    return EventSourceResponse(
        _sse_generator(),
        headers={"X-Trace-ID": trace_id},
    )


# ===== 批量分析 =====

@router.post("/api/analyze/batch", response_model=AnalyzeBatchResponse)
async def analyze_batch(req: AnalyzeBatchRequest):
    """批量分析多条评论（同步执行）"""
    agent = get_agent()
    config = get_config()

    if req.enable_reflection is not None:
        config.ENABLE_REFLECTION = req.enable_reflection

    use_fast = req.mode != "agent"
    start = time.time()

    results = await asyncio.to_thread(agent.run_batch, req.texts, use_fast_path=use_fast)

    items = []
    completed = 0
    failed = 0
    for raw in results:
        normalized = _normalize_result(raw)
        normalized["batch_index"] = raw.get("batch_index", 0)
        item = BatchItemResult(**normalized)
        items.append(item)
        if normalized.get("analysis_complete"):
            completed += 1
        else:
            failed += 1

    total_elapsed = round((time.time() - start) * 1000, 2)

    return AnalyzeBatchResponse(
        total=len(req.texts),
        completed=completed,
        failed=failed,
        results=items,
        total_elapsed_ms=total_elapsed,
    )


# ===== 异步任务 =====

async def _process_async_task(task_id: str, texts: list, mode: str, redis: RedisManager):
    """后台处理异步任务"""
    try:
        await redis.update_task_status(task_id, "processing")

        agent = get_agent()
        use_fast = mode != "agent"
        start = time.time()

        results = await asyncio.to_thread(agent.run_batch, texts, use_fast_path=use_fast)

        normalized_results = []
        completed = 0
        failed = 0
        for raw in results:
            normalized = _normalize_result(raw)
            normalized["batch_index"] = raw.get("batch_index", 0)
            normalized_results.append(normalized)
            if normalized.get("analysis_complete"):
                completed += 1
            else:
                failed += 1

        total_elapsed = round((time.time() - start) * 1000, 2)

        await redis.update_task_status(
            task_id,
            "completed",
            completed=completed,
            failed=failed,
            results=normalized_results,
            total_elapsed_ms=total_elapsed,
        )
        logger.info(f"异步任务 {task_id} 完成: {completed}/{len(texts)} 成功")

    except Exception as e:
        logger.exception(f"异步任务 {task_id} 失败")
        await redis.update_task_status(task_id, "failed", error=str(e))


@router.post("/api/task/submit", response_model=AsyncTaskSubmitResponse)
async def submit_async_task(
    req: AsyncTaskSubmitRequest,
    background_tasks: BackgroundTasks,
):
    """提交异步分析任务"""
    redis = get_redis()
    if not redis or not await redis.ping():
        raise HTTPException(
            status_code=503,
            detail="异步任务需要 Redis 服务，当前 Redis 不可用",
        )

    task_id = str(uuid.uuid4())
    await redis.create_task(task_id, len(req.texts), req.texts, req.mode.value)

    background_tasks.add_task(
        _process_async_task, task_id, req.texts, req.mode.value, redis
    )

    return AsyncTaskSubmitResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        total=len(req.texts),
        message=f"任务已提交，共 {len(req.texts)} 条评论",
    )


@router.get("/api/task/{task_id}", response_model=AsyncTaskStatusResponse)
async def get_task_status(task_id: str):
    """查询异步任务状态"""
    redis = get_redis()
    if not redis:
        raise HTTPException(status_code=503, detail="Redis 不可用")

    task = await redis.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

    return AsyncTaskStatusResponse(
        task_id=task["task_id"],
        status=TaskStatus(task["status"]),
        total=task["total"],
        completed=task["completed"],
        failed=task["failed"],
        results=task.get("results") if task["status"] == "completed" else None,
        total_elapsed_ms=float(task.get("total_elapsed_ms", 0)) if task.get("total_elapsed_ms") else None,
        error=task.get("error") or None,
    )


# ===== 健康检查 =====

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """服务健康检查"""
    agent = get_agent()
    redis = get_redis()

    redis_ok = False
    if redis:
        redis_ok = await redis.ping()

    return HealthResponse(
        status="ok",
        agent_ready=agent is not None,
        redis_connected=redis_ok,
        tools_loaded=len(agent.tools) if agent else 0,
    )

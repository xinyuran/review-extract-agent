"""
FastAPI 应用入口

启动方式：
    uvicorn extract_agent.api.app:app --host 0.0.0.0 --port 8000 --reload

环境变量：
    REDIS_URL              Redis 连接地址（默认 redis://127.0.0.1:6379/0）
    AGENT_LLM_BASE_URL     Agent LLM 服务地址
    TOOL_LLM_BASE_URL      Tool LLM 服务地址
    API_KEY                API 认证密钥（未设置则不启用认证）
    CORS_ALLOWED_ORIGINS   CORS 允许的域名（逗号分隔）
"""

import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from starlette.middleware.base import BaseHTTPMiddleware

from .metrics import ACTIVE_REQUESTS, REQUEST_COUNT, REQUEST_DURATION
from .redis_client import RedisManager
from .routes import router, set_redis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace ID 中间件
# ---------------------------------------------------------------------------

class TraceIDMiddleware(BaseHTTPMiddleware):
    """为每个请求注入 trace_id，贯穿日志和响应头。"""

    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.trace_id = trace_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = trace_id
        return response


# ---------------------------------------------------------------------------
# API Key 认证中间件
# ---------------------------------------------------------------------------

_API_KEY = os.getenv("API_KEY", "").strip()
_AUTH_SKIP_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    简单的 API Key 认证。

    当环境变量 API_KEY 已设置时启用，通过 Authorization: Bearer <key>
    或 X-API-Key 头传递。未设置 API_KEY 时所有请求放行。
    """

    async def dispatch(self, request: Request, call_next):
        if not _API_KEY:
            return await call_next(request)

        if request.url.path in _AUTH_SKIP_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        api_key_header = request.headers.get("X-API-Key", "")

        provided_key = ""
        if auth_header.startswith("Bearer "):
            provided_key = auth_header[7:].strip()
        elif api_key_header:
            provided_key = api_key_header.strip()

        if not hmac.compare_digest(provided_key.encode(), _API_KEY.encode()):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": "无效的 API Key"},
            )

        return await call_next(request)


# ---------------------------------------------------------------------------
# Prometheus 监控中间件
# ---------------------------------------------------------------------------

class PrometheusMiddleware(BaseHTTPMiddleware):
    """采集 /api/ 路径下的请求计数、耗时和并发数。"""

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        ACTIVE_REQUESTS.inc()
        start = time.time()
        try:
            response = await call_next(request)
            status = "success" if response.status_code < 400 else "error"
            mode = request.url.path.split("/")[-1]
            REQUEST_COUNT.labels(mode=mode, status=status).inc()
            REQUEST_DURATION.labels(mode=mode).observe(time.time() - start)
            return response
        except Exception:
            REQUEST_COUNT.labels(mode="unknown", status="error").inc()
            raise
        finally:
            ACTIVE_REQUESTS.dec()


# ---------------------------------------------------------------------------
# 应用初始化
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    redis = RedisManager()
    try:
        await redis.connect()
        set_redis(redis)
        logger.info("Redis 连接成功")
    except Exception as e:
        logger.warning(f"Redis 连接失败（异步任务不可用）: {e}")

    yield

    await redis.disconnect()


app = FastAPI(
    title="Extract Agent API",
    description=(
        "中文电商评论智能分析 API\n\n"
        "基于 ReAct 范式的 AI Agent，支持关键词提取、情感分析、自反思修正。\n"
        "提供单条分析、批量分析和异步任务三种模式。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(TraceIDMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(PrometheusMiddleware)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

_cors_origins = os.getenv(
    "CORS_ALLOWED_ORIGINS", "http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

"""
FastAPI 应用入口

启动方式：
    uvicorn extract_agent.api.app:app --host 0.0.0.0 --port 8000 --reload

环境变量：
    REDIS_URL              Redis 连接地址（默认 redis://127.0.0.1:6379/0）
    AGENT_LLM_BASE_URL     Agent LLM 服务地址
    TOOL_LLM_BASE_URL      Tool LLM 服务地址
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .redis_client import RedisManager
from .routes import router, set_redis

logger = logging.getLogger(__name__)


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

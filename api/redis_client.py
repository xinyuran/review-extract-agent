"""
Redis 客户端封装

提供异步任务队列、结果缓存和任务状态管理。
使用 redis-py 异步接口，与 FastAPI 的异步特性配合。
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
TASK_KEY_PREFIX = "extract_agent:task:"
RESULT_KEY_PREFIX = "extract_agent:result:"
TASK_QUEUE_KEY = "extract_agent:queue"
RESULT_EXPIRE_SECONDS = int(os.getenv("RESULT_EXPIRE_SECONDS", "3600"))


class RedisManager:
    """
    Redis 管理器

    职责：
    - 异步任务的状态管理（pending → processing → completed/failed）
    - 批量任务结果的分片存储与查询
    - 结果缓存（TTL 自动过期）
    """

    def __init__(self, redis_url: str = REDIS_URL):
        self._redis_url = redis_url
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        if self._redis is None:
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                max_connections=20,
            )
            logger.info(f"Redis 连接已建立: {self._redis_url}")

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("Redis 连接已关闭")

    async def ping(self) -> bool:
        try:
            if self._redis is None:
                return False
            return await self._redis.ping()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 任务状态管理
    # ------------------------------------------------------------------

    async def create_task(
        self, task_id: str, total: int, texts: List[str], mode: str
    ) -> None:
        """创建异步任务记录"""
        task_data = {
            "task_id": task_id,
            "status": "pending",
            "total": total,
            "completed": 0,
            "failed": 0,
            "mode": mode,
            "texts": json.dumps(texts, ensure_ascii=False),
            "results": "[]",
            "error": "",
        }
        key = f"{TASK_KEY_PREFIX}{task_id}"
        await self._redis.hset(key, mapping=task_data)
        await self._redis.expire(key, RESULT_EXPIRE_SECONDS)
        await self._redis.lpush(TASK_QUEUE_KEY, task_id)

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        key = f"{TASK_KEY_PREFIX}{task_id}"
        data = await self._redis.hgetall(key)
        if not data:
            return None

        data["total"] = int(data.get("total", 0))
        data["completed"] = int(data.get("completed", 0))
        data["failed"] = int(data.get("failed", 0))
        if data.get("results"):
            data["results"] = json.loads(data["results"])
        else:
            data["results"] = []
        if data.get("texts"):
            data["texts"] = json.loads(data["texts"])
        return data

    async def update_task_status(self, task_id: str, status: str, **kwargs) -> None:
        """更新任务状态和附加字段"""
        key = f"{TASK_KEY_PREFIX}{task_id}"
        update: Dict[str, str] = {"status": status}
        for k, v in kwargs.items():
            if isinstance(v, (dict, list)):
                update[k] = json.dumps(v, ensure_ascii=False, default=str)
            else:
                update[k] = str(v)
        await self._redis.hset(key, mapping=update)

    async def pop_task_from_queue(self) -> Optional[str]:
        """从任务队列中弹出一个任务 ID（非阻塞）"""
        return await self._redis.rpop(TASK_QUEUE_KEY)

    # ------------------------------------------------------------------
    # 结果缓存
    # ------------------------------------------------------------------

    async def cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """缓存分析结果"""
        key = f"{RESULT_KEY_PREFIX}{cache_key}"
        await self._redis.setex(
            key,
            RESULT_EXPIRE_SECONDS,
            json.dumps(result, ensure_ascii=False, default=str),
        )

    async def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存的分析结果"""
        key = f"{RESULT_KEY_PREFIX}{cache_key}"
        data = await self._redis.get(key)
        if data:
            return json.loads(data)
        return None

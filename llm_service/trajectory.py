"""
轨迹记录器

作为 LLM Service 的可插拔组件，自动记录每次 LLM 调用的完整轨迹：
- Agent LLM 轨迹：完整 messages + tool_calls + tool_result
- Tool LLM 轨迹：skill_name + messages + response

输出为 JSONL 格式，供后续 SFT 数据导出使用。
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import LLMResponse

logger = logging.getLogger(__name__)


class TrajectoryRecorder:
    """
    轨迹记录器

    分别记录 Agent LLM 和 Tool LLM 的调用轨迹到 JSONL 文件。
    每个 session 对应一组文件。
    """

    def __init__(
        self,
        output_dir: str = "extract_agent_output/trajectory",
        session_id: Optional[str] = None,
        include_thinking: bool = True,
    ):
        self._output_dir = Path(output_dir)
        self._session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._include_thinking = include_thinking

        self._session_dir = self._output_dir / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._agent_file = self._session_dir / "agent_trajectory.jsonl"
        self._tool_file = self._session_dir / "tool_trajectory.jsonl"

        self._agent_turns: List[Dict[str, Any]] = []
        self._tool_turns: List[Dict[str, Any]] = []

        self._current_agent_messages: List[Dict[str, Any]] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    # ------------------------------------------------------------------
    # Agent LLM 轨迹记录
    # ------------------------------------------------------------------

    def record_agent_turn(
        self,
        messages: List[Dict[str, Any]],
        response: LLMResponse,
        elapsed_ms: float = 0,
    ) -> None:
        """记录 Agent LLM 的一轮交互"""
        self._current_agent_messages = list(messages)

        turn = {
            "type": "agent_llm",
            "timestamp": datetime.now().isoformat(),
            "messages_snapshot": messages,
            "response": {
                "content": response.content,
                "tool_calls": response.tool_calls,
                "finish_reason": response.finish_reason,
                "model": response.model,
                "usage": response.usage,
            },
            "elapsed_ms": elapsed_ms,
        }

        self._agent_turns.append(turn)
        self._append_jsonl(self._agent_file, turn)

    def record_tool_execution(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        tool_call_id: Optional[str] = None,
    ) -> None:
        """记录工具执行结果（纯计算工具或 LLM 工具）"""
        record = {
            "type": "tool_execution",
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "arguments": arguments,
            "result": result,
        }
        self._agent_turns.append(record)
        self._append_jsonl(self._agent_file, record)

    # ------------------------------------------------------------------
    # Tool LLM 轨迹记录
    # ------------------------------------------------------------------

    def record_tool_turn(
        self,
        skill_name: str,
        messages: List[Dict[str, Any]],
        response: LLMResponse,
        elapsed_ms: float = 0,
    ) -> None:
        """记录 Tool LLM 的一次调用"""
        turn = {
            "type": "tool_llm",
            "timestamp": datetime.now().isoformat(),
            "skill_name": skill_name,
            "messages": messages,
            "response": {
                "content": response.content,
                "finish_reason": response.finish_reason,
                "model": response.model,
                "usage": response.usage,
            },
            "elapsed_ms": elapsed_ms,
        }

        self._tool_turns.append(turn)
        self._append_jsonl(self._tool_file, turn)

    # ------------------------------------------------------------------
    # 完整会话轨迹
    # ------------------------------------------------------------------

    def finalize_session(
        self,
        final_messages: Optional[List[Dict[str, Any]]] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        结束当前 session，保存会话摘要。

        Args:
            final_messages: Agent 的最终完整 messages 列表
            result: 最终分析结果
        """
        summary = {
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "agent_turns": len(self._agent_turns),
            "tool_turns": len(self._tool_turns),
            "final_messages": final_messages,
            "result_summary": {
                "keywords_count": len(result.get("keywords", [])) if result else 0,
                "sentiment": result.get("sentiment", {}).get("label") if result else None,
                "mode": result.get("mode") if result else None,
            } if result else None,
        }

        summary_file = self._session_dir / "session_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary

    def get_full_agent_trajectory(self) -> List[Dict[str, Any]]:
        return list(self._agent_turns)

    def get_full_tool_trajectory(self) -> List[Dict[str, Any]]:
        return list(self._tool_turns)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning("写入轨迹文件失败 %s: %s", path, e)

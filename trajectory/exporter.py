"""
轨迹数据导出器

读取 TrajectoryRecorder 产出的 JSONL 文件，
转换为三种 SFT 训练数据格式并导出。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .formats import SFTFormatter

logger = logging.getLogger(__name__)


class TrajectoryExporter:
    """从轨迹文件目录导出训练数据"""

    def __init__(self, trajectory_dir: str):
        self._base_dir = Path(trajectory_dir)

    def list_sessions(self) -> List[str]:
        """列出所有可用的 session 目录"""
        if not self._base_dir.is_dir():
            return []
        return sorted([
            d.name for d in self._base_dir.iterdir()
            if d.is_dir() and (d / "agent_trajectory.jsonl").exists()
        ])

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if not path.exists():
            return records
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("跳过无效 JSONL 行: %s", line[:100])
        return records

    def load_session(self, session_id: str) -> tuple:
        """
        加载指定 session 的轨迹数据。

        Returns:
            (agent_turns, tool_turns) 元组
        """
        session_dir = self._base_dir / session_id
        agent_turns = self._load_jsonl(session_dir / "agent_trajectory.jsonl")
        tool_turns = self._load_jsonl(session_dir / "tool_trajectory.jsonl")
        return agent_turns, tool_turns

    def export_agent_sft(
        self,
        output_path: str,
        session_ids: Optional[List[str]] = None,
    ) -> int:
        """
        导出 OpenAI native tool_calls SFT 数据。

        Args:
            output_path: 输出 JSONL 文件路径
            session_ids: 指定 session，None 表示全部

        Returns:
            导出的样本数
        """
        sessions = session_ids or self.list_sessions()
        all_samples: List[Dict[str, Any]] = []

        for sid in sessions:
            agent_turns, _ = self.load_session(sid)
            samples = SFTFormatter.format_agent_sft(agent_turns)
            for s in samples:
                s["session_id"] = sid
            all_samples.extend(samples)

        self._write_jsonl(output_path, all_samples)
        logger.info("导出 Agent SFT 数据: %d 条 → %s", len(all_samples), output_path)
        return len(all_samples)

    def export_tool_sft(
        self,
        output_path: str,
        session_ids: Optional[List[str]] = None,
    ) -> int:
        """
        导出 Tool LLM SFT 数据。

        Returns:
            导出的样本数
        """
        sessions = session_ids or self.list_sessions()
        all_samples: List[Dict[str, Any]] = []

        for sid in sessions:
            _, tool_turns = self.load_session(sid)
            samples = SFTFormatter.format_tool_sft(tool_turns)
            for s in samples:
                s["session_id"] = sid
            all_samples.extend(samples)

        self._write_jsonl(output_path, all_samples)
        logger.info("导出 Tool SFT 数据: %d 条 → %s", len(all_samples), output_path)
        return len(all_samples)

    def export_tool_supervision(
        self,
        output_path: str,
        session_ids: Optional[List[str]] = None,
    ) -> int:
        """
        导出工具调用监督三元组。

        Returns:
            导出的样本数
        """
        sessions = session_ids or self.list_sessions()
        all_triples: List[Dict[str, Any]] = []

        for sid in sessions:
            agent_turns, tool_turns = self.load_session(sid)
            triples = SFTFormatter.format_tool_supervision(agent_turns, tool_turns)
            for t in triples:
                t["session_id"] = sid
            all_triples.extend(triples)

        self._write_jsonl(output_path, all_triples)
        logger.info("导出工具监督数据: %d 条 → %s", len(all_triples), output_path)
        return len(all_triples)

    def get_stats(
        self, session_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """统计已采集轨迹的数量和分布"""
        sessions = session_ids or self.list_sessions()
        total_agent_turns = 0
        total_tool_turns = 0
        skill_counts: Dict[str, int] = {}

        for sid in sessions:
            agent_turns, tool_turns = self.load_session(sid)
            total_agent_turns += len(agent_turns)
            total_tool_turns += len(tool_turns)
            for turn in tool_turns:
                skill = turn.get("skill_name", "unknown")
                skill_counts[skill] = skill_counts.get(skill, 0) + 1

        return {
            "total_sessions": len(sessions),
            "total_agent_turns": total_agent_turns,
            "total_tool_turns": total_tool_turns,
            "skill_distribution": skill_counts,
        }

    @staticmethod
    def _write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

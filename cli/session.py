"""
CLI Session 管理器 — 管理 Agent 生命周期、结果存储
"""

import json
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AgentConfig
from ..agent.agent import ReviewAnalysisAgent
from .config_loader import DEFAULT_OUTPUT_DIR


class CLISession:
    """
    一次 CLI 调用的生命周期管理。

    单次命令 = 创建 session -> 分析 -> 关闭 session
    REPL = 创建 session -> 分析 -> 分析 -> ... -> 关闭 session
    """

    @classmethod
    def resume_from(
        cls,
        session_dir: Path,
        config: AgentConfig,
    ) -> "CLISession":
        """
        从已保存的 session 目录恢复一个 CLISession。

        读取 session_meta.json 还原元信息，读取 result.json 还原 history。
        兼容旧版多文件格式 (result_001.json, result_002.json ...)。
        """
        meta_path = session_dir / "session_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"未找到 session 元信息: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        mode = meta.get("mode", "agent")
        full_output = meta.get("full_output", False)

        instance = cls.__new__(cls)
        instance.session_id = meta["session_id"]
        instance.created_at = datetime.fromisoformat(meta["created_at"])
        instance.config = config
        instance.mode = mode
        instance.full_output = full_output
        instance.cli_command = meta.get("cli_command", "resumed")
        instance.result_counter = 0
        instance.history = []

        if getattr(config, "ENABLE_KNOWLEDGE", False):
            instance.reviewer_id = meta.get("reviewer_id", f"reviewer_{instance.session_id}")
            instance.product_id = meta.get("product_id", f"product_{instance.session_id}")
        else:
            instance.reviewer_id = None
            instance.product_id = None

        instance.agent = ReviewAnalysisAgent(config)
        instance.output_dir = session_dir

        instance._trajectory_recorder = None
        if getattr(config, "ENABLE_TRAJECTORY", False):
            instance._init_trajectory_recorder(config)

        result_json = session_dir / "result.json"
        if result_json.exists():
            try:
                with open(result_json, "r", encoding="utf-8") as f:
                    records = json.load(f)
                if isinstance(records, list):
                    for record in records:
                        instance.result_counter += 1
                        text = record.get("text", "")
                        result = record.get("result") or record.get("result_summary", {})
                        instance.history.append({
                            "index": instance.result_counter,
                            "text": text,
                            "result": result,
                            "timestamp": record.get("timestamp", ""),
                        })
            except (json.JSONDecodeError, KeyError):
                pass
        else:
            result_files = sorted(session_dir.glob("result_*.json"))
            for rf in result_files:
                try:
                    with open(rf, "r", encoding="utf-8") as f:
                        record = json.load(f)
                    instance.result_counter += 1
                    text = record.get("text", "")
                    result = record.get("result") or record.get("result_summary", {})
                    instance.history.append({
                        "index": instance.result_counter,
                        "text": text,
                        "result": result,
                        "timestamp": record.get("timestamp", ""),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue

        return instance

    @staticmethod
    def list_saved_sessions(output_root: str = DEFAULT_OUTPUT_DIR) -> List[Dict[str, Any]]:
        """
        列出所有已保存的 session（扫描 output_root 下所有包含 session_meta.json 的目录）。
        返回列表按创建时间降序。
        """
        root = Path(output_root)
        sessions = []
        if not root.exists():
            return sessions

        for meta_path in root.rglob("session_meta.json"):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["_dir"] = str(meta_path.parent)
                sessions.append(meta)
            except (json.JSONDecodeError, OSError):
                continue

        sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return sessions

    def __init__(
        self,
        config: AgentConfig,
        mode: str = "agent",
        full_output: bool = False,
        output_root: str = DEFAULT_OUTPUT_DIR,
        cli_command: str = "",
    ):
        self.session_id: str = f"{secrets.token_hex(4)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.created_at: datetime = datetime.now()
        self.config: AgentConfig = config
        self.mode: str = mode
        self.full_output: bool = full_output
        self.cli_command: str = cli_command
        self.result_counter: int = 0
        self.history: List[Dict[str, Any]] = []

        if getattr(config, "ENABLE_KNOWLEDGE", False):
            self.reviewer_id: Optional[str] = f"reviewer_{self.session_id}"
            self.product_id: Optional[str] = f"product_{self.session_id}"
        else:
            self.reviewer_id = None
            self.product_id = None

        self.agent: ReviewAnalysisAgent = ReviewAnalysisAgent(config)

        self._trajectory_recorder = None
        if getattr(config, "ENABLE_TRAJECTORY", False):
            self._init_trajectory_recorder(config)

        today = datetime.now().strftime("%Y-%m-%d")
        self.output_dir: Path = Path(output_root) / today / self.session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_trajectory_recorder(self, config: AgentConfig) -> None:
        """初始化轨迹记录器并设置到 Agent 的 LLMService 中"""
        try:
            from ..llm_service.trajectory import TrajectoryRecorder

            traj_dir = getattr(config, "TRAJECTORY_OUTPUT_DIR", "extract_agent_output/trajectory")
            include_thinking = getattr(config, "TRAJECTORY_INCLUDE_THINKING", True)

            self._trajectory_recorder = TrajectoryRecorder(
                output_dir=traj_dir,
                session_id=self.session_id,
                include_thinking=include_thinking,
            )

            if hasattr(self.agent, "llm_service") and self.agent.llm_service:
                self.agent.llm_service.set_trajectory_recorder(self._trajectory_recorder)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("轨迹记录器初始化失败: %s", e)
            self._trajectory_recorder = None

    def analyze(
        self,
        text: str,
        reviewer_id: Optional[str] = None,
        product_id: Optional[str] = None,
        product_name: str = "",
    ) -> Dict[str, Any]:
        """分析单条评论，返回结果，同时持久化到文件。

        若调用方未显式传入 reviewer_id / product_id，
        则使用 session 级别自动生成的 ID（仅当 ENABLE_KNOWLEDGE 开启时存在）。
        """
        effective_reviewer = reviewer_id or self.reviewer_id
        effective_product = product_id or self.product_id

        use_fast = self.mode == "fast"
        result = self.agent.run(
            text,
            use_fast_path=use_fast,
            reviewer_id=effective_reviewer,
            product_id=effective_product,
            product_name=product_name,
        )

        self.result_counter += 1
        self.history.append({
            "index": self.result_counter,
            "text": text,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })

        self._save_result_always(result, text)

        return result

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量分析，逐条调用 analyze"""
        results = []
        for text in texts:
            result = self.analyze(text)
            results.append(result)
        return results

    def _save_result_always(self, result: Dict[str, Any], text: str) -> Path:
        """将分析结果追加到 result.json（单文件存储所有分析记录）"""
        record = {
            "index": self.result_counter,
            "text": text,
            "timestamp": datetime.now().isoformat(),
        }
        if self.full_output:
            record["result"] = result
        else:
            record["result_summary"] = {
                "analysis_complete": result.get("analysis_complete", False),
                "keywords": result.get("keywords", []),
                "sentiment": result.get("sentiment", {}),
                "elapsed_ms": result.get("elapsed_ms", 0),
                "mode": result.get("mode", ""),
                "steps": result.get("steps", 0),
                "warnings": result.get("warnings", []),
            }

        filepath = self.output_dir / "result.json"
        records: list = []
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, list):
                    records = existing
            except (json.JSONDecodeError, OSError):
                pass

        records.append(record)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return filepath

    def save_result(self, result: Dict[str, Any]) -> Path:
        """兼容旧调用：将完整结果追加到 result.json"""
        filepath = self.output_dir / "result.json"
        records: list = []
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, list):
                    records = existing
            except (json.JSONDecodeError, OSError):
                pass
        records.append(result)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return filepath

    def get_result_path(self) -> Optional[str]:
        """返回结果文件的路径"""
        if self.output_dir:
            return str(self.output_dir / "result.json")
        return None

    def get_history_summary(self) -> List[Dict[str, Any]]:
        """返回分析历史的摘要列表"""
        summaries = []
        for h in self.history:
            r = h["result"]
            keywords = r.get("keywords", [])
            kw_parts = []
            for k in keywords[:5]:
                if isinstance(k, dict):
                    kw_parts.append(k.get("keyword", ""))
                elif isinstance(k, list) and len(k) >= 2:
                    kw_parts.append(str(k[1]))
            kw_str = ", ".join(kw_parts)
            sentiment = r.get("sentiment")
            if sentiment is None:
                label = "-"
            elif isinstance(sentiment, dict):
                raw_label = sentiment.get("label", "unknown")
                label = str(raw_label) if not isinstance(raw_label, str) else raw_label
            else:
                label = "unknown"
            summaries.append({
                "index": h["index"],
                "text_preview": h["text"][:30] + ("..." if len(h["text"]) > 30 else ""),
                "keywords": kw_str,
                "sentiment": label,
                "elapsed_ms": r.get("elapsed_ms", 0),
                "timestamp": h["timestamp"],
            })
        return summaries

    def get_session_info(self) -> Dict[str, Any]:
        """返回当前 session 的信息"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "mode": self.mode,
            "full_output": self.full_output,
            "total_analyzed": self.result_counter,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "config_source": getattr(self.config, "_config_source", "默认配置"),
        }

    def close(self) -> None:
        """关闭 session，写入 session_meta.json（所有模式都保存）"""
        if self._trajectory_recorder:
            try:
                final_result = self.history[-1]["result"] if self.history else None
                self._trajectory_recorder.finalize_session(result=final_result)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("轨迹记录器关闭失败: %s", e)

        meta = {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
            "mode": self.mode,
            "full_output": self.full_output,
            "total_analyzed": self.result_counter,
            "cli_command": self.cli_command,
            "reviewer_id": self.reviewer_id,
            "product_id": self.product_id,
            "config_source": getattr(self.config, "_config_source", "默认配置"),
            "history_summary": [
                {
                    "index": h["index"],
                    "text_preview": h["text"][:50],
                    "timestamp": h["timestamp"],
                }
                for h in self.history
            ],
        }
        meta_path = self.output_dir / "session_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

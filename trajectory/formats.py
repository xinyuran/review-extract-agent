"""
SFT 训练数据格式化器

将原始轨迹数据转换为三种格式：
1. OpenAI native tool_calls SFT (Agent LLM)
2. Tool LLM SFT (keyword/sentiment)
3. 工具调用监督三元组
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class SFTFormatter:
    """将原始轨迹转换为 SFT 训练数据格式"""

    @staticmethod
    def format_agent_sft(
        agent_turns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        格式一：OpenAI native tool_calls SFT（Agent LLM 训练数据）

        从 agent_trajectory 中重建完整的 messages 序列，
        包含 system/user/assistant(tool_calls)/tool 完整对话流。

        Returns:
            列表，每项是一条完整的训练样本 {"messages": [...]}
        """
        samples: List[Dict[str, Any]] = []

        current_messages: Optional[List[Dict[str, Any]]] = None

        for turn in agent_turns:
            if turn.get("type") == "agent_llm":
                snapshot = turn.get("messages_snapshot")
                if snapshot:
                    current_messages = snapshot

                resp = turn.get("response", {})
                if current_messages and resp:
                    assistant_msg: Dict[str, Any] = {"role": "assistant"}
                    if resp.get("content"):
                        assistant_msg["content"] = resp["content"]
                    if resp.get("tool_calls"):
                        assistant_msg["tool_calls"] = resp["tool_calls"]
                        if "content" not in assistant_msg:
                            assistant_msg["content"] = ""

                    if not resp.get("tool_calls") and resp.get("content"):
                        messages_copy = list(current_messages)
                        messages_copy.append(assistant_msg)
                        samples.append({"messages": messages_copy})

        if current_messages and not samples:
            last_turn = None
            for t in reversed(agent_turns):
                if t.get("type") == "agent_llm":
                    last_turn = t
                    break
            if last_turn:
                resp = last_turn.get("response", {})
                assistant_msg = {"role": "assistant"}
                if resp.get("content"):
                    assistant_msg["content"] = resp["content"]
                if resp.get("tool_calls"):
                    assistant_msg["tool_calls"] = resp["tool_calls"]
                messages_copy = list(current_messages)
                messages_copy.append(assistant_msg)
                samples.append({"messages": messages_copy})

        return samples

    @staticmethod
    def format_tool_sft(
        tool_turns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        格式三：Tool LLM SFT 数据

        每条 Tool LLM 调用转为一条训练样本：
        {"messages": [system, user, assistant]}

        Returns:
            列表，每项是 {"messages": [...], "skill_name": str}
        """
        samples: List[Dict[str, Any]] = []

        for turn in tool_turns:
            if turn.get("type") != "tool_llm":
                continue

            messages = turn.get("messages", [])
            resp = turn.get("response", {})
            content = resp.get("content", "")

            if not messages or not content:
                continue

            sample_messages = list(messages)
            sample_messages.append({
                "role": "assistant",
                "content": content,
            })

            samples.append({
                "messages": sample_messages,
                "skill_name": turn.get("skill_name", ""),
                "model": resp.get("model", ""),
            })

        return samples

    @staticmethod
    def format_tool_supervision(
        agent_turns: List[Dict[str, Any]],
        tool_turns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        格式二：工具调用监督三元组

        (tool_name, input, result) 三元组，来源于：
        - agent_trajectory 中的 tool_execution 记录
        - tool_trajectory 中的 tool_llm 记录

        Returns:
            列表，每项是 {"tool_name": str, "input": dict, "result": dict, ...}
        """
        triples: List[Dict[str, Any]] = []

        for turn in agent_turns:
            if turn.get("type") != "tool_execution":
                continue
            triples.append({
                "tool_name": turn.get("tool_name", ""),
                "input": turn.get("arguments", {}),
                "result": turn.get("result", {}),
                "timestamp": turn.get("timestamp", ""),
            })

        for turn in tool_turns:
            if turn.get("type") != "tool_llm":
                continue
            resp = turn.get("response", {})
            skill_name = turn.get("skill_name", "")
            input_messages = turn.get("messages", [])
            user_msg = ""
            for m in input_messages:
                if m.get("role") == "user":
                    user_msg = m.get("content", "")

            triples.append({
                "tool_name": skill_name,
                "input": {"text": user_msg},
                "result": {"raw_output": resp.get("content", "")},
                "skill_used": skill_name,
                "model": resp.get("model", ""),
                "timestamp": turn.get("timestamp", ""),
            })

        return triples

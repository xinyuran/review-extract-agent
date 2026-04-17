"""
Agent 工作记忆管理

维护当前会话的完整消息历史（OpenAI messages 格式），
支持系统消息、用户消息、助手消息、工具调用及工具返回。
"""

from typing import Any, Dict, List, Optional
import copy


class AgentMemory:
    """
    ReAct Agent 的工作记忆。

    管理一次分析会话中的所有消息流转，包括：
    - 系统提示（system prompt）
    - 用户请求
    - Agent LLM 的思考与工具调用
    - 工具执行结果
    - Agent 最终回答
    """

    def __init__(self, system_prompt: str):
        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # 消息操作
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        记录 Agent LLM 的回复。

        Args:
            content: 文本回复内容（Agent 直接回答时使用）
            tool_calls: 工具调用列表（Agent 决定使用工具时使用）
        """
        msg: Dict[str, Any] = {"role": "assistant"}
        if content is not None:
            msg["content"] = content
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
            if content is None:
                msg["content"] = ""
        self._messages.append(msg)
        self._step_count += 1

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> None:
        """
        记录工具执行结果。

        Args:
            tool_call_id: 对应的 tool_call 的 id
            tool_name: 工具名称
            result: 工具返回的字符串化结果
        """
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        })

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def to_messages(self) -> List[Dict[str, Any]]:
        """返回完整消息历史（深拷贝，防止外部修改）"""
        return copy.deepcopy(self._messages)

    def get_step_count(self) -> int:
        """返回 Agent 已执行的决策步数（每次 LLM 回复计为一步）"""
        return self._step_count

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """返回最后一条消息"""
        return self._messages[-1] if self._messages else None

    def get_message_count(self) -> int:
        """返回总消息条数"""
        return len(self._messages)

    # TODO: 如果后续需要更详细的记忆摘要，可以考虑使用更复杂的摘要方法。而不是简单地截取前80个字符。
    def summarize(self) -> str:
        """生成记忆摘要（调试用）"""
        lines = [f"Memory: {len(self._messages)} messages, {self._step_count} steps"]
        for msg in self._messages:
            role = msg["role"]
            if role == "system":
                lines.append(f"  [system] (length={len(msg.get('content', ''))})")
            elif role == "user":
                content = msg.get("content", "")
                lines.append(f"  [user] {content[:80]}{'...' if len(content) > 80 else ''}")
            elif role == "assistant":
                tc = msg.get("tool_calls")
                if tc:
                    names = [c.get("function", {}).get("name", "?") for c in tc]
                    lines.append(f"  [assistant] tool_calls={names}")
                else:
                    content = msg.get("content", "")
                    lines.append(f"  [assistant] {content[:80]}{'...' if len(content) > 80 else ''}")
            elif role == "tool":
                name = msg.get("name", "?")
                content = msg.get("content", "")
                lines.append(f"  [tool:{name}] {content[:60]}{'...' if len(content) > 60 else ''}")
        return "\n".join(lines)

"""
LLM Service 层数据模型
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillPrompt:
    """从 SKILL.md 文件解析并注入变量后的结果"""

    name: str
    description: str
    target: str  # "agent_llm" | "tool_llm"
    system: str
    user: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """LLM 调用的统一返回结构"""

    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    role: str = "assistant"
    raw: Any = None
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class LLMStreamChunk:
    """流式 LLM 调用中的单个 chunk"""

    delta_content: Optional[str] = None
    tool_call_index: Optional[int] = None
    tool_call_id: Optional[str] = None
    tool_call_function_name: Optional[str] = None
    tool_call_function_args_delta: Optional[str] = None
    finish_reason: Optional[str] = None
    model: Optional[str] = None

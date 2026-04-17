"""
工具基类与统一返回模型

所有 Agent 工具继承 BaseTool，通过 to_openai_tool() 自动转换为
OpenAI Function Calling 格式，Agent 调度层无需手写 schema。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """标准工具返回格式"""

    success: bool = Field(description="工具是否执行成功")
    data: Any = Field(default=None, description="执行结果数据")
    error: Optional[str] = Field(default=None, description="错误信息（失败时）")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元信息，如耗时、调用的子模型等",
    )


class BaseTool(ABC):
    """Agent 工具抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具唯一标识符（英文，snake_case）"""

    @property
    @abstractmethod
    def description(self) -> str:
        """给 Agent LLM 阅读的工具功能描述（中文）"""

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """
        JSON Schema 格式的参数定义，与 OpenAI Function Calling 兼容。

        示例::

            {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "待处理的评论文本"
                    }
                },
                "required": ["text"]
            }
        """

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        执行工具逻辑。

        所有参数通过 kwargs 传入，与 parameters_schema 中定义的字段对应。
        """

    # ------------------------------------------------------------------
    # OpenAI Function Calling 转换
    # ------------------------------------------------------------------

    def to_openai_tool(self) -> Dict[str, Any]:
        """将工具描述转为 OpenAI ``tools`` 参数所需的格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"

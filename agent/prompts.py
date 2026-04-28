"""
Agent Prompt 定义（向后兼容垫片）

四层架构中 Prompt 已迁移到 skills/ 目录下的 SKILL.md 文件。
本模块保留 build_tool_descriptions_for_prompt 以兼容外部引用，
原始字符串常量标记为 deprecated。

新代码请使用 llm_service.SkillLoader 加载技能文件。
"""

import json
import warnings
from typing import Dict

from ..tools.base_tool import BaseTool


def _load_skill_text(skill_name: str, section: str = "system") -> str:
    """从 SKILL.md 加载指定段落（运行时使用）"""
    try:
        from ..llm_service.skill_loader import SkillLoader
        loader = SkillLoader()
        raw = loader._cache.get(skill_name)
        if raw:
            return getattr(raw, section, "")
    except Exception:
        pass
    return ""


def _deprecated_prompt(name: str) -> str:
    warnings.warn(
        f"{name} 已弃用，请使用 SkillLoader 加载对应的 SKILL.md 文件",
        DeprecationWarning,
        stacklevel=3,
    )
    return ""


AGENT_SYSTEM_PROMPT = _load_skill_text("agent_system", "system") or (
    "你是中文电商评论分析Agent。依次调用工具完成分析，结果由系统自动组装。"
)

USER_REQUEST_TEMPLATE = "分析此评论：{comment}"

BATCH_USER_REQUEST_TEMPLATE = (
    "请逐条分析以下 {count} 条电商评论：\n\n{comments}\n\n"
    "请对每条评论按照标准流程完成完整的分析（关键词提取 + 情感分析），并返回所有结果。"
)

AGENT_SYSTEM_PROMPT_WITH_TOOLS = (
    AGENT_SYSTEM_PROMPT
    + "\n\n## Prompt-based 模式：工具调用格式\n\n"
    "调用工具时，使用 `<tool_call>` 标签输出 JSON，字段需与下述 schema 一致。\n\n"
    "{tool_descriptions}\n"
)


def build_tool_descriptions_for_prompt(tools: Dict[str, BaseTool]) -> str:
    """将已注册工具转为可供模型阅读的 Markdown 说明（含 parameters schema）。"""
    parts: list[str] = []
    for tool in tools.values():
        spec = tool.to_openai_tool()
        fn = spec["function"]
        params = json.dumps(fn["parameters"], ensure_ascii=False, indent=2)
        parts.append(
            f"### `{fn['name']}`\n{fn['description']}\n\n```json\n{params}\n```"
        )
    return "\n\n".join(parts)

"""
SKILL.md 文件解析器

解析 skills/ 目录下的 .skill.md 文件，支持：
- YAML frontmatter 元数据
- ## system / ## user 段落分割
- {{variable}} 动态变量注入
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .models import SkillPrompt

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION_RE = re.compile(r"^##\s+(\S+)\s*$", re.MULTILINE)
_VAR_RE = re.compile(r"\{\{(\w+)\}\}")


class SkillLoader:
    """加载并缓存 skills/ 目录下的 SKILL.md 文件"""

    def __init__(self, skills_dir: Optional[str] = None):
        if skills_dir is None:
            skills_dir = str(Path(__file__).resolve().parent.parent / "skills")
        self._skills_dir = Path(skills_dir)
        self._cache: Dict[str, _RawSkill] = {}
        self._scan()

    def _scan(self) -> None:
        if not self._skills_dir.is_dir():
            logger.warning("Skills directory not found: %s", self._skills_dir)
            return
        for p in self._skills_dir.glob("*.skill.md"):
            name = p.stem.replace(".skill", "")
            try:
                self._cache[name] = _parse_skill_file(p)
                logger.debug("Loaded skill: %s from %s", name, p.name)
            except Exception:
                logger.exception("Failed to parse skill file: %s", p)

    @property
    def available_skills(self) -> list[str]:
        return list(self._cache.keys())

    def load(self, skill_name: str, **variables: Any) -> SkillPrompt:
        """
        加载指定技能并注入变量。

        Raises:
            KeyError: 技能名不存在
            ValueError: 缺少必需变量
        """
        raw = self._cache.get(skill_name)
        if raw is None:
            raise KeyError(
                f"Skill '{skill_name}' not found. "
                f"Available: {self.available_skills}"
            )

        required = {
            v["name"] if isinstance(v, dict) else v
            for v in raw.meta.get("variables", [])
            if (isinstance(v, dict) and v.get("required", False))
            or isinstance(v, str)
        }
        missing = required - set(variables.keys())
        if missing:
            raise ValueError(
                f"Skill '{skill_name}' missing required variables: {missing}"
            )

        system = _inject_vars(raw.system, variables)
        user = _inject_vars(raw.user, variables)

        return SkillPrompt(
            name=raw.meta.get("name", skill_name),
            description=raw.meta.get("description", ""),
            target=raw.meta.get("target", "agent_llm"),
            system=system,
            user=user,
            metadata=raw.meta,
        )

    def reload(self) -> None:
        self._cache.clear()
        self._scan()


class _RawSkill:
    __slots__ = ("meta", "system", "user")

    def __init__(self, meta: Dict[str, Any], system: str, user: str):
        self.meta = meta
        self.system = system
        self.user = user


def _parse_skill_file(path: Path) -> _RawSkill:
    text = path.read_text(encoding="utf-8")

    fm_match = _FRONTMATTER_RE.match(text)
    if fm_match:
        meta = yaml.safe_load(fm_match.group(1)) or {}
        body = text[fm_match.end():]
    else:
        meta = {}
        body = text

    sections = _split_sections(body)
    system = sections.get("system", "").strip()
    user = sections.get("user", "").strip()

    if not system and not user:
        logger.warning("Skill file %s has no ## system or ## user section", path.name)
    elif not system:
        logger.debug("Skill file %s has no ## system section (user-only template)", path.name)

    return _RawSkill(meta=meta, system=system, user=user)


def _split_sections(body: str) -> Dict[str, str]:
    """将 markdown 正文按 ## heading 切分为 {heading: content}"""
    parts: Dict[str, str] = {}
    positions = [(m.start(), m.end(), m.group(1).lower()) for m in _SECTION_RE.finditer(body)]

    for i, (start, end, name) in enumerate(positions):
        next_start = positions[i + 1][0] if i + 1 < len(positions) else len(body)
        content = body[end:next_start]
        if name in parts:
            parts[name] += "\n" + content
        else:
            parts[name] = content

    return parts


def _inject_vars(template: str, variables: Dict[str, Any]) -> str:
    def _replacer(m: re.Match) -> str:
        key = m.group(1)
        if key in variables:
            return str(variables[key])
        return m.group(0)

    return _VAR_RE.sub(_replacer, template)

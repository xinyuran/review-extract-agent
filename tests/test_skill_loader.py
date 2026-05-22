"""
llm_service/skill_loader.py 单元测试

使用 tmp_path fixture 创建临时 .skill.md 文件，
验证 SkillLoader 的 _scan、load、reload 和变量注入逻辑。
"""

import pytest

from extract_agent.llm_service.skill_loader import SkillLoader


SAMPLE_SKILL = """\
---
name: test_skill
description: A test skill
target: tool_llm
variables:
  - name: comment
    required: true
  - name: lang
    required: false
---

## system

You are an assistant analyzing: {{comment}}

## user

Please extract keywords from: {{comment}} (lang={{lang}})
"""

SKILL_NO_VARS = """\
---
name: no_vars_skill
description: Skill without variables
target: agent_llm
---

## system

System prompt without variables.

## user

User prompt without variables.
"""

SKILL_NO_FRONTMATTER = """\
## system

Bare system prompt.

## user

Bare user prompt.
"""


@pytest.fixture
def skills_dir(tmp_path):
    """创建临时 skills 目录并写入测试用技能文件。"""
    (tmp_path / "test_skill.skill.md").write_text(SAMPLE_SKILL, encoding="utf-8")
    (tmp_path / "no_vars_skill.skill.md").write_text(SKILL_NO_VARS, encoding="utf-8")
    return tmp_path


@pytest.fixture
def loader(skills_dir):
    return SkillLoader(skills_dir=str(skills_dir))


class TestScan:

    def test_scans_skill_files(self, loader):
        assert "test_skill" in loader.available_skills
        assert "no_vars_skill" in loader.available_skills

    def test_empty_directory(self, tmp_path):
        loader = SkillLoader(skills_dir=str(tmp_path))
        assert loader.available_skills == []

    def test_nonexistent_directory(self, tmp_path):
        loader = SkillLoader(skills_dir=str(tmp_path / "nonexistent"))
        assert loader.available_skills == []

    def test_ignores_non_skill_files(self, tmp_path):
        (tmp_path / "README.md").write_text("# Not a skill", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")
        loader = SkillLoader(skills_dir=str(tmp_path))
        assert loader.available_skills == []

    def test_reload_picks_up_new_files(self, skills_dir):
        loader = SkillLoader(skills_dir=str(skills_dir))
        assert "new_skill" not in loader.available_skills

        (skills_dir / "new_skill.skill.md").write_text(SKILL_NO_VARS, encoding="utf-8")
        loader.reload()
        assert "new_skill" in loader.available_skills


class TestLoad:

    def test_load_with_required_variable(self, loader):
        result = loader.load("test_skill", comment="好评如潮")
        assert "好评如潮" in result.system
        assert "好评如潮" in result.user
        assert result.name == "test_skill"
        assert result.target == "tool_llm"

    def test_load_with_optional_variable(self, loader):
        result = loader.load("test_skill", comment="text", lang="zh")
        assert "lang=zh" in result.user

    def test_optional_variable_unset_kept_as_placeholder(self, loader):
        result = loader.load("test_skill", comment="text")
        assert "{{lang}}" in result.user

    def test_missing_required_variable_raises(self, loader):
        with pytest.raises(ValueError, match="missing required variables"):
            loader.load("test_skill")

    def test_nonexistent_skill_raises(self, loader):
        with pytest.raises(KeyError, match="not found"):
            loader.load("nonexistent_skill")

    def test_skill_without_variables(self, loader):
        result = loader.load("no_vars_skill")
        assert "System prompt without variables" in result.system
        assert result.target == "agent_llm"

    def test_skill_without_frontmatter(self, tmp_path):
        (tmp_path / "bare.skill.md").write_text(SKILL_NO_FRONTMATTER, encoding="utf-8")
        loader = SkillLoader(skills_dir=str(tmp_path))
        result = loader.load("bare")
        assert "Bare system prompt" in result.system
        assert "Bare user prompt" in result.user
        assert result.metadata == {}


class TestVariableInjection:

    def test_multiple_occurrences_replaced(self, loader):
        result = loader.load("test_skill", comment="双重替换")
        assert result.system.count("双重替换") == 1
        assert result.user.count("双重替换") == 1

    def test_special_characters_in_variable(self, loader):
        val = '包含 "引号" 和 {大括号}'
        result = loader.load("test_skill", comment=val)
        assert val in result.system

    def test_numeric_variable_converted_to_string(self, loader):
        result = loader.load("test_skill", comment=12345)
        assert "12345" in result.system

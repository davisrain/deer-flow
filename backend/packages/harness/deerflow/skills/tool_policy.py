import logging
from typing import Protocol

from deerflow.skills.types import Skill

logger = logging.getLogger(__name__)


class NamedTool(Protocol):
    name: str


def allowed_tool_names_for_skills(skills: list[Skill]) -> set[str] | None:
    """Return the union of explicit skill allowed-tools declarations.

    None means legacy allow-all behavior. It is returned only when no loaded
    skill declares allowed-tools. Once any skill declares the field, legacy
    skills without the field contribute no tools instead of disabling the
    explicit restrictions from other skills.
    """
    if not skills:
        return None

    allowed: set[str] = set()
    has_explicit_declaration = False
    # 遍历skills
    for skill in skills:
        # 如果skill中没有配置allowed_tools，继续遍历
        if skill.allowed_tools is None:
            continue
        # 如果有配置allowed_tools，添加进set中，并且设置标识
        has_explicit_declaration = True
        if not skill.allowed_tools:
            logger.info("Skill %s declared empty allowed-tools", skill.name)
        allowed.update(skill.allowed_tools)

    # 如果skill中有声明allowed_tools，返回这些工具，否则返回None
    if not has_explicit_declaration:
        return None
    return allowed


def filter_tools_by_skill_allowed_tools[ToolT: NamedTool](tools: list[ToolT], skills: list[Skill]) -> list[ToolT]:
    # 如果skill中声明了allowed_tools，那么只返回这些允许的工具；
    # 否则，返回所有工具，不进行过滤
    allowed = allowed_tool_names_for_skills(skills)
    if allowed is None:
        return tools

    return [tool for tool in tools if tool.name in allowed]

"""Subagent registry for managing available subagents."""

import logging
from dataclasses import replace
from typing import Any

from deerflow.sandbox.security import is_host_bash_allowed
from deerflow.subagents.builtins import BUILTIN_SUBAGENTS
from deerflow.subagents.config import SubagentConfig

logger = logging.getLogger(__name__)


def _resolve_subagents_app_config(app_config: Any | None = None):
    if app_config is None:
        from deerflow.config.subagents_config import get_subagents_app_config

        return get_subagents_app_config()
    # 获取subagent模块的内容
    return getattr(app_config, "subagents", app_config)


def _build_custom_subagent_config(name: str, *, app_config: Any | None = None) -> SubagentConfig | None:
    """Build a SubagentConfig from config.yaml custom_agents section.

    Args:
        name: The name of the custom subagent.
        app_config: Optional AppConfig or SubagentsAppConfig to resolve from.

    Returns:
        SubagentConfig if found in custom_agents, None otherwise.
    """
    subagents_config = _resolve_subagents_app_config(app_config)
    custom = subagents_config.custom_agents.get(name)
    if custom is None:
        return None

    return SubagentConfig(
        name=name,
        description=custom.description,
        system_prompt=custom.system_prompt,
        tools=custom.tools,
        disallowed_tools=custom.disallowed_tools,
        skills=custom.skills,
        model=custom.model,
        max_turns=custom.max_turns,
        timeout_seconds=custom.timeout_seconds,
    )


def get_subagent_config(name: str, *, app_config: Any | None = None) -> SubagentConfig | None:
    """Get a subagent configuration by name, with config.yaml overrides applied.

    Resolution order (mirrors Codex's config layering):
    1. Built-in subagents (general-purpose, bash)
    2. Custom subagents from config.yaml custom_agents section
    3. Per-agent overrides from config.yaml agents section (timeout, max_turns, model, skills)

    Args:
        name: The name of the subagent.
        app_config: Optional AppConfig or SubagentsAppConfig to resolve overrides from.

    Returns:
        SubagentConfig if found (with any config.yaml overrides applied), None otherwise.
    """
    # Step 1: Look up built-in, then fall back to custom_agents
    # 第一步，从built-in以及custom_agents里面查找配置
    # 如果是内置的subagent，直接通过dict获取
    config = BUILTIN_SUBAGENTS.get(name)
    # 否则，根据配置文件的subagents模块中的配置来构建config
    if config is None:
        config = _build_custom_subagent_config(name, app_config=app_config)
    # 如果仍然为None，返回None，表示没有找到配置
    if config is None:
        return None

    # Step 2: Apply per-agent overrides from config.yaml agents section.
    # Only explicit per-agent overrides are applied here. Global defaults
    # (timeout_seconds, max_turns at the top level) apply to built-in agents
    # but must NOT override custom agents' own values — custom agents define
    # their own defaults in the custom_agents section.
    # 第二步，将配置文件中subagents.agents模块的配置应用给对应的subagent，这些配置就是指定的需要重写的配置

    # 查找到subagents配置模块
    subagents_config = _resolve_subagents_app_config(app_config)
    # 判断这个subagent是否是内置的
    is_builtin = name in BUILTIN_SUBAGENTS
    # 从subagents.agents模块找到该subagent需要重写的配置
    agent_override = subagents_config.agents.get(name)

    overrides = {}

    # Timeout: per-agent override > global default (builtins only) > config's own value
    # 如果override存在，且timeout属性也存在，并且和当前的配置不一样，使用override重写当前的配置
    if agent_override is not None and agent_override.timeout_seconds is not None:
        if agent_override.timeout_seconds != config.timeout_seconds:
            logger.debug("Subagent '%s': timeout overridden (%ss -> %ss)", name, config.timeout_seconds, agent_override.timeout_seconds)
            overrides["timeout_seconds"] = agent_override.timeout_seconds
    # 如果是内置subagent 且 subagents配置模块中global的timeout存在 并且 不等于当前的timeout，使用global的覆盖内置的subagent的配置
    elif is_builtin and subagents_config.timeout_seconds != config.timeout_seconds:
        logger.debug("Subagent '%s': timeout from global default (%ss -> %ss)", name, config.timeout_seconds, subagents_config.timeout_seconds)
        overrides["timeout_seconds"] = subagents_config.timeout_seconds

    # Max turns: per-agent override > global default (builtins only) > config's own value
    # 同理，覆盖max turns字段
    if agent_override is not None and agent_override.max_turns is not None:
        if agent_override.max_turns != config.max_turns:
            logger.debug("Subagent '%s': max_turns overridden (%s -> %s)", name, config.max_turns, agent_override.max_turns)
            overrides["max_turns"] = agent_override.max_turns
    elif is_builtin and subagents_config.max_turns is not None and subagents_config.max_turns != config.max_turns:
        logger.debug("Subagent '%s': max_turns from global default (%s -> %s)", name, config.max_turns, subagents_config.max_turns)
        overrides["max_turns"] = subagents_config.max_turns

    # Model: per-agent override only (no global default for model)
    # model字段，只使用per-agent的重写，subagents模块中没有global的配置
    effective_model = subagents_config.get_model_for(name)
    # 如果per-agent里面重写了model，那么使用重写的
    if effective_model is not None and effective_model != config.model:
        logger.debug("Subagent '%s': model overridden (%s -> %s)", name, config.model, effective_model)
        overrides["model"] = effective_model

    # Skills: per-agent override only (no global default for skills)
    # skills字段，只使用per-agent的重写，subagents模块中没有global的配置
    effective_skills = subagents_config.get_skills_for(name)
    if effective_skills is not None and effective_skills != config.skills:
        logger.debug("Subagent '%s': skills overridden (%s -> %s)", name, config.skills, effective_skills)
        overrides["skills"] = effective_skills

    # 如果overrides有值，填入到subagent config对象中
    if overrides:
        config = replace(config, **overrides)

    return config


def list_subagents(*, app_config: Any | None = None) -> list[SubagentConfig]:
    """List all available subagent configurations (with config.yaml overrides applied).

    Returns:
        List of all registered SubagentConfig instances (built-in + custom).
    """
    configs = []
    for name in get_subagent_names(app_config=app_config):
        config = get_subagent_config(name, app_config=app_config)
        if config is not None:
            configs.append(config)
    return configs


def get_subagent_names(*, app_config: Any | None = None) -> list[str]:
    """Get all available subagent names (built-in + custom).

    Returns:
        List of subagent names.
    """
    # 获取内置的subagent名称，general-purpose和bash
    names = list(BUILTIN_SUBAGENTS.keys())

    # Merge custom_agents from config.yaml
    # 尝试去配置文件中查找subagent的名称，从配置文件中获取subagent的配置
    subagents_config = _resolve_subagents_app_config(app_config)
    # 遍历subagent下的custom_agents的配置，将名称收集起来
    for custom_name in subagents_config.custom_agents:
        if custom_name not in names:
            names.append(custom_name)

    return names


def get_available_subagent_names(*, app_config: Any | None = None) -> list[str]:
    """Get subagent names that should be exposed to the active runtime.

    Returns:
        List of subagent names visible to the current sandbox configuration.
    """
    # 获取subagent的名称，里面包括内置的，还有在config.yaml的subagents模块自定义配置的
    names = get_subagent_names(app_config=app_config)
    try:
        # 查看配置文件中sandbox模块的配置，判断是否允许bash命令
        host_bash_allowed = is_host_bash_allowed(app_config) if hasattr(app_config, "sandbox") else is_host_bash_allowed()
    except Exception:
        logger.debug("Could not determine host bash availability; exposing all subagents")
        return names

    # 如果不允许bash命令的话，将name=bash的subagent名称剔除
    if not host_bash_allowed:
        names = [name for name in names if name != "bash"]
    return names

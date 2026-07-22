"""Run naming helpers for LangChain/LangSmith tracing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_root_run_name(config: Mapping[str, Any], assistant_id: str | None) -> str:
    # 尝试去config的context和configurable里面去查找 agent_name属性，找到了就返回
    for container_name in ("context", "configurable"):
        container = config.get(container_name)
        if isinstance(container, Mapping):
            agent_name = container.get("agent_name")
            if isinstance(agent_name, str) and agent_name.strip():
                return agent_name
    # 没找到的话，如果assistant_id有值，就返回。否则默认返回lead_agent
    return assistant_id or "lead_agent"

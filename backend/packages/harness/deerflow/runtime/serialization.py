"""Canonical serialization for LangChain / LangGraph objects.

Provides a single source of truth for converting LangChain message
objects, Pydantic models, and LangGraph state dicts into plain
JSON-serialisable Python structures.

Consumers: ``deerflow.runtime.runs.worker`` (SSE publishing) and
``app.gateway.routers.threads`` (REST responses).
"""

from __future__ import annotations

from typing import Any


def serialize_lc_object(obj: Any) -> Any:
    """Recursively serialize a LangChain object to a JSON-serialisable dict."""
    if obj is None:
        return None
    # 如果是str int float bool类型的，直接返回
    if isinstance(obj, (str, int, float, bool)):
        return obj
    # 如果dict类型的，递归序列化value后返回
    if isinstance(obj, dict):
        return {k: serialize_lc_object(v) for k, v in obj.items()}
    # 如果是list或者tuple类型的，递归序列化item后返回
    if isinstance(obj, (list, tuple)):
        return [serialize_lc_object(item) for item in obj]
    # Pydantic v2
    # 如果存在model_dump方法，返回调用结果
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1 / older objects
    # 如果存在dict方法，返回调用结果
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    # Last resort
    # 兜底，将obj转换为str返回
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def serialize_channel_values(channel_values: dict[str, Any]) -> dict[str, Any]:
    """Serialize channel values, stripping internal LangGraph keys.

    Internal keys like ``__pregel_*`` and ``__interrupt__`` are removed
    to match what the LangGraph Platform API returns.
    """
    result: dict[str, Any] = {}
    # 遍历channel_values，将__pregel_开头的 以及 __interrupt__过滤属性过滤掉
    for key, value in channel_values.items():
        if key.startswith("__pregel_") or key == "__interrupt__":
            continue
        # 其余属性值使用通用的序列化方法
        result[key] = serialize_lc_object(value)
    return result


def serialize_messages_tuple(obj: Any) -> Any:
    """Serialize a messages-mode tuple ``(chunk, metadata)``."""
    # 如果是llm返回的(chunk, metadata)的结构
    if isinstance(obj, tuple) and len(obj) == 2:
        chunk, metadata = obj
        # chunk使用通用的序列化方法，metadata直接作为dict，组合为列表返回
        return [serialize_lc_object(chunk), metadata if isinstance(metadata, dict) else {}]
    return serialize_lc_object(obj)


def serialize(obj: Any, *, mode: str = "") -> Any:
    """Serialize LangChain objects with mode-specific handling.

    * ``messages`` — obj is ``(message_chunk, metadata_dict)``
    * ``values`` — obj is the full state dict; ``__pregel_*`` keys stripped
    * everything else — recursive ``model_dump()`` / ``dict()`` fallback
    """
    # 对于不同的mode，采用不同的序列化方式
    if mode == "messages":
        return serialize_messages_tuple(obj)
    if mode == "values":
        # 对于values类型的chunk，如果chunk是dict类型的，使用channel的序列化方法，否则使用通用的
        return serialize_channel_values(obj) if isinstance(obj, dict) else serialize_lc_object(obj)
    return serialize_lc_object(obj)

from __future__ import annotations

from typing import Any

from deerflow.config import (
    get_enabled_tracing_providers,
    get_tracing_config,
    validate_enabled_tracing_providers,
)


def _create_langsmith_tracer(config) -> Any:
    from langchain_core.tracers.langchain import LangChainTracer

    # 将langsmith配置中的project属性传入，默认是deer-flow
    return LangChainTracer(project_name=config.project)


def _create_langfuse_handler(config) -> Any:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

    # langfuse>=4 initializes project-specific credentials through the client
    # singleton; the LangChain callback then attaches to that configured client.
    Langfuse(
        secret_key=config.secret_key,
        public_key=config.public_key,
        host=config.host,
    )
    return LangfuseCallbackHandler(public_key=config.public_key)


def build_tracing_callbacks() -> list[Any]:
    """Build callbacks for all explicitly enabled tracing providers."""
    # 校验tracing相关的配置是否有问题，即enable了但没有配置api_key等信息
    validate_enabled_tracing_providers()
    # 返回正确配置的tracing providers
    enabled_providers = get_enabled_tracing_providers()
    # 如果都没有配置，返回空集合
    if not enabled_providers:
        return []

    # 获取tracing配置
    tracing_config = get_tracing_config()
    callbacks: list[Any] = []

    # 遍历enabled的providers
    for provider in enabled_providers:
        # 如果是langsmith
        if provider == "langsmith":
            try:
                # 创建langsmith的callback添加进集合
                callbacks.append(_create_langsmith_tracer(tracing_config.langsmith))
            except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
                raise RuntimeError(f"LangSmith tracing initialization failed: {exc}") from exc
        # 如果是langfuse，创建langfuse的callback添加进集合
        elif provider == "langfuse":
            try:
                callbacks.append(_create_langfuse_handler(tracing_config.langfuse))
            except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
                raise RuntimeError(f"Langfuse tracing initialization failed: {exc}") from exc

    return callbacks

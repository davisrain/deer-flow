"""Background agent execution.

Runs an agent graph inside an ``asyncio.Task``, publishing events to
a :class:`StreamBridge` as they are produced.

Uses ``graph.astream(stream_mode=[...])`` which gives correct full-state
snapshots for ``values`` mode, proper ``{node: writes}`` for ``updates``,
and ``(chunk, metadata)`` tuples for ``messages`` mode.

Note: ``events`` mode is not supported through the gateway — it requires
``graph.astream_events()`` which cannot simultaneously produce ``values``
snapshots.  The JS open-source LangGraph API server works around this via
internal checkpoint callbacks that are not exposed in the Python public API.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, cast

from langgraph.checkpoint.base import empty_checkpoint

if TYPE_CHECKING:
    from langchain_core.messages import HumanMessage

from deerflow.config.app_config import AppConfig
from deerflow.runtime.serialization import serialize
from deerflow.runtime.stream_bridge import StreamBridge
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.tracing import inject_langfuse_metadata

from .manager import RunManager, RunRecord
from .naming import resolve_root_run_name
from .schemas import RunStatus

logger = logging.getLogger(__name__)

# Valid stream_mode values for LangGraph's graph.astream()
_VALID_LG_MODES = {"values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"}


def _build_runtime_context(
    thread_id: str,
    run_id: str,
    caller_context: Any | None,
    app_config: AppConfig | None = None,
) -> dict[str, Any]:
    """Build the dict that becomes ``ToolRuntime.context`` for the run.

    Always includes ``thread_id`` and ``run_id``. Additional keys from the caller's
    ``config['context']`` (e.g. ``agent_name`` for the bootstrap flow — issue #2677)
    are merged in but never override ``thread_id``/``run_id``. The resolved
    ``AppConfig`` is added by the worker so tools can consume it without ambient
    global lookups.

    langgraph 1.1+ surfaces this as ``runtime.context`` via the parent runtime stored
    under ``config['configurable']['__pregel_runtime']`` — see
    ``langgraph.pregel.main`` where ``parent_runtime.merge(...)`` is invoked.
    """
    # 首先放入thread_id和run_id
    runtime_ctx: dict[str, Any] = {"thread_id": thread_id, "run_id": run_id}
    # 如果传入的context是dict话，将里面的k,v都放入runtime_ctx中
    if isinstance(caller_context, dict):
        for key, value in caller_context.items():
            runtime_ctx.setdefault(key, value)
    # 并且把配置文件构建的app_config也存入
    if app_config is not None:
        runtime_ctx["app_config"] = app_config
    return runtime_ctx


@dataclass(frozen=True)
class RunContext:
    """Infrastructure dependencies for a single agent run.

    Groups checkpointer, store, and persistence-related singletons so that
    ``run_agent`` (and any future callers) receive one object instead of a
    growing list of keyword arguments.
    """

    checkpointer: Any
    store: Any | None = field(default=None)
    event_store: Any | None = field(default=None)
    run_events_config: Any | None = field(default=None)
    thread_store: Any | None = field(default=None)
    app_config: AppConfig | None = field(default=None)


def _install_runtime_context(config: dict, runtime_context: dict[str, Any]) -> None:
    existing_context = config.get("context")
    # 先判断当前的config中是否存在context
    if isinstance(existing_context, dict):
        # 如果存在的话，将context可能没有的thread_id，run_id，app_config写入
        # 因为context如果存在，说明context里面的内容已经在_build_runtime_context的时候就全部放入runtime_context中了
        existing_context.setdefault("thread_id", runtime_context["thread_id"])
        existing_context.setdefault("run_id", runtime_context["run_id"])
        if "app_config" in runtime_context:
            existing_context["app_config"] = runtime_context["app_config"]
        return
    # 如果不存在，直接将整个runtime_context当作context存入
    config["context"] = dict(runtime_context)

    # 这个方法实际的作用也就等于将thread_id，run_id，app_config写入config的context中


def _compute_agent_factory_supports_app_config(agent_factory: Any) -> bool:
    try:
        # 判断对应的工厂函数的参数里面是否有app_config
        return "app_config" in inspect.signature(agent_factory).parameters
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=128)
def _cached_agent_factory_supports_app_config(agent_factory: Any) -> bool:
    return _compute_agent_factory_supports_app_config(agent_factory)


def _agent_factory_supports_app_config(agent_factory: Any) -> bool:
    try:
        return _cached_agent_factory_supports_app_config(agent_factory)
    except TypeError:
        # Some callable instances are unhashable; fall back to a direct check.
        return _compute_agent_factory_supports_app_config(agent_factory)


async def run_agent(
    bridge: StreamBridge,
    run_manager: RunManager,
    record: RunRecord,
    *,
    ctx: RunContext,
    agent_factory: Any,
    graph_input: dict,
    config: dict,
    stream_modes: list[str] | None = None,
    stream_subgraphs: bool = False,
    interrupt_before: list[str] | Literal["*"] | None = None,
    interrupt_after: list[str] | Literal["*"] | None = None,
) -> None:
    """Execute an agent in the background, publishing events to *bridge*."""

    # Unpack infrastructure dependencies from RunContext.
    # 将RunContext里面持有的属性都拿出来
    checkpointer = ctx.checkpointer
    store = ctx.store
    event_store = ctx.event_store
    run_events_config = ctx.run_events_config
    thread_store = ctx.thread_store

    # 拿出RunRecord中的run_id和thread_id
    run_id = record.run_id
    thread_id = record.thread_id
    requested_modes: set[str] = set(stream_modes or ["values"])
    pre_run_checkpoint_id: str | None = None
    pre_run_snapshot: dict[str, Any] | None = None
    snapshot_capture_failed = False
    llm_error_fallback_message: str | None = None

    journal = None

    # Track whether "events" was requested but skipped
    # 查看stream_modes里面是否存在events这个模式，如果存在，打印日志提示不支持
    if "events" in requested_modes:
        logger.info(
            "Run %s: 'events' stream_mode not supported in gateway (requires astream_events + checkpoint callbacks). Skipping.",
            run_id,
        )

    try:
        # Initialize RunJournal + write human_message event.
        # These are inside the try block so any exception (e.g. a DB
        # error writing the event) flows through the except/finally
        # path that publishes an "end" event to the SSE bridge —
        # otherwise a failure here would leave the stream hanging
        # with no terminator.
        # RunJournal 作为 LangChain CallbackHandler 注入，负责全程记录 token 用量、LLM 调用次数、消息内容，最终写入 event_store（即 GET /runs/{id}/events 查到的数据）。
        if event_store is not None:
            from deerflow.runtime.journal import RunJournal

            journal = RunJournal(
                run_id=run_id,
                thread_id=thread_id,
                event_store=event_store,
                track_token_usage=getattr(run_events_config, "track_token_usage", True),
                progress_reporter=lambda snapshot: run_manager.update_run_progress(run_id, **snapshot),
            )

        # 1. Mark running
        # 将RunRecord的状态更新为running，并持久化
        await run_manager.set_status(run_id, RunStatus.running)

        # Snapshot the latest pre-run checkpoint so rollback can restore it.
        # 从checkpointer中查找thread_id对应的最新的snapshot
        if checkpointer is not None:
            try:
                config_for_check = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
                ckpt_tuple = await checkpointer.aget_tuple(config_for_check)
                if ckpt_tuple is not None:
                    # 获取checkpoint的configurable配置
                    ckpt_config = getattr(ckpt_tuple, "config", {}).get("configurable", {})
                    # 获取上一次运行的checkpoint_id
                    pre_run_checkpoint_id = ckpt_config.get("checkpoint_id")
                    # 获取上一次运行的checkpoint快照
                    pre_run_snapshot = {
                        "checkpoint_ns": ckpt_config.get("checkpoint_ns", ""),
                        "checkpoint": copy.deepcopy(getattr(ckpt_tuple, "checkpoint", {})),
                        "metadata": copy.deepcopy(getattr(ckpt_tuple, "metadata", {})),
                        "pending_writes": copy.deepcopy(getattr(ckpt_tuple, "pending_writes", []) or []),
                    }
            except Exception:
                snapshot_capture_failed = True
                logger.warning("Could not capture pre-run checkpoint snapshot for run %s", run_id, exc_info=True)

        # 2. Publish metadata — useStream needs both run_id AND thread_id
        # 推送metadata类型的数据给consumer，包含thread_id和run_id
        await bridge.publish(
            run_id,
            "metadata",
            {
                "run_id": run_id,
                "thread_id": thread_id,
            },
        )

        # 3. Build the agent
        from langchain_core.runnables import RunnableConfig
        from langgraph.runtime import Runtime

        # Inject runtime context so middlewares and tools (via ToolRuntime.context) can
        # access thread-level data. langgraph-cli does this automatically; we must do it
        # manually here because we drive the graph through ``agent.astream(config=...)``
        # without passing the official ``context=`` parameter.

        # 构建agent loop使用的runtime_ctx
        # 持有thread_id, run_id, config中的context(大部分来自于请求体中的context)，以及app_config
        runtime_ctx = _build_runtime_context(thread_id, run_id, config.get("context"), ctx.app_config)

        # Expose the run-scoped journal under a sentinel key so middleware can
        # write audit events (e.g. SafetyFinishReasonMiddleware recording
        # suppressed tool calls). Double-underscore prefix marks it as a
        # runtime-internal channel; user code must not depend on the key name.
        # todo 确认下这里在干嘛
        # 将journal放进runtime_ctx中，让journal在运行中能够被找到
        # 并且__开始的key表示这是一个内部变量，用户并不感知
        if journal is not None:
            runtime_ctx["__run_journal"] = journal

        # 这个方法实际的作用是将thread_id，run_id，app_config写入config的context中
        _install_runtime_context(config, runtime_ctx)
        # 将runtime_ctx和store一起构建成Runtime对象
        runtime = Runtime(context=cast(Any, runtime_ctx), store=store)
        # 设置进config的configurable中，langgraph会自动将其转换成运行时的context
        config.setdefault("configurable", {})["__pregel_runtime"] = runtime

        # Inject RunJournal as a LangChain callback handler.
        # on_llm_end captures token usage; on_chain_start/end captures lifecycle.

        # 将RunJournal注册为langchain的callback handler，用于获取token使用量，存储run_events等信息
        if journal is not None:
            config.setdefault("callbacks", []).append(journal)

        # Inject Langfuse trace-attribute metadata so the langchain CallbackHandler
        # can lift session_id / user_id / trace_name / tags onto the root trace.
        # Shared helper with ``DeerFlowClient.stream`` so both entry points stay
        # in sync; caller-provided metadata wins via setdefault inside the helper.
        inject_langfuse_metadata(
            config,
            thread_id=thread_id,
            user_id=get_effective_user_id(),
            assistant_id=record.assistant_id,
            model_name=record.model_name,
            environment=os.environ.get("DEER_FLOW_ENV") or os.environ.get("ENVIRONMENT"),
        )

        # Resolve after runtime context installation so context/configurable reflect
        # the agent name that this run will actually execute.
        # 向config里面设置run_name属性，默认为lead_agent
        config.setdefault("run_name", resolve_root_run_name(config, record.assistant_id))

        # 最终构建好要传入agent loop的RunnableConfig
        # RunnableConfig(**{
        #     # ── 来自 build_run_config ──────────────────────────
        #     "recursion_limit": 100,
        #     "run_name": "lead_agent",
        #
        #     # ── 来自 build_run_config + 多方合并 ──────────────
        #     "configurable": {
        #         "thread_id": "thread-xxx",           # build_run_config
        #         "model_name": "claude-opus-4-5",     # body.context → merge_run_context_overrides
        #         "thinking_enabled": True,            # body.context → merge_run_context_overrides
        #         "is_plan_mode": False,               # body.context → merge_run_context_overrides
        #         "subagent_enabled": True,            # body.context → merge_run_context_overrides
        #         "agent_name": "my-agent",            # body.assistant_id → build_run_config or merge_run_context_overrides
        #         "__pregel_runtime": <Runtime对象>,    # work.py
        #     },
        #
        #     # ── 来自 _install_runtime_context / body.context ──
        #     "context": {
        #         "thread_id": "thread-xxx",           # _install_runtime_context
        #         "run_id": "run-yyy",                 # _install_runtime_context
        #         "app_config": <AppConfig对象>,       # _install_runtime_context
        #         "user_id": "auth-user-id",           # inject_authenticated_user_context（服务端权威）
        #         "model_name": "claude-opus-4-5",     # body.context → merge_run_context_overrides
        #         # ... 其他 body.context 白名单字段
        #     },
        #
        #     # ── 来自 body.metadata + inject_langfuse_metadata ─
        #     "metadata": {
        #         # body.metadata 透传字段
        #         "langfuse_session_id": "thread-xxx", # inject_langfuse_metadata
        #         "langfuse_user_id": "default",       # inject_langfuse_metadata
        #         "langfuse_trace_name": "lead_agent", # inject_langfuse_metadata
        #         "langfuse_tags": ["env:prod", ...],  # inject_langfuse_metadata
        #     },
        #
        #     # ── 来自 worker.py ─────────────────────────────────
        #     "callbacks": [<RunJournal>],             # token统计 / 事件记录
        # })
        runnable_config = RunnableConfig(**config)

        # 调用agent_factory生成agent，传入构建好的RunnableConfig
        # 即make_lead_agent函数
        if ctx.app_config is not None and _agent_factory_supports_app_config(agent_factory):
            agent = agent_factory(config=runnable_config, app_config=ctx.app_config)
        else:
            agent = agent_factory(config=runnable_config)

        # Capture the effective (resolved) model name from the agent's metadata.
        # _resolve_model_name in agent.py may return the default model if the
        # requested name is not in the allowlist — this update ensures the
        # persisted model_name reflects the actual model used.

        # 如果agent解析出来的model_name不一样的话，替换RunRecord持有的model_name
        if record.model_name is not None:
            resolved = getattr(agent, "metadata", {}) or {}
            if isinstance(resolved, dict):
                effective = resolved.get("model_name")
                if effective and effective != record.model_name:
                    await run_manager.update_model_name(record.run_id, effective)

        # 4. Attach checkpointer and store
        # 将checkpointer和 store都传入agent中
        if checkpointer is not None:
            agent.checkpointer = checkpointer
        if store is not None:
            agent.store = store

        # 5. Set interrupt nodes
        # 设置interrupt nodes
        if interrupt_before:
            agent.interrupt_before_nodes = interrupt_before
        if interrupt_after:
            agent.interrupt_after_nodes = interrupt_after

        # 6. Build LangGraph stream_mode list
        #    "events" is NOT a valid astream mode — skip it
        #    "messages-tuple" maps to LangGraph's "messages" mode
        # 构建langgraph使用的stream_modes
        lg_modes: list[str] = []
        for m in requested_modes:
            # 如果是messages-tuple，映射为messages
            if m == "messages-tuple":
                lg_modes.append("messages")
            # events不支持，跳过
            elif m == "events":
                # Skipped — see log above
                continue
            # 其他合法的modes，直接添加进去
            elif m in _VALID_LG_MODES:
                lg_modes.append(m)
        # 如果上一步完成后仍是空集合，默认使用values
        if not lg_modes:
            lg_modes = ["values"]

        # Deduplicate while preserving order
        # 对lg_modes去重
        seen: set[str] = set()
        deduped: list[str] = []
        for m in lg_modes:
            if m not in seen:
                seen.add(m)
                deduped.append(m)
        lg_modes = deduped

        logger.info("Run %s: streaming with modes %s (requested: %s)", run_id, lg_modes, requested_modes)
        # 在这里说明下不同mode下，graph返回的chunk的区别
        # values：是在每次pregel的superstep后返回的，对应就是整个AgentState的快照

        # updates：是在每个节点执行完成之后就emit，不会等到superstep执行完成，chunk里面的key是节点名，value是本次节点执行完成要更新的AgentState中的内容

        # messages: 在llm每生成一个token的时候都emit，结构是一个二元组（AIMessageChunk, metadata）
        # # 源码：_messages.py _emit()
        # (
        #     AIMessageChunk(content="你好", id="run-xxx"),  # LangChain 消息对象
        #     {
        #         "langgraph_step": 1,
        #         "langgraph_node": "chatbot",
        #         "langgraph_triggers": ["start"],
        #         "langgraph_checkpoint_ns": "chatbot:task-id",
        #         "ls_model_name": "gpt-4o",
        #         # ...其他 LangChain 元数据
        #     }
        # )

        # tasks：任务开始和任务结束时各emit一次，每执行一个节点都被抽象为一个task
        # # 任务开始时（源码：debug.py map_debug_tasks()）
        # {
        #     "id": "abc123",           # task_id
        #     "name": "chatbot",        # 节点名
        #     "input": {...},           # 节点输入
        #     "triggers": ["messages"], # 触发该任务的 channel 名
        #     "metadata": {...}         # 可选，用户元数据
        # }
        #
        # # 任务结束时（源码：debug.py map_debug_task_results()）
        # {
        #     "id": "abc123",
        #     "name": "chatbot",
        #     "error": None,            # 或异常信息字符串
        #     "result": {               # 节点写的内容（只含 stream_keys 里的 channel）
        #         "messages": [...]
        #     },
        #     "interrupts": []          # 若触发了 interrupt，这里有 Interrupt 对象序列化结果
        # }

        # checkpoints：每次checkpoint保存的时候emit，内容等同于get_state()方法

        # debug：tasks和checkpoints的包装版，也就是在外面套了一层
        # # tasks 事件包装成：
        # {
        #     "step": 1,
        #     "timestamp": "2026-07-23T10:00:00+00:00",
        #     "type": "task/checkpoint", # 或 "task_result"
        #     "payload": { ... }         # 同 tasks/checkpoint 模式的内容
        # }

        # custom：节点内部主动调用StreamWriter时emit，可以是任意结构

        # 7. Stream using graph.astream
        # 如果stream_mode只有一个，且不存在stream子graph
        if len(lg_modes) == 1 and not stream_subgraphs:
            # Single mode, no subgraphs: astream yields raw chunks
            single_mode = lg_modes[0]
            async for chunk in agent.astream(graph_input, config=runnable_config, stream_mode=single_mode):
                # 如果RunRecord被其他run设置了信号旗，退出循环，停止loop
                if record.abort_event.is_set():
                    logger.info("Run %s abort requested — stopping", run_id)
                    break
                # 尝试从chunk中提出llm的错误降级信息
                llm_error_fallback_message = llm_error_fallback_message or _extract_llm_error_fallback_message(chunk)
                # 将langgraph的stream_mode替换成http event-stream的event类型。
                # 直接映射就好，因为sse_event可以自定义为任何类型，http协议并没有做任何限制
                sse_event = _lg_mode_to_sse_event(single_mode)
                # 将chunk的内容序列化之后push到bridge中，然后由bridge给sse消费，传输给前端
                await bridge.publish(run_id, sse_event, serialize(chunk, mode=single_mode))
        else:
            # 当使用多个stream_mode的时候，agent返回的是一个二元组（mode, data）
            # Multiple modes or subgraphs: astream yields tuples
            async for item in agent.astream(
                graph_input,
                config=runnable_config,
                stream_mode=lg_modes,
                subgraphs=stream_subgraphs,
            ):
                # 同样的，如果RunRecord被后来的run给中断了或回滚了，直接跳出循环
                if record.abort_event.is_set():
                    logger.info("Run %s abort requested — stopping", run_id)
                    break
                # 将item解析为mode 和 chunk的结构
                mode, chunk = _unpack_stream_item(item, lg_modes, stream_subgraphs)
                # 如果mode不存在，继续循环
                if mode is None:
                    continue

                # 后续流程和前面的分支一致
                llm_error_fallback_message = llm_error_fallback_message or _extract_llm_error_fallback_message(chunk)
                sse_event = _lg_mode_to_sse_event(mode)
                await bridge.publish(run_id, sse_event, serialize(chunk, mode=mode))

        # 8. Final status
        # 如果判断出当前的RunRecord的abort_event被设置了，即被后续的run给中断了
        if record.abort_event.is_set():
            # 获取它的abort_action来决定后续的操作
            # abort_action是被中断它的那次run设置的，详见RunManager的create_or_reject方法
            action = record.abort_action
            # 如果操作是rollback的话
            if action == "rollback":
                # 将RunRecord的内存和db状态都更新为error，并标注原因
                await run_manager.set_status(run_id, RunStatus.error, error="Rolled back by user")
                try:
                    # 回滚到上一个checkpoint的状态
                    await _rollback_to_pre_run_checkpoint(
                        checkpointer=checkpointer,
                        thread_id=thread_id,
                        run_id=run_id,
                        pre_run_checkpoint_id=pre_run_checkpoint_id,
                        pre_run_snapshot=pre_run_snapshot,
                        snapshot_capture_failed=snapshot_capture_failed,
                    )
                    logger.info("Run %s rolled back to pre-run checkpoint %s", run_id, pre_run_checkpoint_id)
                except Exception:
                    logger.warning("Failed to rollback checkpoint for run %s", run_id, exc_info=True)
            else:
                # 如果操作是interrupt的话，更新对应RunRecord在RunManager中的内存状态和db状态
                # 实际在create_or_reject方法里面就已经更新了，这里可能是为了兜底吧
                await run_manager.set_status(run_id, RunStatus.interrupted)
        # 如果存在llm错误兜底信息 或者journal里面存在llm错误兜底信息
        elif llm_error_fallback_message or (journal is not None and journal.had_llm_error_fallback):
            error_msg = llm_error_fallback_message
            if error_msg is None and journal is not None:
                error_msg = journal.llm_error_fallback_message
            error_msg = error_msg or "LLM provider failed after retries"
            # 将错误兜底信息获取出来，更新RunRecord在RunManager中的内存状态，并持久化到db
            await run_manager.set_status(run_id, RunStatus.error, error=error_msg)
        else:
            # 其他情况将RunRecord状态更新为success并持久化到db
            await run_manager.set_status(run_id, RunStatus.success)

    except asyncio.CancelledError:
        action = record.abort_action
        if action == "rollback":
            await run_manager.set_status(run_id, RunStatus.error, error="Rolled back by user")
            try:
                await _rollback_to_pre_run_checkpoint(
                    checkpointer=checkpointer,
                    thread_id=thread_id,
                    run_id=run_id,
                    pre_run_checkpoint_id=pre_run_checkpoint_id,
                    pre_run_snapshot=pre_run_snapshot,
                    snapshot_capture_failed=snapshot_capture_failed,
                )
                logger.info("Run %s was cancelled and rolled back", run_id)
            except Exception:
                logger.warning("Run %s cancellation rollback failed", run_id, exc_info=True)
        else:
            await run_manager.set_status(run_id, RunStatus.interrupted)
            logger.info("Run %s was cancelled", run_id)

    except Exception as exc:
        error_msg = f"{exc}"
        logger.exception("Run %s failed: %s", run_id, error_msg)
        # 如果出现了异常，更新RunRecord状态为error
        await run_manager.set_status(run_id, RunStatus.error, error=error_msg)
        # 并且向stream_bridge publish异常信息
        await bridge.publish(
            run_id,
            "error",
            {
                "message": error_msg,
                "name": type(exc).__name__,
            },
        )

    finally:
        # Flush any buffered journal events and persist completion data
        # 把 journal 里缓冲的事件全部刷入 event_store
        if journal is not None:
            try:
                await journal.flush()
            except Exception:
                logger.warning("Failed to flush journal for run %s", run_id, exc_info=True)

            try:
                # Persist token usage + convenience fields to RunStore
                # 获取journal里面统计的完整的token等信息 llm_invoke_count msg_count first_human_msg last_ai_msg等信息
                completion = journal.get_completion_data()
                # 将上一步获取到的journal中统计的信息更新到RunRecord中
                await run_manager.update_run_completion(run_id, status=record.status.value, **completion)
            except Exception:
                logger.warning("Failed to persist run completion for %s (non-fatal)", run_id, exc_info=True)

        # Sync title from checkpoint to threads_meta.display_name
        # 从 checkpoint 读取 AI 生成的标题，写入 thread_store.display_name
        if checkpointer is not None and thread_store is not None:
            try:
                ckpt_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
                ckpt_tuple = await checkpointer.aget_tuple(ckpt_config)
                # 从checkpoint里面获取到State里面的title字段，作为整个thread的标题维护进thread里面，这个就是前端看到的每个会话的标题
                if ckpt_tuple is not None:
                    ckpt = getattr(ckpt_tuple, "checkpoint", {}) or {}
                    title = ckpt.get("channel_values", {}).get("title")
                    if title:
                        await thread_store.update_display_name(thread_id, title)
            except Exception:
                logger.debug("Failed to sync title for thread %s (non-fatal)", thread_id)

        # Update threads_meta status based on run outcome
        # 更新 thread 状态（running → idle）
        if thread_store is not None:
            try:
                # 如果RunRecord的最终状态是success，更新thread的状态为idle，否则，将RunRecord的状态更新到thread中
                final_status = "idle" if record.status == RunStatus.success else record.status.value
                await thread_store.update_status(thread_id, final_status)
            except Exception:
                logger.debug("Failed to update thread_meta status for %s (non-fatal)", thread_id)

        # 推送 END 事件，通知前端流结束
        await bridge.publish_end(run_id)
        # 60 秒后清理 bridge 里这个 run 的队列
        asyncio.create_task(bridge.cleanup(run_id, delay=60))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _call_checkpointer_method(checkpointer: Any, async_name: str, sync_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call a checkpointer method, supporting async and sync variants."""
    method = getattr(checkpointer, async_name, None) or getattr(checkpointer, sync_name, None)
    if method is None:
        raise AttributeError(f"Missing checkpointer method: {async_name}/{sync_name}")
    result = method(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def _rollback_to_pre_run_checkpoint(
    *,
    checkpointer: Any,
    thread_id: str,
    run_id: str,
    pre_run_checkpoint_id: str | None,
    pre_run_snapshot: dict[str, Any] | None,
    snapshot_capture_failed: bool,
) -> None:
    """Restore thread state to the checkpoint snapshot captured before run start."""
    # 如果checkpointer不存在的话，打印日志，直接返回
    if checkpointer is None:
        logger.info("Run %s rollback requested but no checkpointer is configured", run_id)
        return
    # 如果前面获取checkpoint的快照失败了，也打印日志，直接返回
    if snapshot_capture_failed:
        logger.warning("Run %s rollback skipped: pre-run checkpoint snapshot capture failed", run_id)
        return
    # 如果上一次checkpoint的快照不存在，调用checkpointer的删除方法，将thread_id对应的快照清空后返回
    if pre_run_snapshot is None:
        await _call_checkpointer_method(checkpointer, "adelete_thread", "delete_thread", thread_id)
        logger.info("Run %s rollback reset thread %s to empty state", run_id, thread_id)
        return

    checkpoint_to_restore = None
    metadata_to_restore: dict[str, Any] = {}
    checkpoint_ns = ""
    # 从之前获取的快照中获取checkpoint的值
    checkpoint = pre_run_snapshot.get("checkpoint")
    # 如果checkpoint不是dict的话，返回
    if not isinstance(checkpoint, dict):
        logger.warning("Run %s rollback skipped: invalid pre-run checkpoint snapshot", run_id)
        return
    checkpoint_to_restore = checkpoint
    # 如果保存的快照里面没有id，但存在pre_run_checkpoint_id，将其赋值给快照
    if checkpoint_to_restore.get("id") is None and pre_run_checkpoint_id is not None:
        checkpoint_to_restore = {**checkpoint_to_restore, "id": pre_run_checkpoint_id}
    # 如果快照里面仍没有id，返回
    if checkpoint_to_restore.get("id") is None:
        logger.warning("Run %s rollback skipped: pre-run checkpoint has no checkpoint id", run_id)
        return
    # 创建一个新的checkpoint_maker，即id 和 ts
    restore_marker = _new_checkpoint_marker()
    # 将其更新到restore快照中
    checkpoint_to_restore = {
        **checkpoint_to_restore,
        "id": restore_marker["id"],
        "ts": restore_marker["ts"],
    }
    # 从快照中获取metadata和namespace、channel_versions
    metadata = pre_run_snapshot.get("metadata", {})
    metadata_to_restore = metadata if isinstance(metadata, dict) else {}
    raw_checkpoint_ns = pre_run_snapshot.get("checkpoint_ns")
    checkpoint_ns = raw_checkpoint_ns if isinstance(raw_checkpoint_ns, str) else ""

    channel_versions = checkpoint_to_restore.get("channel_versions")
    new_versions = dict(channel_versions) if isinstance(channel_versions, dict) else {}

    restore_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}
    # 保存新的checkpoint
    restored_config = await _call_checkpointer_method(
        checkpointer,
        "aput",
        "put",
        restore_config,
        checkpoint_to_restore,
        metadata_to_restore if isinstance(metadata_to_restore, dict) else {},
        new_versions,
    )
    if not isinstance(restored_config, dict):
        raise RuntimeError(f"Run {run_id} rollback restore returned invalid config: expected dict")
    restored_configurable = restored_config.get("configurable", {})
    if not isinstance(restored_configurable, dict):
        raise RuntimeError(f"Run {run_id} rollback restore returned invalid config payload")
    restored_checkpoint_id = restored_configurable.get("checkpoint_id")
    if not restored_checkpoint_id:
        raise RuntimeError(f"Run {run_id} rollback restore did not return checkpoint_id")

    # 将上一次checkpoint的pending_writes也写入到回滚后的新的checkpoint中
    pending_writes = pre_run_snapshot.get("pending_writes", [])
    if not pending_writes:
        return

    writes_by_task: dict[str, list[tuple[str, Any]]] = {}
    for item in pending_writes:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise RuntimeError(f"Run {run_id} rollback failed: pending_write is not a 3-tuple: {item!r}")
        task_id, channel, value = item
        if not isinstance(channel, str):
            raise RuntimeError(f"Run {run_id} rollback failed: pending_write has non-string channel: task_id={task_id!r}, channel={channel!r}")
        writes_by_task.setdefault(str(task_id), []).append((channel, value))

    for task_id, writes in writes_by_task.items():
        await _call_checkpointer_method(
            checkpointer,
            "aput_writes",
            "put_writes",
            restored_config,
            writes,
            task_id=task_id,
        )


def _new_checkpoint_marker() -> dict[str, str]:
    marker = empty_checkpoint()
    return {"id": marker["id"], "ts": marker["ts"]}


def _lg_mode_to_sse_event(mode: str) -> str:
    """Map LangGraph internal stream_mode name to SSE event name.

    LangGraph's ``astream(stream_mode="messages")`` produces message
    tuples.  The SSE protocol calls this ``messages-tuple`` when the
    client explicitly requests it, but the default SSE event name used
    by LangGraph Platform is simply ``"messages"``.
    """
    # All LG modes map 1:1 to SSE event names — "messages" stays "messages"
    return mode


def _error_fallback_message_from_metadata(metadata: dict[str, Any], content: Any) -> str:
    # 如果存在error_detail，返回
    detail = metadata.get("error_detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    # 如果存在error_reason，返回
    reason = metadata.get("error_reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    # 如果content是str，截取前2000个字符返回
    if isinstance(content, str) and content.strip():
        return content.strip()[:2000]
    # 最后返回兜底信息
    return "LLM provider failed after retries"


def _try_extract_from_message(obj: Any) -> str | None:
    """Try to extract fallback marker from a single message object or dict."""
    # 获取消息的additional_kwargs
    additional_kwargs = getattr(obj, "additional_kwargs", None)
    # 如果附加属性里面存在deerflow_error_fallback=true
    if isinstance(additional_kwargs, dict) and additional_kwargs.get("deerflow_error_fallback"):
        # 解析消息中的元数据和content，返回错误兜底信息
        return _error_fallback_message_from_metadata(additional_kwargs, getattr(obj, "content", None))

    # 如果msg是一个dict，逻辑不变
    if isinstance(obj, dict):
        nested_kwargs = obj.get("additional_kwargs")
        if isinstance(nested_kwargs, dict) and nested_kwargs.get("deerflow_error_fallback"):
            return _error_fallback_message_from_metadata(nested_kwargs, obj.get("content"))
    return None


def _extract_llm_error_fallback_message(value: Any) -> str | None:
    """Find LLM fallback markers in streamed LangGraph chunks.

    Error fallback messages returned by model-call middleware are not guaranteed
    to pass through LLM end callbacks, but they do appear in graph state chunks.
    """
    # Fast path: large state chunks produced by stream_mode="values" have a
    # top-level "messages" list. Scanning only that list avoids expensive deep
    # recursion into large state dicts.
    # 如果stream_mode是values的情况，入参传入的value就是整个AgentState的快照，所以肯定有messages这个属性
    if isinstance(value, dict):
        messages = value.get("messages")
        # 如果messages是list或者tuple类型的
        if isinstance(messages, (list, tuple)):
            # 遍历messages，逐个解析消息
            for msg in messages:
                # 根据message中的additional_kwargs和metadata以及content等信息提取出错误降低信息
                result = _try_extract_from_message(msg)
                # 只要有一个结果被提取出来，直接返回
                if result is not None:
                    return result
            # Fallback marker is attached to an AI message in the messages
            # channel; it will never appear elsewhere in a values chunk.
            # 降低标志被当作一个AI message添加到messages列表中，它不会出现在其他地方，对于values类型的chunk来说
            return None
        # No top-level "messages" — this is likely an "updates" chunk (small
        # dict keyed by node name). Fall through to deep walk, which is cheap
        # for these payloads.
        # 如果顶层没有messages这个属性，那么有可能是updates类型的chunk，是以node_name为key的，走后面的deep walk

    # Deep walk for updates / messages / tuple / list modes. Payloads are
    # small, so full recursion is acceptable here.
    # 其他模式进行deep walk，其他模式每个chunk的payload比较小，所以全量递归是可以接受的
    seen: set[int] = set()

    def walk(obj: Any) -> str | None:
        # 先判断该对象有没有被查看过，如果有，直接返回，如果没有，添加进set中
        oid = id(obj)
        if oid in seen:
            return None
        seen.add(oid)

        # 解析错误兜底信息，如果不为None，返回
        result = _try_extract_from_message(obj)
        if result is not None:
            return result

        # 如果obj是dict类型的，遍历它values中的所有元素，递归walk
        if isinstance(obj, dict):
            for item in obj.values():
                result = walk(item)
                if result is not None:
                    return result
            return None

        # 如果obj是集合类型的，遍历所有元素，递归walk
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                result = walk(item)
                if result is not None:
                    return result
        return None

    # 调用walk来进行解析
    return walk(value)


def _extract_human_message(graph_input: dict) -> HumanMessage | None:
    """Extract or construct a HumanMessage from graph_input for event recording.

    Returns a LangChain HumanMessage so callers can use .model_dump() to get
    the checkpoint-aligned serialization format.
    """
    from langchain_core.messages import HumanMessage

    messages = graph_input.get("messages")
    if not messages:
        return None
    last = messages[-1] if isinstance(messages, list) else messages
    if isinstance(last, HumanMessage):
        return last
    if isinstance(last, str):
        return HumanMessage(content=last) if last else None
    if hasattr(last, "content"):
        content = last.content
        return HumanMessage(content=content)
    if isinstance(last, dict):
        content = last.get("content", "")
        return HumanMessage(content=content) if content else None
    return None


def _unpack_stream_item(
    item: Any,
    lg_modes: list[str],
    stream_subgraphs: bool,
) -> tuple[str | None, Any]:
    """Unpack a multi-mode or subgraph stream item into (mode, chunk).

    Returns ``(None, None)`` if the item cannot be parsed.
    """
    if stream_subgraphs:
        if isinstance(item, tuple) and len(item) == 3:
            _ns, mode, chunk = item
            return str(mode), chunk
        if isinstance(item, tuple) and len(item) == 2:
            mode, chunk = item
            return str(mode), chunk
        return None, None

    if isinstance(item, tuple) and len(item) == 2:
        mode, chunk = item
        return str(mode), chunk

    # Fallback: single-element output from first mode
    return lg_modes[0] if lg_modes else None, item

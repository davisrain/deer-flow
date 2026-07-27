"""Run event capture via LangChain callbacks.

RunJournal sits between LangChain's callback mechanism and the pluggable
RunEventStore. It standardizes callback data into RunEvent records and
handles token usage accumulation.

Key design decisions:
- on_llm_new_token is NOT implemented -- only complete messages via on_llm_end
- on_chat_model_start captures structured prompts as llm_request (OpenAI format) and
  extracts the first human message for run.input, because it is more reliable than
  on_chain_start (fires on every node) — messages here are fully structured.
- on_chain_start with parent_run_id=None emits a run.start trace marking root invocation.
- on_llm_end emits llm_response in OpenAI Chat Completions format
- Token usage accumulated in memory, written to RunRow on run completion
- Caller identification via tags injection (lead_agent / subagent:{name} / middleware:{name})
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Command

if TYPE_CHECKING:
    from deerflow.runtime.events.store.base import RunEventStore

logger = logging.getLogger(__name__)


class RunJournal(BaseCallbackHandler):
    """LangChain callback handler that captures events to RunEventStore."""

    def __init__(
        self,
        run_id: str,
        thread_id: str,
        event_store: RunEventStore,
        *,
        track_token_usage: bool = True,
        flush_threshold: int = 20,
        progress_reporter: Callable[[dict], Awaitable[None]] | None = None,
        progress_flush_interval: float = 5.0,
    ):
        super().__init__()
        self.run_id = run_id
        self.thread_id = thread_id
        self._store = event_store
        self._track_tokens = track_token_usage
        # buffer里面保存的数量阈值，达到阈值后会刷入event_store
        self._flush_threshold = flush_threshold
        self._progress_reporter = progress_reporter
        # flush progress的间隔时间，单位是秒
        self._progress_flush_interval = progress_flush_interval

        # Write buffer
        # 用于保存run_event的列表，当达到阈值之后，刷入到event_store中
        self._buffer: list[dict] = []
        self._pending_flush_tasks: set[asyncio.Task[None]] = set()
        #
        self._pending_progress_task: asyncio.Task[None] | None = None
        self._pending_progress_delayed = False
        self._progress_dirty = False
        # 记录最后一次flush progress的时间
        self._last_progress_flush = 0.0

        # Token accumulators
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_tokens = 0
        self._llm_call_count = 0

        # Caller-bucketed token accumulators
        self._lead_agent_tokens = 0
        self._subagent_tokens = 0
        self._middleware_tokens = 0

        # Dedup: LangChain may fire on_llm_end multiple times for the same run_id
        # 用于记录run_id是否已经统计过token了
        self._counted_llm_run_ids: set[str] = set()
        self._counted_external_source_ids: set[str] = set()
        # 用于记录run_id是否已经统计过消息数量了
        self._counted_message_llm_run_ids: set[str] = set()

        # Convenience fields
        self._last_ai_msg: str | None = None
        self._first_human_msg: str | None = None
        # 统计总共的消息数量
        self._msg_count = 0
        # 统计是否存在llm错误兜底信息
        self._had_llm_error_fallback = False
        # llm的错误兜底信息内容
        self._llm_error_fallback_message: str | None = None

        # Latency tracking
        # run_id对应的开始时间，在on_chat_model_start里面会设置，在on_llm_end的时候会删除
        self._llm_start_times: dict[str, float] = {}  # langchain run_id -> start time

        # LLM request/response tracking
        self._llm_call_index = 0
        self._seen_llm_starts: set[str] = set()  # langchain run_ids that fired on_chat_model_start

    # -- Lifecycle callbacks --

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        """Extract displayable text from a message's mixed content shape."""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, Mapping):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    else:
                        nested = block.get("content")
                        if isinstance(nested, str):
                            parts.append(nested)
            return "".join(parts)
        if isinstance(content, Mapping):
            for key in ("text", "content"):
                value = content.get(key)
                if isinstance(value, str):
                    return value

        text = getattr(message, "text", None)
        if isinstance(text, str):
            return text
        return ""

    def _record_message_summary(self, message: BaseMessage, *, caller: str | None = None) -> None:
        """Update run-level convenience fields for persisted run rows."""
        # 将msg数量+1
        self._msg_count += 1

        # ``last_ai_message`` should represent the lead agent's user-facing
        # answer. Middleware/subagent model calls and empty tool-call-only
        # AI messages must not overwrite the last useful assistant text.
        is_ai_message = isinstance(message, AIMessage) or getattr(message, "type", None) == "ai"
        # 判断消息是否是AIMessage且caller是lead_agent的
        if is_ai_message and (caller is None or caller == "lead_agent"):
            # 维护最后一个ai_msg的字段为message对应的content
            text = self._message_text(message).strip()
            if text:
                self._last_ai_msg = text[:2000]

    def on_chain_start(
        self,
        # chain/节点的元信息序列化结果
        serialized: dict[str, Any],
        # 节点收到的原始入参
        inputs: dict[str, Any],
        *,
        # 本次执行的唯一id，关联同一次调用的on_chain_start和on_chain_end
        run_id: UUID,
        # 父调用的id
        parent_run_id: UUID | None = None,
        # 来自于RunnableConfig中的tags
        tags: list[str] | None = None,
        # 来自于RunnableConfig中的metadata
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # 根据tags定位出caller
        caller = self._identify_caller(tags)
        # 如果没有parent_run_id的话，说明是graph的根调用
        if parent_run_id is None:
            # Root graph invocation — emit a single trace event for the run start.
            # 获取chain_name，在graph中就对应节点名称
            chain_name = (serialized or {}).get("name", "unknown")
            # 保存一个run.start的event
            self._put(
                event_type="run.start",
                category="trace",
                content={"chain": chain_name},
                metadata={"caller": caller, **(metadata or {})},
            )

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        # 保存run.end类型的event
        self._put(event_type="run.end", category="outputs", content=outputs, metadata={"status": "success"})
        # flush
        self._flush_sync()

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._put(
            event_type="run.error",
            category="error",
            content=str(error),
            metadata={"error_type": type(error).__name__},
        )
        self._flush_sync()

    # -- LLM callbacks --

    def on_chat_model_start(
        self,
        serialized: dict,
        # 这里是双层list的原因是llm支持batch方法批量调用
        # llm.batch([[HumanMessage("你好")],[HumanMessage("天气怎么样"), AIMessage("今天晴"), HumanMessage("明天呢")]])
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture structured prompt messages for llm_request event.

        This is also the canonical place to extract the first human message:
        messages are fully structured here, it fires only on real LLM calls,
        and the content is never compressed by checkpoint trimming.
        """
        rid = str(run_id)
        # 记录run_id对应的llm开始时间
        self._llm_start_times[rid] = time.monotonic()
        # 统计当前的llm调用的index
        self._llm_call_index += 1
        # 统计哪些run_id调用了llm_start
        self._seen_llm_starts.add(rid)

        logger.debug(
            "on_chat_model_start %s: tags=%s num_batches=%d message_counts=%s",
            run_id,
            tags,
            len(messages),
            [len(batch) for batch in messages],
        )

        # Capture the first human message sent to any LLM in this run.
        # 获取传递的给llm的第一个HumanMessage
        if not self._first_human_msg and messages:
            for batch in reversed(messages):
                for m in reversed(batch):
                    # 如果是HumanMessage且不是summary
                    if isinstance(m, HumanMessage) and m.name != "summary":
                        # 保存第一个HumanMessage的text
                        caller = self._identify_caller(tags)
                        self.set_first_human_message(m.text)
                        # 保存llm.human.input的event
                        self._put(
                            event_type="llm.human.input",
                            category="message",
                            content=m.model_dump(),
                            metadata={"caller": caller},
                        )
                        # todo 这个方法没太看懂要干嘛，感觉只是统计下消息数量
                        self._record_message_summary(m, caller=caller)
                        break
                # 如果找到了第一个HumanMessage，跳出循环
                if self._first_human_msg:
                    break

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id: UUID, parent_run_id: UUID | None = None, tags: list[str] | None = None, metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        # Fallback: on_chat_model_start is preferred. This just tracks latency.
        self._llm_start_times[str(run_id)] = time.monotonic()

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        messages: list[AnyMessage] = []
        logger.debug("on_llm_end %s: tags=%s", run_id, tags)
        # 解析llm的返回结果，收集对应的message信息到messages集合
        for generation in response.generations:
            for gen in generation:
                if hasattr(gen, "message"):
                    messages.append(gen.message)
                else:
                    logger.warning(f"on_llm_end {run_id}: generation has no message attribute: {gen}")

        # 遍历messages
        for message in messages:
            caller = self._identify_caller(tags)

            # Latency
            rid = str(run_id)
            # 获取rid之前保存的llm开始时间
            start = self._llm_start_times.pop(rid, None)
            # 计算llm的耗时
            latency_ms = int((time.monotonic() - start) * 1000) if start else None

            # Token usage from message
            # 从message中获取usage_metadata信息
            usage = getattr(message, "usage_metadata", None)
            usage_dict = dict(usage) if usage else {}
            additional_kwargs = getattr(message, "additional_kwargs", None) or {}
            # 如果additional_kwargs存在deerflow_error_fallback
            if isinstance(additional_kwargs, dict) and additional_kwargs.get("deerflow_error_fallback"):
                # 设置存在错误兜底信息的标志为true
                self._had_llm_error_fallback = True
                # 尝试从additional_kwargs里面获取error信息
                detail = additional_kwargs.get("error_detail")
                reason = additional_kwargs.get("error_reason")
                # 获取message的里面的content内容作为兜底的error信息
                fallback_text = self._message_text(message).strip()
                # 按照detail > reason > text的优先级获取错误兜底信息
                if isinstance(detail, str) and detail.strip():
                    self._llm_error_fallback_message = detail.strip()
                elif isinstance(reason, str) and reason.strip():
                    self._llm_error_fallback_message = reason.strip()
                elif fallback_text:
                    self._llm_error_fallback_message = fallback_text[:2000]

            # Resolve call index
            call_index = self._llm_call_index
            # 判断rid是否在_seen_llm_starts中，在on_chat_model_start的时候会将rid放进去
            if rid not in self._seen_llm_starts:
                # Fallback: on_chat_model_start was not called
                # 兜底，on_chat_model_start方法没有被调用
                # 执行on_chat_model_start中应该执行的逻辑
                self._llm_call_index += 1
                call_index = self._llm_call_index
                self._seen_llm_starts.add(rid)

            # Trace event: llm_response (OpenAI completion format)
            # 保存llm.ai.response类型的event
            self._put(
                event_type="llm.ai.response",
                category="message",
                content=message.model_dump(),
                metadata={
                    "caller": caller,
                    "usage": usage_dict,
                    "latency_ms": latency_ms,
                    "llm_call_index": call_index,
                },
            )

            # 如果rid不在_counted_message_llm_run_ids中，说明该run_id的on_llm_end还没有被调用过
            if rid not in self._counted_message_llm_run_ids:
                # 记录message的相关信息
                # 作用就是维护消息数量 和 _last_ai_msg字段
                self._record_message_summary(message, caller=caller)

            # Token accumulation (dedup by langchain run_id to avoid double-counting
            # when the callback fires more than once for the same response)
            # 如果_track_tokens开关是true，这个是根据配置文件中run_event模块的配置来的，默认是true
            if self._track_tokens:
                # 获取input_tokens output_tokens total_tokens
                input_tk = usage_dict.get("input_tokens", 0) or 0
                output_tk = usage_dict.get("output_tokens", 0) or 0
                total_tk = usage_dict.get("total_tokens", 0) or 0
                if total_tk == 0:
                    total_tk = input_tk + output_tk
                # 如果token使用量大于0 且 该run_id还没有统计过
                if total_tk > 0 and rid not in self._counted_llm_run_ids:
                    self._counted_llm_run_ids.add(rid)
                    # 将该消息的token使用量累加到journal中
                    self._total_input_tokens += input_tk
                    self._total_output_tokens += output_tk
                    self._total_tokens += total_tk
                    # 将llm的调用次数+1
                    self._llm_call_count += 1

                    # 如果caller是以不同的前缀开头的，将token使用量累加进journal对应的字段中
                    if caller.startswith("subagent:"):
                        self._subagent_tokens += total_tk
                    elif caller.startswith("middleware:"):
                        self._middleware_tokens += total_tk
                    else:
                        self._lead_agent_tokens += total_tk
                    # 当更新了token使用量之后，定时调用进度flush方法
                    self._schedule_progress_flush()

        # 如果消息存在，将run_id维护进_counted_message_llm_run_ids中，表示已经统计过消息数量了
        if messages:
            self._counted_message_llm_run_ids.add(str(run_id))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        # 将run_id从_llm_start_times删除
        self._llm_start_times.pop(str(run_id), None)
        # 保存一个llm.error类型的event
        self._put(event_type="llm.error", category="trace", content=str(error))

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, inputs=None, **kwargs):
        """Handle tool start event, cache tool call ID for later correlation"""
        # run_id就是对应的tool_call_id
        tool_call_id = str(run_id)
        logger.debug("Tool start for node %s, tool_call_id=%s, tags=%s", run_id, tool_call_id, tags)

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        """Handle tool end event, append message and clear node data"""
        try:
            # 如果output是ToolMessage类型的
            if isinstance(output, ToolMessage):
                msg = cast(ToolMessage, output)
                # 保存llm.tool.result类型的event
                self._put(event_type="llm.tool.result", category="message", content=msg.model_dump())
                # 统计消息数量
                self._record_message_summary(msg)
            elif isinstance(output, Command):
                # 如果返回的是Command类型的
                cmd = cast(Command, output)
                # 获取要更新的消息列表
                messages = cmd.update.get("messages", [])
                # 遍历消息
                for message in messages:
                    # 如果消息是BaseMessage类型的，构建llm.tool.result类型的event并报错
                    if isinstance(message, BaseMessage):
                        self._put(event_type="llm.tool.result", category="message", content=message.model_dump())
                        # 统计消息数量
                        self._record_message_summary(message)
                    else:
                        logger.warning(f"on_tool_end {run_id}: command update message is not BaseMessage: {type(message)}")
            else:
                logger.warning(f"on_tool_end {run_id}: output is not ToolMessage: {type(output)}")
        finally:
            logger.debug("Tool end for node %s", run_id)

    # -- Internal methods --

    def _put(self, *, event_type: str, category: str, content: str | dict = "", metadata: dict | None = None) -> None:
        # 向_buffer里面添加run_event的信息
        self._buffer.append(
            {
                "thread_id": self.thread_id,
                "run_id": self.run_id,
                "event_type": event_type,
                "category": category,
                "content": content,
                "metadata": metadata or {},
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        # 如果_buffer里面元素超过阈值了，同步flush到store中
        if len(self._buffer) >= self._flush_threshold:
            self._flush_sync()

    def _flush_sync(self) -> None:
        """Best-effort flush of buffer to RunEventStore.

        BaseCallbackHandler methods are synchronous.  If an event loop is
        running we schedule an async ``put_batch``; otherwise the events
        stay in the buffer and are flushed later by the async ``flush()``
        call in the worker's ``finally`` block.
        """
        if not self._buffer:
            return
        # Skip if a flush is already in flight — avoids concurrent writes
        # to the same SQLite file from multiple fire-and-forget tasks.
        # 如果当前已经有flush任务了，直接返回
        if self._pending_flush_tasks:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop — keep events in buffer for later async flush.
            return
        batch = self._buffer.copy()
        self._buffer.clear()
        # 向event_loop中创建flush任务
        task = loop.create_task(self._flush_async(batch))
        self._pending_flush_tasks.add(task)
        # 向task中注册完成后的callback
        # 用于清理_pending_flush_tasks中的task
        task.add_done_callback(self._on_flush_done)

    async def _flush_async(self, batch: list[dict]) -> None:
        try:
            await self._store.put_batch(batch)
        except Exception:
            logger.warning(
                "Failed to flush %d events for run %s — returning to buffer",
                len(batch),
                self.run_id,
                exc_info=True,
            )
            # Return failed events to buffer for retry on next flush
            self._buffer = batch + self._buffer

    def _on_flush_done(self, task: asyncio.Task) -> None:
        # 将task从_pending_flush_tasks集合中移除
        self._pending_flush_tasks.discard(task)
        # 如果task是被cancel的，直接返回
        if task.cancelled():
            return
        # 如果存在异常，打印日志
        exc = task.exception()
        if exc:
            logger.warning("Journal flush task failed: %s", exc)

    def _identify_caller(self, tags: list[str] | None) -> str:
        _tags = tags or []
        # 如果tags里面存在subagent:或者middleware:开头的 活着 lead_agent，直接返回
        for tag in _tags:
            if isinstance(tag, str) and (tag.startswith("subagent:") or tag.startswith("middleware:") or tag == "lead_agent"):
                return tag
        # Default to lead_agent: the main agent graph does not inject
        # callback tags, while subagents and middleware explicitly tag
        # themselves.
        # 默认返回lead_agent
        return "lead_agent"

    # -- Public methods (called by worker) --

    def record_external_llm_usage_records(
        self,
        records: list[dict[str, int | str]],
    ) -> None:
        """Record token usage from external sources (e.g., subagents).

        Each record should contain:
            source_run_id: Unique identifier to prevent double-counting
            caller: Caller tag (e.g. "subagent:general-purpose")
            input_tokens: Input token count
            output_tokens: Output token count
            total_tokens: Total token count (computed from input+output if 0/missing)
        """
        if not self._track_tokens:
            return
        for record in records:
            source_id = str(record.get("source_run_id", ""))
            if not source_id:
                continue
            if source_id in self._counted_external_source_ids:
                continue

            total_tk = record.get("total_tokens", 0) or 0
            if total_tk <= 0:
                input_tk = record.get("input_tokens", 0) or 0
                output_tk = record.get("output_tokens", 0) or 0
                total_tk = input_tk + output_tk
            if total_tk <= 0:
                continue

            self._counted_external_source_ids.add(source_id)
            self._total_input_tokens += record.get("input_tokens", 0) or 0
            self._total_output_tokens += record.get("output_tokens", 0) or 0
            self._total_tokens += total_tk

            caller = str(record.get("caller", ""))
            if caller.startswith("subagent:"):
                self._subagent_tokens += total_tk
            elif caller.startswith("middleware:"):
                self._middleware_tokens += total_tk
            else:
                self._lead_agent_tokens += total_tk

            self._schedule_progress_flush()

    def set_first_human_message(self, content: str) -> None:
        """Record the first human message for convenience fields."""
        self._first_human_msg = content[:2000] if content else None

    def record_middleware(self, tag: str, *, name: str, hook: str, action: str, changes: dict) -> None:
        """Record a middleware state-change event.

        Called by middleware implementations when they perform a meaningful
        state change (e.g., title generation, summarization, HITL approval).
        Pure-observation middleware should not call this.

        Args:
            tag: Short identifier for the middleware (e.g., "title", "summarize",
                 "guardrail"). Used to form event_type="middleware:{tag}".
            name: Full middleware class name.
            hook: Lifecycle hook that triggered the action (e.g., "after_model").
            action: Specific action performed (e.g., "generate_title").
            changes: Dict describing the state changes made.
        """
        self._put(
            event_type=f"middleware:{tag}",
            category="middleware",
            content={"name": name, "hook": hook, "action": action, "changes": changes},
        )

    async def flush(self) -> None:
        """Force flush remaining buffer. Called in worker's finally block."""
        if self._pending_flush_tasks:
            await asyncio.gather(*tuple(self._pending_flush_tasks), return_exceptions=True)
        while self._pending_progress_task is not None and not self._pending_progress_task.done():
            if self._pending_progress_delayed:
                self._pending_progress_task.cancel()
                await asyncio.gather(self._pending_progress_task, return_exceptions=True)
                self._progress_dirty = False
                self._pending_progress_delayed = False
                break
            await asyncio.gather(self._pending_progress_task, return_exceptions=True)

        while self._buffer:
            batch = self._buffer[: self._flush_threshold]
            del self._buffer[: self._flush_threshold]
            try:
                await self._store.put_batch(batch)
            except Exception:
                self._buffer = batch + self._buffer
                raise

    def _schedule_progress_flush(self) -> None:
        """Best-effort throttled progress snapshot for active run visibility."""
        # 如果reporter为None，直接返回。
        # worker中的实现默认传入的是RunManager的update_run_progress方法
        if self._progress_reporter is None:
            return
        now = time.monotonic()
        # 判断当前离上一次flush progress间隔了多久
        elapsed = now - self._last_progress_flush
        # 如果小于间隔阈值
        if elapsed < self._progress_flush_interval:
            # 将_progress_dirty设置为true，然后创建一个延时任务
            self._progress_dirty = True
            self._schedule_delayed_progress_flush(self._progress_flush_interval - elapsed)
            return
        # 如果大于阈值，判断当前是否有正在运行的flush task，如果有，仅是将_progress_dirty设置为true
        if self._pending_progress_task is not None and not self._pending_progress_task.done():
            self._progress_dirty = True
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._progress_dirty = False
        # 创建flush task，snapshot就是根据当前状态下的token使用量等信息创建一个快照
        self._pending_progress_task = loop.create_task(self._flush_progress_async(snapshot=self.get_completion_data()))

    def _schedule_delayed_progress_flush(self, delay: float) -> None:
        if self._pending_progress_task is not None and not self._pending_progress_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        delay = max(0.0, delay)
        self._pending_progress_delayed = delay > 0
        self._pending_progress_task = loop.create_task(self._flush_progress_async(delay=delay))

    async def _flush_progress_async(self, *, snapshot: dict | None = None, delay: float = 0.0) -> None:
        if self._progress_reporter is None:
            return
        # 如果delay时间大于0的话，sleep对应时间
        if delay > 0:
            self._pending_progress_delayed = True
            await asyncio.sleep(delay)
            self._pending_progress_delayed = False
        dirty_before_write = self._progress_dirty
        self._progress_dirty = False
        snapshot_to_write = snapshot or self.get_completion_data()
        try:
            # 将snapshot flush
            await self._progress_reporter(snapshot_to_write)
            # 设置最后一次flush的时间
            self._last_progress_flush = time.monotonic()
        except Exception:
            logger.warning("Failed to persist progress snapshot for run %s", self.run_id, exc_info=True)
        # 做一些清理工作，然后再schedule下一次progress flush
        if dirty_before_write or self._progress_dirty:
            self._progress_dirty = False
            self._pending_progress_task = None
            self._schedule_delayed_progress_flush(self._progress_flush_interval)

    def get_completion_data(self) -> dict:
        """Return accumulated token and message data for run completion."""
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_tokens,
            "llm_call_count": self._llm_call_count,
            "lead_agent_tokens": self._lead_agent_tokens,
            "subagent_tokens": self._subagent_tokens,
            "middleware_tokens": self._middleware_tokens,
            "message_count": self._msg_count,
            "last_ai_message": self._last_ai_msg,
            "first_human_message": self._first_human_msg,
        }

    @property
    def had_llm_error_fallback(self) -> bool:
        return self._had_llm_error_fallback

    @property
    def llm_error_fallback_message(self) -> str | None:
        return self._llm_error_fallback_message

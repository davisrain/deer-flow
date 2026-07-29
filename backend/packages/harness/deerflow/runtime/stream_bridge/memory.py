"""In-memory stream bridge backed by an in-process event log."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .base import END_SENTINEL, HEARTBEAT_SENTINEL, StreamBridge, StreamEvent

logger = logging.getLogger(__name__)


@dataclass
class _RunStream:
    events: list[StreamEvent] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    ended: bool = False
    start_offset: int = 0


class MemoryStreamBridge(StreamBridge):
    """Per-run in-memory event log implementation.

    Events are retained for a bounded time window per run so late subscribers
    and reconnecting clients can replay buffered events from ``Last-Event-ID``.
    """

    def __init__(self, *, queue_maxsize: int = 256) -> None:
        # 用于限制单个run_id能保存的events的数量
        self._maxsize = queue_maxsize
        # key对应的是run_id
        self._streams: dict[str, _RunStream] = {}
        # 用于统计run_id产生的event的数量
        self._counters: dict[str, int] = {}

    # -- helpers ---------------------------------------------------------------

    def _get_or_create_stream(self, run_id: str) -> _RunStream:
        # 判断run_id是否在_streams中，如果不存在，初始化对应的_RunStream对象
        if run_id not in self._streams:
            self._streams[run_id] = _RunStream()
            self._counters[run_id] = 0
        return self._streams[run_id]

    def _next_id(self, run_id: str) -> str:
        # 维护run_id产生的event的数量
        self._counters[run_id] = self._counters.get(run_id, 0) + 1
        # 并且根据seq生成event的唯一id
        ts = int(time.time() * 1000)
        seq = self._counters[run_id] - 1
        return f"{ts}-{seq}"

    def _resolve_start_offset(self, stream: _RunStream, last_event_id: str | None) -> int:
        # 如果没有传入last_event_id，使用RunStream中的起始偏移量
        if last_event_id is None:
            return stream.start_offset

        # 否则，遍历events，找到id和last_event_id一致的event
        # 偏移量为RunStream中的start_offset + index + 1
        for index, entry in enumerate(stream.events):
            if entry.id == last_event_id:
                return stream.start_offset + index + 1

        # 如果在当前的events中没有找到id等于last_event_id的元素，返回RunStream的start_offset兜底
        if stream.events:
            logger.warning(
                "last_event_id=%s not found in retained buffer; replaying from earliest retained event",
                last_event_id,
            )
        return stream.start_offset

    # -- StreamBridge API ------------------------------------------------------

    async def publish(self, run_id: str, event: str, data: Any) -> None:
        # 根据run_id获取或创建对应的RunStream对象，用于存储StreamEvent
        stream = self._get_or_create_stream(run_id)
        # 创建对应的StreamEvent
        entry = StreamEvent(id=self._next_id(run_id), event=event, data=data)
        async with stream.condition:
            # 将event添加进RunStream的events集合中
            stream.events.append(entry)
            # 如果events的长度已经超过最大值了
            if len(stream.events) > self._maxsize:
                # 计算超出的数量，并队列前面删除掉这些events
                overflow = len(stream.events) - self._maxsize
                del stream.events[:overflow]
                # 维护events队列的元素起始偏移量
                stream.start_offset += overflow
            # 唤醒等待在condition上的其他task
            stream.condition.notify_all()

    async def publish_end(self, run_id: str) -> None:
        stream = self._get_or_create_stream(run_id)
        async with stream.condition:
            stream.ended = True
            stream.condition.notify_all()

    async def subscribe(
        self,
        run_id: str,
        *,
        last_event_id: str | None = None,
        heartbeat_interval: float = 15.0,
    ) -> AsyncIterator[StreamEvent]:
        # 获取或者创建run_id对应的RunStream对象
        stream = self._get_or_create_stream(run_id)
        async with stream.condition:
            # 解析next_offset
            # 如果找到了last_event_id，那么偏移量为stream.start_offset + index(last_event_id) + 1
            # 否则使用stream.start_offset
            next_offset = self._resolve_start_offset(stream, last_event_id)

        while True:
            async with stream.condition:
                # 如果next_offset比start_offset还小，使用start_offset作为next_offset
                if next_offset < stream.start_offset:
                    logger.warning(
                        "subscriber for run %s fell behind retained buffer; resuming from offset %s",
                        run_id,
                        stream.start_offset,
                    )
                    next_offset = stream.start_offset

                # 计算出stream.events集合的索引位置
                local_index = next_offset - stream.start_offset
                # 找到对应的event，然后将next_offset + 1
                if 0 <= local_index < len(stream.events):
                    entry = stream.events[local_index]
                    next_offset += 1
                # 如果发现RunStream已经结束了，返回END_SENTINEL这个event
                elif stream.ended:
                    entry = END_SENTINEL
                # 如果local_index不在stream.events的范围内，说明已经消费完所有的events了，等待在condition上
                else:
                    try:
                        # 等待heartbeat_interval的时间，如果队列中仍没有event的话，向前端输出一个HEARTBEAT_SENTINEL event保持心跳
                        await asyncio.wait_for(stream.condition.wait(), timeout=heartbeat_interval)
                    except TimeoutError:
                        entry = HEARTBEAT_SENTINEL
                    # 如果在等待时间内被唤醒了，说明是被condition notify的，继续遍历队列中的event
                    else:
                        continue

            # 如果发现event是END_SENTINEL，yield它之后return，结束sse
            if entry is END_SENTINEL:
                yield END_SENTINEL
                return
            # 其余情况yield对应的event后继续获取event
            yield entry

    async def cleanup(self, run_id: str, *, delay: float = 0) -> None:
        if delay > 0:
            await asyncio.sleep(delay)
        self._streams.pop(run_id, None)
        self._counters.pop(run_id, None)

    async def close(self) -> None:
        self._streams.clear()
        self._counters.clear()

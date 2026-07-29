"""In-memory RunEventStore. Used when run_events.backend=memory (default) and in tests.

Thread-safe for single-process async usage (no threading locks needed
since all mutations happen within the same event loop).
"""

from __future__ import annotations

from datetime import UTC, datetime

from deerflow.runtime.events.store.base import RunEventStore


class MemoryRunEventStore(RunEventStore):
    def __init__(self) -> None:
        # key为thread_id，value为RunEvent的集合
        self._events: dict[str, list[dict]] = {}  # thread_id -> sorted event list
        # key为thread_id，value为对应的seq
        self._seq_counters: dict[str, int] = {}  # thread_id -> last assigned seq

    def _next_seq(self, thread_id: str) -> int:
        current = self._seq_counters.get(thread_id, 0)
        next_val = current + 1
        self._seq_counters[thread_id] = next_val
        return next_val

    def _put_one(
        self,
        *,
        thread_id: str,
        run_id: str,
        event_type: str,
        category: str,
        content: str | dict = "",
        metadata: dict | None = None,
        created_at: str | None = None,
    ) -> dict:
        # 获取thread_id维度的下一个seq
        seq = self._next_seq(thread_id)
        # 构建RunEventRecord对象
        record = {
            "thread_id": thread_id,
            "run_id": run_id,
            "event_type": event_type,
            "category": category,
            "content": content,
            "metadata": metadata or {},
            "seq": seq,
            "created_at": created_at or datetime.now(UTC).isoformat(),
        }
        # 保存进_events中thread_id对应的集合中
        self._events.setdefault(thread_id, []).append(record)
        return record

    async def put(
        self,
        *,
        thread_id,
        run_id,
        event_type,
        category,
        content="",
        metadata=None,
        created_at=None,
    ):
        return self._put_one(
            thread_id=thread_id,
            run_id=run_id,
            event_type=event_type,
            category=category,
            content=content,
            metadata=metadata,
            created_at=created_at,
        )

    async def put_batch(self, events):
        results = []
        # 遍历events集合
        for ev in events:
            # 依次调用_put_one方法
            record = self._put_one(**ev)
            results.append(record)
        return results

    async def list_messages(self, thread_id, *, limit=50, before_seq=None, after_seq=None):
        all_events = self._events.get(thread_id, [])
        messages = [e for e in all_events if e["category"] == "message"]

        if before_seq is not None:
            messages = [e for e in messages if e["seq"] < before_seq]
            # Take the last `limit` records
            return messages[-limit:]
        elif after_seq is not None:
            messages = [e for e in messages if e["seq"] > after_seq]
            return messages[:limit]
        else:
            # Return the latest `limit` records, ascending
            return messages[-limit:]

    async def list_events(self, thread_id, run_id, *, event_types=None, limit=500):
        all_events = self._events.get(thread_id, [])
        filtered = [e for e in all_events if e["run_id"] == run_id]
        if event_types is not None:
            filtered = [e for e in filtered if e["event_type"] in event_types]
        return filtered[:limit]

    async def list_messages_by_run(self, thread_id, run_id, *, limit=50, before_seq=None, after_seq=None):
        # 根据thread_id找到对应的event集合
        all_events = self._events.get(thread_id, [])
        # 根据run_id筛选出类型是message的event集合
        filtered = [e for e in all_events if e["run_id"] == run_id and e["category"] == "message"]
        # 如果before_seq或者after_seq存在，对集合进行截取
        if before_seq is not None:
            filtered = [e for e in filtered if e["seq"] < before_seq]
        if after_seq is not None:
            filtered = [e for e in filtered if e["seq"] > after_seq]
        # 下面这段逻辑是双游标分页。
        # 当无游标的时候，说明是初始加载，[-limit:]取最新数据
        # 当before_seq=n，说明是向上翻页的情况，[-limit:]取seq < n的最新的数据
        # 当after_seq=n，说明是向下翻页，[:limit]取seq > n的最旧的数据

        # 如果after_seq存在，limit从前往后截取
        if after_seq is not None:
            return filtered[:limit]
        else:
            # 如果after_seq不存在，limit从后往前截取
            return filtered[-limit:] if len(filtered) > limit else filtered

    async def count_messages(self, thread_id):
        all_events = self._events.get(thread_id, [])
        return sum(1 for e in all_events if e["category"] == "message")

    async def delete_by_thread(self, thread_id):
        events = self._events.pop(thread_id, [])
        self._seq_counters.pop(thread_id, None)
        return len(events)

    async def delete_by_run(self, thread_id, run_id):
        all_events = self._events.get(thread_id, [])
        if not all_events:
            return 0
        remaining = [e for e in all_events if e["run_id"] != run_id]
        removed = len(all_events) - len(remaining)
        self._events[thread_id] = remaining
        return removed

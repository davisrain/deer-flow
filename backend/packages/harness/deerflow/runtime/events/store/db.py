"""SQLAlchemy-backed RunEventStore implementation.

Persists events to the ``run_events`` table. Trace content is truncated
at ``max_trace_content`` bytes to avoid bloating the database.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deerflow.persistence.models.run_event import RunEventRow
from deerflow.runtime.events.store.base import RunEventStore
from deerflow.runtime.user_context import AUTO, _AutoSentinel, get_current_user, resolve_user_id
from deerflow.utils.time import coerce_iso

logger = logging.getLogger(__name__)


class DbRunEventStore(RunEventStore):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], *, max_trace_content: int = 10240):
        self._sf = session_factory
        self._max_trace_content = max_trace_content

    @staticmethod
    def _row_to_dict(row: RunEventRow) -> dict:
        # 将持久化对象转换为dict类型的
        d = row.to_dict()
        # 将event_metadata转换成metadata
        d["metadata"] = d.pop("event_metadata", {})
        val = d.get("created_at")
        if isinstance(val, datetime):
            # SQLite drops tzinfo on read despite ``DateTime(timezone=True)``;
            # ``coerce_iso`` normalizes naive datetimes as UTC.
            d["created_at"] = coerce_iso(val)
        d.pop("id", None)
        # Restore structured content that was JSON-serialized on write.
        raw = d.get("content", "")
        metadata = d.get("metadata", {})
        # 如果content内容是str 且 metadata里面说明了content是json或者dict
        if isinstance(raw, str) and (metadata.get("content_is_json") or metadata.get("content_is_dict")):
            try:
                # 使用json进行反序列化
                d["content"] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                # Content looked like JSON but failed to parse;
                # keep the raw string as-is.
                logger.debug("Failed to deserialize content as JSON for event seq=%s", d.get("seq"))
        return d

    def _truncate_trace(self, category: str, content: Any, metadata: dict | None) -> tuple[Any, dict]:
        # 如果是trace类型的，对content内容做截断，并且将截断标志和原始长度放入metadata中
        if category == "trace":
            text = content if isinstance(content, str) else json.dumps(content, default=str, ensure_ascii=False)
            encoded = text.encode("utf-8")
            if len(encoded) > self._max_trace_content:
                # Truncate by bytes, then decode back (may cut a multi-byte char, so use errors="ignore")
                content = encoded[: self._max_trace_content].decode("utf-8", errors="ignore")
                metadata = {**(metadata or {}), "content_truncated": True, "original_byte_length": len(encoded)}
        return content, metadata or {}

    @staticmethod
    def _content_to_db(content: Any, metadata: dict | None) -> tuple[str, dict]:
        # 如果content就是str类型，直接返回
        metadata = metadata or {}
        if isinstance(content, str):
            return content, metadata

        # 否则将content序列化为json字符串
        db_content = json.dumps(content, default=str, ensure_ascii=False)
        # 并且在metadata里面标注content是json
        metadata = {**metadata, "content_is_json": True}
        # 如果content原本是dict类型的，也在metadata里面标注
        if isinstance(content, dict):
            metadata["content_is_dict"] = True
        # 返回序列化之后的content
        return db_content, metadata

    @staticmethod
    def _user_id_from_context() -> str | None:
        """Soft read of user_id from contextvar for write paths.

        Returns ``None`` (no filter / no stamp) if contextvar is unset,
        which is the expected case for background worker writes. HTTP
        request writes will have the contextvar set by auth middleware
        and get their user_id stamped automatically.

        Coerces ``user.id`` to ``str`` at the boundary: ``User.id`` is
        typed as ``UUID`` by the auth layer, but ``run_events.user_id``
        is ``VARCHAR(64)`` and aiosqlite cannot bind a raw UUID object
        to a VARCHAR column ("type 'UUID' is not supported") — the
        INSERT would silently roll back and the worker would hang.
        """
        user = get_current_user()
        return str(user.id) if user is not None else None

    @staticmethod
    async def _max_seq_for_thread(session: AsyncSession, thread_id: str) -> int | None:
        """Return the current max seq while serializing writers per thread.

        PostgreSQL rejects ``SELECT max(...) FOR UPDATE`` because aggregate
        results are not lockable rows. As a release-safe workaround, take a
        transaction-level advisory lock keyed by thread_id before reading the
        aggregate. Other dialects keep the existing row-locking statement.
        """
        stmt = select(func.max(RunEventRow.seq)).where(RunEventRow.thread_id == thread_id)
        bind = session.get_bind()
        dialect_name = bind.dialect.name if bind is not None else ""

        if dialect_name == "postgresql":
            await session.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(CAST(:thread_id AS text))::bigint)"),
                {"thread_id": thread_id},
            )
            return await session.scalar(stmt)

        return await session.scalar(stmt.with_for_update())

    async def put(self, *, thread_id, run_id, event_type, category, content="", metadata=None, created_at=None):  # noqa: D401
        """Write a single event — low-frequency path only.

        This opens a dedicated transaction with a FOR UPDATE lock to
        assign a monotonic *seq*.  For high-throughput writes use
        :meth:`put_batch`, which acquires the lock once for the whole
        batch.  Currently the only caller is ``worker.run_agent`` for
        the initial ``human_message`` event (once per run).
        """
        content, metadata = self._truncate_trace(category, content, metadata)
        db_content, metadata = self._content_to_db(content, metadata)
        user_id = self._user_id_from_context()
        async with self._sf() as session:
            async with session.begin():
                max_seq = await self._max_seq_for_thread(session, thread_id)
                seq = (max_seq or 0) + 1
                row = RunEventRow(
                    thread_id=thread_id,
                    run_id=run_id,
                    user_id=user_id,
                    event_type=event_type,
                    category=category,
                    content=db_content,
                    event_metadata=metadata,
                    seq=seq,
                    created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(UTC),
                )
                session.add(row)
            return self._row_to_dict(row)

    async def put_batch(self, events):
        if not events:
            return []
        # 获取所有要批量put的events中的thread_id集合
        thread_ids = {e["thread_id"] for e in events}
        # 如果来自不同的thread_id，报错
        if len(thread_ids) > 1:
            raise ValueError(f"put_batch requires all events to belong to the same thread; got {thread_ids!r}")
        user_id = self._user_id_from_context()
        async with self._sf() as session:
            # session.begion 自动提交事务
            async with session.begin():
                # All events belong to the same thread (validated above).
                thread_id = events[0]["thread_id"]
                # 获取当前thread_id对应的event中最大的seq
                max_seq = await self._max_seq_for_thread(session, thread_id)
                seq = max_seq or 0
                rows = []
                # 遍历events列表
                for e in events:
                    seq += 1
                    # 获取content、category、metadata属性
                    content = e.get("content", "")
                    category = e.get("category", "trace")
                    metadata = e.get("metadata")
                    # 对trace类型的event的content做截断
                    content, metadata = self._truncate_trace(category, content, metadata)
                    # 将content转换成db能保存的数据
                    db_content, metadata = self._content_to_db(content, metadata)
                    row = RunEventRow(
                        thread_id=e["thread_id"],
                        run_id=e["run_id"],
                        user_id=e.get("user_id", user_id),
                        event_type=e["event_type"],
                        category=category,
                        content=db_content,
                        event_metadata=metadata,
                        seq=seq,
                        created_at=datetime.fromisoformat(e["created_at"]) if e.get("created_at") else datetime.now(UTC),
                    )
                    # 插入到db
                    session.add(row)
                    rows.append(row)
            return [self._row_to_dict(r) for r in rows]

    async def list_messages(
        self,
        thread_id,
        *,
        limit=50,
        before_seq=None,
        after_seq=None,
        user_id: str | None | _AutoSentinel = AUTO,
    ):
        resolved_user_id = resolve_user_id(user_id, method_name="DbRunEventStore.list_messages")
        stmt = select(RunEventRow).where(RunEventRow.thread_id == thread_id, RunEventRow.category == "message")
        if resolved_user_id is not None:
            stmt = stmt.where(RunEventRow.user_id == resolved_user_id)
        if before_seq is not None:
            stmt = stmt.where(RunEventRow.seq < before_seq)
        if after_seq is not None:
            stmt = stmt.where(RunEventRow.seq > after_seq)

        if after_seq is not None:
            # Forward pagination: first `limit` records after cursor
            stmt = stmt.order_by(RunEventRow.seq.asc()).limit(limit)
            async with self._sf() as session:
                result = await session.execute(stmt)
                return [self._row_to_dict(r) for r in result.scalars()]
        else:
            # before_seq or default (latest): take last `limit` records, return ascending
            stmt = stmt.order_by(RunEventRow.seq.desc()).limit(limit)
            async with self._sf() as session:
                result = await session.execute(stmt)
                rows = list(result.scalars())
                return [self._row_to_dict(r) for r in reversed(rows)]

    async def list_events(
        self,
        thread_id,
        run_id,
        *,
        event_types=None,
        limit=500,
        user_id: str | None | _AutoSentinel = AUTO,
    ):
        resolved_user_id = resolve_user_id(user_id, method_name="DbRunEventStore.list_events")
        stmt = select(RunEventRow).where(RunEventRow.thread_id == thread_id, RunEventRow.run_id == run_id)
        if resolved_user_id is not None:
            stmt = stmt.where(RunEventRow.user_id == resolved_user_id)
        if event_types:
            stmt = stmt.where(RunEventRow.event_type.in_(event_types))
        stmt = stmt.order_by(RunEventRow.seq.asc()).limit(limit)
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [self._row_to_dict(r) for r in result.scalars()]

    async def list_messages_by_run(
        self,
        thread_id,
        run_id,
        *,
        limit=50,
        before_seq=None,
        after_seq=None,
        user_id: str | None | _AutoSentinel = AUTO,
    ):
        resolved_user_id = resolve_user_id(user_id, method_name="DbRunEventStore.list_messages_by_run")
        # 构建从db中查找thread_id和run_id对应的message类型的event的sql
        stmt = select(RunEventRow).where(
            RunEventRow.thread_id == thread_id,
            RunEventRow.run_id == run_id,
            RunEventRow.category == "message",
        )
        # 如果user_id before_seq after_seq条件存在，拼接到sql中
        if resolved_user_id is not None:
            stmt = stmt.where(RunEventRow.user_id == resolved_user_id)
        if before_seq is not None:
            stmt = stmt.where(RunEventRow.seq < before_seq)
        if after_seq is not None:
            stmt = stmt.where(RunEventRow.seq > after_seq)

        # 如果after_seq存在，按照seq正序limit
        if after_seq is not None:
            stmt = stmt.order_by(RunEventRow.seq.asc()).limit(limit)
            async with self._sf() as session:
                result = await session.execute(stmt)
                return [self._row_to_dict(r) for r in result.scalars()]
        # 其他情况 按照seq倒序limit
        else:
            stmt = stmt.order_by(RunEventRow.seq.desc()).limit(limit)
            async with self._sf() as session:
                result = await session.execute(stmt)
                rows = list(result.scalars())
                # 最后解析到时候再将rows倒序回来
                return [self._row_to_dict(r) for r in reversed(rows)]

    async def count_messages(
        self,
        thread_id,
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ):
        resolved_user_id = resolve_user_id(user_id, method_name="DbRunEventStore.count_messages")
        stmt = select(func.count()).select_from(RunEventRow).where(RunEventRow.thread_id == thread_id, RunEventRow.category == "message")
        if resolved_user_id is not None:
            stmt = stmt.where(RunEventRow.user_id == resolved_user_id)
        async with self._sf() as session:
            return await session.scalar(stmt) or 0

    async def delete_by_thread(
        self,
        thread_id,
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ):
        resolved_user_id = resolve_user_id(user_id, method_name="DbRunEventStore.delete_by_thread")
        async with self._sf() as session:
            count_conditions = [RunEventRow.thread_id == thread_id]
            if resolved_user_id is not None:
                count_conditions.append(RunEventRow.user_id == resolved_user_id)
            count_stmt = select(func.count()).select_from(RunEventRow).where(*count_conditions)
            count = await session.scalar(count_stmt) or 0
            if count > 0:
                await session.execute(delete(RunEventRow).where(*count_conditions))
                await session.commit()
            return count

    async def delete_by_run(
        self,
        thread_id,
        run_id,
        *,
        user_id: str | None | _AutoSentinel = AUTO,
    ):
        resolved_user_id = resolve_user_id(user_id, method_name="DbRunEventStore.delete_by_run")
        async with self._sf() as session:
            count_conditions = [RunEventRow.thread_id == thread_id, RunEventRow.run_id == run_id]
            if resolved_user_id is not None:
                count_conditions.append(RunEventRow.user_id == resolved_user_id)
            count_stmt = select(func.count()).select_from(RunEventRow).where(*count_conditions)
            count = await session.scalar(count_stmt) or 0
            if count > 0:
                await session.execute(delete(RunEventRow).where(*count_conditions))
                await session.commit()
            return count

from deerflow.runtime.events.store.base import RunEventStore
from deerflow.runtime.events.store.memory import MemoryRunEventStore


def make_run_event_store(config=None) -> RunEventStore:
    """Create a RunEventStore based on run_events.backend configuration."""
    # 如果配置存在，且配置的是memory
    if config is None or config.backend == "memory":
        # 返回一个内存级别的RunEventStore
        return MemoryRunEventStore()
    # 如果配置的是db
    if config.backend == "db":
        from deerflow.persistence.engine import get_session_factory
        # 使用session_factory创建DbRunEventStore
        sf = get_session_factory()
        if sf is None:
            # database.backend=memory but run_events.backend=db -> fallback
            return MemoryRunEventStore()
        from deerflow.runtime.events.store.db import DbRunEventStore

        return DbRunEventStore(sf, max_trace_content=config.max_trace_content)
    if config.backend == "jsonl":
        from deerflow.runtime.events.store.jsonl import JsonlRunEventStore

        return JsonlRunEventStore()
    raise ValueError(f"Unknown run_events backend: {config.backend!r}")


__all__ = ["MemoryRunEventStore", "RunEventStore", "make_run_event_store"]

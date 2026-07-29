"""Shared pagination helpers for gateway routers."""

from __future__ import annotations


def trim_run_message_page(rows: list[dict], *, limit: int, after_seq: int | None) -> tuple[list[dict], bool]:
    """Trim a ``limit + 1`` run-message page while preserving page boundaries."""
    # 如果event的数量大于limit，说明还有更多的数据可以加载
    # 为什么会比limit大是因为在查询的时候将limit + 1了
    has_more = len(rows) > limit
    # 如果没有更多的消息了，has_more返回false
    if not has_more:
        return rows, False
    # 如果after_seq存在，取最旧的消息
    if after_seq is not None:
        return rows[:limit], True
    # 否则取最新的消息
    return rows[-limit:], True

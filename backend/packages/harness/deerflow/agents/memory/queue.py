"""Memory update queue with debounce mechanism."""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from deerflow.config.memory_config import get_memory_config

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Context for a conversation to be processed for memory update."""

    thread_id: str
    messages: list[Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    agent_name: str | None = None
    user_id: str | None = None
    correction_detected: bool = False
    reinforcement_detected: bool = False


class MemoryUpdateQueue:
    """Queue for memory updates with debounce mechanism.

    This queue collects conversation contexts and processes them after
    a configurable debounce period. Multiple conversations received within
    the debounce window are batched together.
    """

    def __init__(self):
        """Initialize the memory update queue."""
        self._queue: list[ConversationContext] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._processing = False

    @staticmethod
    def _queue_key(
        thread_id: str,
        user_id: str | None,
        agent_name: str | None,
    ) -> tuple[str, str | None, str | None]:
        """Return the debounce identity for a memory update target."""
        return (thread_id, user_id, agent_name)

    def add(
        self,
        thread_id: str,
        messages: list[Any],
        agent_name: str | None = None,
        user_id: str | None = None,
        correction_detected: bool = False,
        reinforcement_detected: bool = False,
    ) -> None:
        """Add a conversation to the update queue.

        Args:
            thread_id: The thread ID.
            messages: The conversation messages.
            agent_name: If provided, memory is stored per-agent. If None, uses global memory.
            user_id: The user ID captured at enqueue time. Stored in ConversationContext so it
                survives the threading.Timer boundary (ContextVar does not propagate across
                raw threads).
            correction_detected: Whether recent turns include an explicit correction signal.
            reinforcement_detected: Whether recent turns include a positive reinforcement signal.
        """
        # 查看memory模块的配置
        config = get_memory_config()
        # 如果没有开启，直接返回，不做任何操作
        if not config.enabled:
            return

        # 加锁，实际入队
        with self._lock:
            self._enqueue_locked(
                thread_id=thread_id,
                messages=messages,
                agent_name=agent_name,
                user_id=user_id,
                correction_detected=correction_detected,
                reinforcement_detected=reinforcement_detected,
            )
            # 重置计时器
            self._reset_timer()

        logger.info("Memory update queued for thread %s, queue size: %d", thread_id, len(self._queue))

    def add_nowait(
        self,
        thread_id: str,
        messages: list[Any],
        agent_name: str | None = None,
        user_id: str | None = None,
        correction_detected: bool = False,
        reinforcement_detected: bool = False,
    ) -> None:
        """Add a conversation and start processing immediately in the background."""
        config = get_memory_config()
        if not config.enabled:
            return

        with self._lock:
            self._enqueue_locked(
                thread_id=thread_id,
                messages=messages,
                agent_name=agent_name,
                user_id=user_id,
                correction_detected=correction_detected,
                reinforcement_detected=reinforcement_detected,
            )
            self._schedule_timer(0)

        logger.info("Memory update queued for immediate processing on thread %s, queue size: %d", thread_id, len(self._queue))

    def _enqueue_locked(
        self,
        *,
        thread_id: str,
        messages: list[Any],
        agent_name: str | None,
        user_id: str | None,
        correction_detected: bool,
        reinforcement_detected: bool,
    ) -> None:
        # 将thread_id user_id agent_name组成一个队列key
        queue_key = self._queue_key(thread_id, user_id, agent_name)
        # 查找队列中是否已经存在了相同key的context
        existing_context = next(
            (context for context in self._queue if self._queue_key(context.thread_id, context.user_id, context.agent_name) == queue_key),
            None,
        )
        # 将纠正 肯定的检测结果 和 队列中已经存在的context中的属性 做整合
        merged_correction_detected = correction_detected or (existing_context.correction_detected if existing_context is not None else False)
        merged_reinforcement_detected = reinforcement_detected or (existing_context.reinforcement_detected if existing_context is not None else False)
        # 根据传入的参数 和 整合出来的参数，创建一个ConversationContext实体对象
        context = ConversationContext(
            thread_id=thread_id,
            messages=messages,
            agent_name=agent_name,
            user_id=user_id,
            correction_detected=merged_correction_detected,
            reinforcement_detected=merged_reinforcement_detected,
        )

        # 将队列中的等于queue_key的元素剔除，然后将新创建的context添加进去
        self._queue = [context for context in self._queue if self._queue_key(context.thread_id, context.user_id, context.agent_name) != queue_key]
        self._queue.append(context)

    def _reset_timer(self) -> None:
        """Reset the debounce timer."""
        config = get_memory_config()
        # 使用配置中的debounce_seconds来调度timer
        self._schedule_timer(config.debounce_seconds)

        logger.debug("Memory update timer set for %ss", config.debounce_seconds)

    def _schedule_timer(self, delay_seconds: float) -> None:
        """Schedule queue processing after the provided delay."""
        # Cancel existing timer if any
        # 如果当前存在timer，cancel掉
        if self._timer is not None:
            self._timer.cancel()

        # 然后创建一个timer，设定延迟时间，到期之后执行_process_queue方法
        # 也就是说，只要delay的时间内，队列中有新的update进来了，都需要重新计时。
        # 默认30s，也就是说30s没有请求的时候，才处理队列中保存的ConversationContext
        self._timer = threading.Timer(
            delay_seconds,
            self._process_queue,
        )
        self._timer.daemon = True
        # 启动timer
        self._timer.start()

    def _process_queue(self) -> None:
        """Process all queued conversation contexts."""
        # Import here to avoid circular dependency
        from deerflow.agents.memory.updater import MemoryUpdater

        # 加锁，处理的时候防止有新的ConversationContext添加进来
        with self._lock:
            # 这里判断processing状态，如果仍在processing的话，执行schedule_timer，立即使用另一个线程来尝试process。
            # 类似于自旋吧。正常应该不会走到这个分支里面，因为update之后默认间隔30s才会进行下一次process
            if self._processing:
                # Preserve immediate flush semantics even if another worker is active.
                self._schedule_timer(0)
                return

            # 如果队列里面没有元素，直接返回
            if not self._queue:
                return

            # 如果存在元素，将队列元素复制出来并清空。设置processing为true，作为标识位。
            # 然后解锁，防止被持久化操作持有锁的时间过长，导致添加元素的业务线程被block
            self._processing = True
            contexts_to_process = self._queue.copy()
            self._queue.clear()
            # 将timer设置为None
            self._timer = None

        logger.info("Processing %d queued memory updates", len(contexts_to_process))

        # 下面的才是具体的处理，抽取事实对象并持久化的逻辑
        try:
            # 创建一个MemoryUpdater对象
            updater = MemoryUpdater()

            # 遍历需要处理的ConversationContext集合
            for context in contexts_to_process:
                try:
                    logger.info("Updating memory for thread %s", context.thread_id)
                    # 调用updater更新memory信息
                    success = updater.update_memory(
                        messages=context.messages,
                        thread_id=context.thread_id,
                        agent_name=context.agent_name,
                        correction_detected=context.correction_detected,
                        reinforcement_detected=context.reinforcement_detected,
                        user_id=context.user_id,
                    )
                    if success:
                        logger.info("Memory updated successfully for thread %s", context.thread_id)
                    else:
                        logger.warning("Memory update skipped/failed for thread %s", context.thread_id)
                except Exception as e:
                    logger.error("Error updating memory for thread %s: %s", context.thread_id, e)

                # Small delay between updates to avoid rate limiting
                # 这里短暂sleep一下防止被限流
                if len(contexts_to_process) > 1:
                    time.sleep(0.5)

        finally:
            # 最后加锁修改processing状态
            with self._lock:
                self._processing = False

    def flush(self) -> None:
        """Force immediate processing of the queue.

        This is useful for testing or graceful shutdown.
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

        self._process_queue()

    def flush_nowait(self) -> None:
        """Start queue processing immediately in a background thread."""
        with self._lock:
            # Daemon thread: queued messages may be lost if the process exits
            # before _process_queue completes. Acceptable for best-effort memory updates.
            self._schedule_timer(0)

    def clear(self) -> None:
        """Clear the queue without processing.

        This is useful for testing.
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._queue.clear()
            self._processing = False

    @property
    def pending_count(self) -> int:
        """Get the number of pending updates."""
        with self._lock:
            return len(self._queue)

    @property
    def is_processing(self) -> bool:
        """Check if the queue is currently being processed."""
        with self._lock:
            return self._processing


# Global singleton instance
_memory_queue: MemoryUpdateQueue | None = None
_queue_lock = threading.Lock()


def get_memory_queue() -> MemoryUpdateQueue:
    """Get the global memory update queue singleton.

    Returns:
        The memory update queue instance.
    """
    global _memory_queue
    with _queue_lock:
        if _memory_queue is None:
            _memory_queue = MemoryUpdateQueue()
        return _memory_queue


def reset_memory_queue() -> None:
    """Reset the global memory queue.

    This is useful for testing.
    """
    global _memory_queue
    with _queue_lock:
        if _memory_queue is not None:
            _memory_queue.clear()
        _memory_queue = None

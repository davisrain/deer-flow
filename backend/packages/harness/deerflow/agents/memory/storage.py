"""Memory storage providers."""

import abc
import json
import logging
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deerflow.config.agents_config import AGENT_NAME_PATTERN
from deerflow.config.memory_config import get_memory_config
from deerflow.config.paths import get_paths

logger = logging.getLogger(__name__)


def utc_now_iso_z() -> str:
    """Current UTC time as ISO-8601 with ``Z`` suffix (matches prior naive-UTC output)."""
    return datetime.now(UTC).isoformat().removesuffix("+00:00") + "Z"


def create_empty_memory() -> dict[str, Any]:
    """Create an empty memory structure."""
    return {
        "version": "1.0",
        "lastUpdated": utc_now_iso_z(),
        "user": {
            "workContext": {"summary": "", "updatedAt": ""},
            "personalContext": {"summary": "", "updatedAt": ""},
            "topOfMind": {"summary": "", "updatedAt": ""},
        },
        "history": {
            "recentMonths": {"summary": "", "updatedAt": ""},
            "earlierContext": {"summary": "", "updatedAt": ""},
            "longTermBackground": {"summary": "", "updatedAt": ""},
        },
        "facts": [],
    }


class MemoryStorage(abc.ABC):
    """Abstract base class for memory storage providers."""

    @abc.abstractmethod
    def load(self, agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
        """Load memory data for the given agent."""
        pass

    @abc.abstractmethod
    def reload(self, agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
        """Force reload memory data for the given agent."""
        pass

    @abc.abstractmethod
    def save(self, memory_data: dict[str, Any], agent_name: str | None = None, *, user_id: str | None = None) -> bool:
        """Save memory data for the given agent."""
        pass


class FileMemoryStorage(MemoryStorage):
    """File-based memory storage provider."""

    def __init__(self):
        """Initialize the file memory storage."""
        # Per-user/agent memory cache: keyed by (user_id, agent_name) tuple (None = global)
        # Value: (memory_data, file_mtime)
        self._memory_cache: dict[tuple[str | None, str | None], tuple[dict[str, Any], float | None]] = {}
        # Guards all reads and writes to _memory_cache across concurrent callers.
        self._cache_lock = threading.Lock()

    def _validate_agent_name(self, agent_name: str) -> None:
        """Validate that the agent name is safe to use in filesystem paths.

        Uses the repository's established AGENT_NAME_PATTERN to ensure consistency
        across the codebase and prevent path traversal or other problematic characters.
        """
        if not agent_name:
            raise ValueError("Agent name must be a non-empty string.")
        if not AGENT_NAME_PATTERN.match(agent_name):
            raise ValueError(f"Invalid agent name {agent_name!r}: names must match {AGENT_NAME_PATTERN.pattern}")

    def _get_memory_file_path(self, agent_name: str | None = None, *, user_id: str | None = None) -> Path:
        """Get the path to the memory file."""
        # 如果user_id不为None
        if user_id is not None:
            # 且agent_name不为None的话
            if agent_name is not None:
                self._validate_agent_name(agent_name)
                # 获取对应的记忆文件
                return get_paths().user_agent_memory_file(user_id, agent_name)
            # 如果agent_name为None，获取记忆配置
            config = get_memory_config()
            # 如果存在storage_path并且path是绝对路径，直接使用这个路径
            if config.storage_path and Path(config.storage_path).is_absolute():
                return Path(config.storage_path)
            # 否则，使用paths来根据user_id获取记忆文件的路径
            # 默认{project_root}/.deer-flow/users/{user_id}/memory.json
            return get_paths().user_memory_file(user_id)
        # Legacy: no user_id
        # 如果仅仅是存在agent_name的话，获取agent对应的memory的路径
        if agent_name is not None:
            self._validate_agent_name(agent_name)
            return get_paths().agent_memory_file(agent_name)
        config = get_memory_config()
        # 如果配置了记忆的存储路径
        if config.storage_path:
            p = Path(config.storage_path)
            # 根据是否是绝对路径，来决定使用的方式
            return p if p.is_absolute() else get_paths().base_dir / p
        # 兜底使用{base_dir}/memory.json这个路径
        return get_paths().memory_file

    def _load_memory_from_file(self, agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
        """Load memory data from file."""
        file_path = self._get_memory_file_path(agent_name, user_id=user_id)

        # 如果文件不存在，创建空的记忆对象
        if not file_path.exists():
            return create_empty_memory()

        try:
            # 打开文件，加载json
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load memory file: %s", e)
            return create_empty_memory()

    @staticmethod
    def _cache_key(agent_name: str | None = None, *, user_id: str | None = None) -> tuple[str | None, str | None]:
        return (user_id, agent_name)

    def load(self, agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
        """Load memory data (cached with file modification time check)."""
        # 获取记忆文件所在的路径
        file_path = self._get_memory_file_path(agent_name, user_id=user_id)
        # 构造缓存的key，使用user_id和agent_name组成的元组
        cache_key = self._cache_key(agent_name, user_id=user_id)

        try:
            # 如果文件存在，获取它的修改时间
            current_mtime = file_path.stat().st_mtime if file_path.exists() else None
        except OSError:
            current_mtime = None

        # 加锁
        with self._cache_lock:
            # 获取缓存
            cached = self._memory_cache.get(cache_key)
            # 判断修改时间，如果相等，直接返回缓存
            if cached is not None and cached[1] == current_mtime:
                return cached[0]

        # 如果修改时间不相等，重新加载
        memory_data = self._load_memory_from_file(agent_name, user_id=user_id)

        # 放入缓存，元组第一个元素是缓存值，第二个元素是修改时间
        with self._cache_lock:
            self._memory_cache[cache_key] = (memory_data, current_mtime)

        return memory_data

    def reload(self, agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
        """Reload memory data from file, forcing cache invalidation."""
        file_path = self._get_memory_file_path(agent_name, user_id=user_id)
        memory_data = self._load_memory_from_file(agent_name, user_id=user_id)
        cache_key = self._cache_key(agent_name, user_id=user_id)

        try:
            mtime = file_path.stat().st_mtime if file_path.exists() else None
        except OSError:
            mtime = None

        with self._cache_lock:
            self._memory_cache[cache_key] = (memory_data, mtime)
        return memory_data

    def save(self, memory_data: dict[str, Any], agent_name: str | None = None, *, user_id: str | None = None) -> bool:
        """Save memory data to file and update cache."""
        file_path = self._get_memory_file_path(agent_name, user_id=user_id)
        cache_key = self._cache_key(agent_name, user_id=user_id)

        try:
            # 创建目录
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # Shallow-copy before adding lastUpdated so the caller's dict is not
            # mutated as a side-effect, and the cache reference is not silently
            # updated before the file write succeeds.
            # 添加更新时间
            memory_data = {**memory_data, "lastUpdated": utc_now_iso_z()}

            # 创建临时目录
            temp_path = file_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            # 写文件
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
            # 替换文件名
            temp_path.replace(file_path)

            try:
                # 获取修改时间
                mtime = file_path.stat().st_mtime
            except OSError:
                mtime = None

            with self._cache_lock:
                # 写入缓存
                self._memory_cache[cache_key] = (memory_data, mtime)
            logger.info("Memory saved to %s", file_path)
            return True
        except OSError as e:
            logger.error("Failed to save memory file: %s", e)
            return False


_storage_instance: MemoryStorage | None = None
_storage_lock = threading.Lock()


def get_memory_storage() -> MemoryStorage:
    """Get the configured memory storage instance."""
    global _storage_instance
    # 如果storage不为None的话，直接返回
    if _storage_instance is not None:
        return _storage_instance

    # 加锁，准备初始化storage
    with _storage_lock:
        # 这里double check，防止已经有线程初始化好了
        if _storage_instance is not None:
            return _storage_instance

        # 获取记忆配置
        config = get_memory_config()
        # 获取记忆存储所使用的class，默认的是FileMemoryStorage
        storage_class_path = config.storage_class

        try:
            # 拆分配置的类为module和类名
            module_path, class_name = storage_class_path.rsplit(".", 1)
            import importlib

            # 导入对应的module，此时就会执行module里面所有的代码，对于class xxx的声明，就会调用元类的构造方法创建一个类对象
            module = importlib.import_module(module_path)
            # 获取这个类对象
            storage_class = getattr(module, class_name)

            # Validate that the configured storage is a MemoryStorage implementation
            # 如果不是类 或者 不是MemoryStorage的子类，报错
            if not isinstance(storage_class, type):
                raise TypeError(f"Configured memory storage '{storage_class_path}' is not a class: {storage_class!r}")
            if not issubclass(storage_class, MemoryStorage):
                raise TypeError(f"Configured memory storage '{storage_class_path}' is not a subclass of MemoryStorage")

            # 对类进行实例化，创建出类的实例对象
            _storage_instance = storage_class()
        except Exception as e:
            logger.error(
                "Failed to load memory storage %s, falling back to FileMemoryStorage: %s",
                storage_class_path,
                e,
            )
            # 如果出现异常，兜底使用FileMemoryStorage
            _storage_instance = FileMemoryStorage()

    return _storage_instance

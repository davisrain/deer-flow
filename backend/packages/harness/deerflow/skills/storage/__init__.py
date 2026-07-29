"""SkillStorage singleton + reflection-based factory.

Mirrors the pattern used by ``deerflow/sandbox/sandbox_provider.py``.
"""

from __future__ import annotations

from deerflow.skills.storage.local_skill_storage import LocalSkillStorage
from deerflow.skills.storage.skill_storage import SkillStorage

_default_skill_storage: SkillStorage | None = None
_default_skill_storage_config: object | None = None  # AppConfig identity the singleton was built from


def get_or_new_skill_storage(**kwargs) -> SkillStorage:
    """Return a ``SkillStorage`` instance — either a new one or the process singleton.

    **New instance** is created (never cached) when:
    - ``skills_path`` is provided — uses it as the ``host_path`` override (class still resolved via config).
    - ``app_config`` is provided — constructs a storage from ``app_config.skills``
      so that per-request config (e.g. Gateway ``Depends(get_config)``) is respected
      without polluting the process-level singleton.

    **Singleton** is returned (created on first call, then reused) when neither
    ``skills_path`` nor ``app_config`` is given — uses ``get_app_config()`` to
    resolve the active configuration.
    """
    global _default_skill_storage, _default_skill_storage_config

    from deerflow.config import get_app_config
    from deerflow.config.skills_config import SkillsConfig

    # 创建生成storage的函数
    def _make_storage(skills_config: SkillsConfig, *, host_path: str | None = None, **kwargs) -> SkillStorage:
        from deerflow.reflection import resolve_class
        # 从SkillsConfig配置对象中拿出use属性，解析出对应的类对象
        # 默认是deerflow.skills.storage.local_skill_storage:LocalSkillStorage
        cls = resolve_class(skills_config.use, SkillStorage)
        # 然后调用类创建出对应的Storage实例
        return cls(
            # 其中宿主机中skill文件存储的路径，如果参数传了就用参数的，否则用配置对象里面的
            host_path=host_path if host_path is not None else str(skills_config.get_skills_path()),
            # 获取配置对象里面容器中的skill文件的存储路径
            container_path=skills_config.container_path,
            **kwargs,
        )

    # 从参数中获取skills的路径 和 app_config
    skills_path = kwargs.pop("skills_path", None)
    app_config = kwargs.pop("app_config", None)

    if skills_path is not None:
        if app_config is not None:
            # 如果skills保存的路径 和 app_config都存在，尝试生成对应的skill storage
            return _make_storage(app_config.skills, host_path=str(skills_path), **kwargs)
        # No app_config: use a default SkillsConfig so we never need to read config.yaml
        # when the caller has already supplied an explicit host path.
        from deerflow.config.skills_config import SkillsConfig

        # 如果只有skills_path存在，创建一个空的SkillsConfig对象调用_make_storage
        return _make_storage(SkillsConfig(), host_path=str(skills_path), **kwargs)

    # 如果app_config存在，直接使用它来创建storage
    if app_config is not None:
        return _make_storage(app_config.skills, **kwargs)

    # If the singleton was manually injected (e.g. in tests) without a config
    # identity (_default_skill_storage_config is None), skip get_app_config()
    # entirely to avoid requiring a config.yaml on disk.
    # 如果默认的skill storage存在，且默认的skill storage配置不存在，直接返回默认的skill storage
    if _default_skill_storage is not None and _default_skill_storage_config is None:
        return _default_skill_storage

    # 重新获取最新的项目配置
    app_config_now = get_app_config()
    # 生成默认的skill storage和 skill_storage_config对象，然后返回默认的storage
    if _default_skill_storage is None or _default_skill_storage_config is not app_config_now:
        _default_skill_storage = _make_storage(app_config_now.skills, **kwargs)
        _default_skill_storage_config = app_config_now
    return _default_skill_storage


def reset_skill_storage() -> None:
    """Clear the cached singleton (used in tests and hot-reload scenarios)."""
    global _default_skill_storage, _default_skill_storage_config
    _default_skill_storage = None
    _default_skill_storage_config = None


__all__ = [
    "LocalSkillStorage",
    "SkillStorage",
    "get_or_new_skill_storage",
    "reset_skill_storage",
]

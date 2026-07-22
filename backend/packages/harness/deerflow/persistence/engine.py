"""Async SQLAlchemy engine lifecycle management.

Initializes at Gateway startup, provides session factory for
repositories, disposes at shutdown.

When database.backend="memory", init_engine is a no-op and
get_session_factory() returns None. Repositories must check for
None and fall back to in-memory implementations.
"""

from __future__ import annotations

import json
import logging

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def _json_serializer(obj: object) -> str:
    """JSON serializer with ensure_ascii=False for Chinese character support."""
    return json.dumps(obj, ensure_ascii=False)


logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def _auto_create_postgres_db(url: str) -> None:
    """Connect to the ``postgres`` maintenance DB and CREATE DATABASE.

    The target database name is extracted from *url*.  The connection is
    made to the default ``postgres`` database on the same server using
    ``AUTOCOMMIT`` isolation (CREATE DATABASE cannot run inside a
    transaction).
    """
    from sqlalchemy import text
    from sqlalchemy.engine.url import make_url

    parsed = make_url(url)
    db_name = parsed.database
    if not db_name:
        raise ValueError("Cannot auto-create database: no database name in URL")

    # Connect to the default 'postgres' database to issue CREATE DATABASE
    maint_url = parsed.set(database="postgres")
    maint_engine = create_async_engine(maint_url, isolation_level="AUTOCOMMIT")
    try:
        async with maint_engine.connect() as conn:
            await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
        logger.info("Auto-created PostgreSQL database: %s", db_name)
    finally:
        await maint_engine.dispose()


async def init_engine(
    backend: str,
    *,
    url: str = "",
    echo: bool = False,
    pool_size: int = 5,
    sqlite_dir: str = "",
) -> None:
    """Create the async engine and session factory, then auto-create tables.

    Args:
        backend: "memory", "sqlite", or "postgres".
        url: SQLAlchemy async URL (for sqlite/postgres).
        echo: Echo SQL to log.
        pool_size: Postgres connection pool size.
        sqlite_dir: Directory to create for SQLite (ensured to exist).
    """
    global _engine, _session_factory

    if backend == "memory":
        logger.info("Persistence backend=memory -- ORM engine not initialized")
        return

    if backend == "postgres":
        try:
            import asyncpg  # noqa: F401
        except ImportError:
            raise ImportError(
                "database.backend is set to 'postgres' but asyncpg is not installed.\n"
                "Install it with:\n"
                "    cd backend && uv sync --all-packages --extra postgres\n"
                "On the next `make dev` the postgres extra is auto-detected from\n"
                "config.yaml (database.backend: postgres) and reinstalled, so it\n"
                "will not be wiped again. Set UV_EXTRAS=postgres in .env to opt in\n"
                "explicitly. Or switch to backend: sqlite in config.yaml for\n"
                "single-node deployment."
            ) from None

    if backend == "sqlite":
        import os

        from sqlalchemy import event
        # 创建对应的sqlite db目录
        os.makedirs(sqlite_dir or ".", exist_ok=True)
        # sqlalchemy python的orm框架，这里应该是在根据数据库url创建对应的数据库引擎
        _engine = create_async_engine(url, echo=echo, json_serializer=_json_serializer)

        # Enable WAL on every new connection. SQLite PRAGMA settings are
        # per-connection, so we wire the listener instead of running PRAGMA
        # once at startup. WAL gives concurrent reads + writers without
        # blocking and is the standard recommendation for any production
        # SQLite deployment (TC-UPG-06 in AUTH_TEST_PLAN.md). The companion
        # ``synchronous=NORMAL`` is the safe-and-fast pairing — fsync only
        # at WAL checkpoint boundaries instead of every commit.
        # Note: we do not set PRAGMA busy_timeout here — Python's sqlite3
        # driver already defaults to a 5-second busy timeout (see the
        # ``timeout`` kwarg of ``sqlite3.connect``), and aiosqlite /
        # SQLAlchemy's aiosqlite dialect inherit that default.  Setting
        # it again would be a no-op.
        @event.listens_for(_engine.sync_engine, "connect")
        # 每次创建连接的时候执行一些语句
        def _enable_sqlite_wal(dbapi_conn, _record):  # noqa: ARG001 — SQLAlchemy contract
            cursor = dbapi_conn.cursor()
            try:
                # write after log，先写日志再写db
                cursor.execute("PRAGMA journal_mode=WAL;")
                # 日志的fsync的模式
                cursor.execute("PRAGMA synchronous=NORMAL;")
                # 外键约束关掉
                cursor.execute("PRAGMA foreign_keys=ON;")
            finally:
                cursor.close()
    elif backend == "postgres":
        _engine = create_async_engine(
            url,
            echo=echo,
            pool_size=pool_size,
            pool_pre_ping=True,
            json_serializer=_json_serializer,
        )
    else:
        raise ValueError(f"Unknown persistence backend: {backend!r}")

    # 创建session工厂，后续每次数据库操作都从这里拿session
    # session相当于是ORM层的，使用业务实体进行db操作
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    # Auto-create tables (dev convenience). Production should use Alembic.
    from deerflow.persistence.base import Base

    # Import all models so Base.metadata discovers them.
    # When no models exist yet (scaffolding phase), this is a no-op.
    try:
        # 这里导入所有的models对象，Base.metadata就会发现它们，并将它们注册为元数据，为下面create_all创建表做准备
        # (继承了Base的类，类定义的时候就会执行注册操作。因为导入类声明的时候，相当于就是创建对应的类对象。
        # 创建类对象就等于执行了metaclass(classname, 父类元组, namespace)，
        # 通常情况下metaclass就是type，type的__new__方法里面会去调用父类的__init_subclass__方法，我猜测注册操作就是在Base类的这个方法中实现的)
        import deerflow.persistence.models  # noqa: F401
    except ImportError:
        # Models package not yet available — tables won't be auto-created.
        # This is expected during initial scaffolding or minimal installs.
        logger.debug("deerflow.persistence.models not found; skipping auto-create tables")

    try:
        # begin获取的Connection是Core层的，能直接执行sql，和session有区别
        async with _engine.begin() as conn:
            # 创建所有的表 create table if not exist
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        if backend == "postgres" and "does not exist" in str(exc):
            # Database not yet created — attempt to auto-create it, then retry.
            await _auto_create_postgres_db(url)
            # Rebuild engine against the now-existing database
            await _engine.dispose()
            _engine = create_async_engine(url, echo=echo, pool_size=pool_size, pool_pre_ping=True, json_serializer=_json_serializer)
            _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
            async with _engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        else:
            raise

    logger.info("Persistence engine initialized: backend=%s", backend)


async def init_engine_from_config(config) -> None:
    """Convenience: init engine from a DatabaseConfig object."""
    if config.backend == "memory":
        await init_engine("memory")
        return
    await init_engine(
        # 配置的database类型，取值是sqlite或者postgres
        backend=config.backend,
        # 配置中对应的db的url
        url=config.app_sqlalchemy_url,
        # 是否要打印所有的sql日志
        echo=config.echo_sql,
        # 数据库连接池的数量，默认5
        pool_size=config.pool_size,
        # 如果是sqlite类型的db，获取其数据目录保存的地方
        sqlite_dir=config.sqlite_dir if config.backend == "sqlite" else "",
    )


def get_session_factory() -> async_sessionmaker[AsyncSession] | None:
    """Return the async session factory, or None if backend=memory."""
    return _session_factory


def get_engine() -> AsyncEngine | None:
    """Return the async engine, or None if not initialized."""
    return _engine


async def close_engine() -> None:
    """Dispose the engine, release all connections."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("Persistence engine closed")
    _engine = None
    _session_factory = None

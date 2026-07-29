"""Runtime path resolution for standalone harness usage."""

import os
from pathlib import Path


def project_root() -> Path:
    """Return the caller project root for runtime-owned files."""
    # 如果环境变量DEER_FLOW_PROJECT_ROOT存在
    if env_root := os.getenv("DEER_FLOW_PROJECT_ROOT"):
        # 解析对应的项目根路径
        root = Path(env_root).resolve()
        # 校验路径是否存在以及路径是否是文件夹
        if not root.exists():
            raise ValueError(f"DEER_FLOW_PROJECT_ROOT is set to '{env_root}', but the resolved path '{root}' does not exist.")
        if not root.is_dir():
            raise ValueError(f"DEER_FLOW_PROJECT_ROOT is set to '{env_root}', but the resolved path '{root}' is not a directory.")
        return root
    # 如果没有设置环境变量，直接返回当前的工作目录
    return Path.cwd().resolve()


def runtime_home() -> Path:
    """Return the writable DeerFlow state directory."""
    if env_home := os.getenv("DEER_FLOW_HOME"):
        return Path(env_home).resolve()
    # 使用${project_root}/.deer-flow作为运行时的home目录
    return project_root() / ".deer-flow"


def resolve_path(value: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    """Resolve absolute paths as-is and relative paths against the project root."""
    path = Path(value)
    # 如果path不是绝对路径的话
    if not path.is_absolute():
        # 在前面加上base 或者 {project_root}路径
        path = (base or project_root()) / path
    return path.resolve()


def existing_project_file(names: tuple[str, ...]) -> Path | None:
    """Return the first existing named file under the project root."""
    # 获取项目根目录
    root = project_root()
    # 遍历传入的names，拼接到根目录之后，如果对应的路径是文件，直接返回
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None

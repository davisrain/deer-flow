"""Shared helpers for turning conversations into memory update inputs."""

from __future__ import annotations

import re
from copy import copy
from typing import Any

_UPLOAD_BLOCK_RE = re.compile(r"<uploaded_files>[\s\S]*?</uploaded_files>\n*", re.IGNORECASE)
_CORRECTION_PATTERNS = (
    re.compile(r"\bthat(?:'s| is) (?:wrong|incorrect)\b", re.IGNORECASE),
    re.compile(r"\byou misunderstood\b", re.IGNORECASE),
    re.compile(r"\btry again\b", re.IGNORECASE),
    re.compile(r"\bredo\b", re.IGNORECASE),
    re.compile(r"不对"),
    re.compile(r"你理解错了"),
    re.compile(r"你理解有误"),
    re.compile(r"重试"),
    re.compile(r"重新来"),
    re.compile(r"换一种"),
    re.compile(r"改用"),
)
_REINFORCEMENT_PATTERNS = (
    re.compile(r"\byes[,.]?\s+(?:exactly|perfect|that(?:'s| is) (?:right|correct|it))\b", re.IGNORECASE),
    re.compile(r"\bperfect(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bexactly\s+(?:right|correct)\b", re.IGNORECASE),
    re.compile(r"\bthat(?:'s| is)\s+(?:exactly\s+)?(?:right|correct|what i (?:wanted|needed|meant))\b", re.IGNORECASE),
    re.compile(r"\bkeep\s+(?:doing\s+)?that\b", re.IGNORECASE),
    re.compile(r"\bjust\s+(?:like\s+)?(?:that|this)\b", re.IGNORECASE),
    re.compile(r"\bthis is (?:great|helpful)\b(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bthis is what i wanted\b(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"对[，,]?\s*就是这样(?:[。！？!?.]|$)"),
    re.compile(r"完全正确(?:[。！？!?.]|$)"),
    re.compile(r"(?:对[，,]?\s*)?就是这个意思(?:[。！？!?.]|$)"),
    re.compile(r"正是我想要的(?:[。！？!?.]|$)"),
    re.compile(r"继续保持(?:[。！？!?.]|$)"),
)


def extract_message_text(message: Any) -> str:
    """Extract plain text from message content for filtering and signal detection."""
    content = getattr(message, "content", "")
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                text_val = part.get("text")
                if isinstance(text_val, str):
                    text_parts.append(text_val)
        return " ".join(text_parts)
    return str(content)


def filter_messages_for_memory(messages: list[Any]) -> list[Any]:
    """Keep only user inputs and final assistant responses for memory updates."""
    filtered = []
    skip_next_ai = False
    # 遍历消息列表
    for msg in messages:
        msg_type = getattr(msg, "type", None)

        # 如果是human类型的消息
        if msg_type == "human":
            # 获取消息内容
            content_str = extract_message_text(msg)
            if "<uploaded_files>" in content_str:
                # 将文件上传的内容模块去掉
                stripped = _UPLOAD_BLOCK_RE.sub("", content_str).strip()
                # 如果是纯文件上传，没有其他内容，丢掉下一个ai消息，因为没有价值
                if not stripped:
                    skip_next_ai = True
                    continue
                # 如果不是纯文件上传，将文件上传的内容剔除之后设置到新的msg中，用于保存memory
                clean_msg = copy(msg)
                # 将剔除文件上传模块之后的文本放入clean_msg
                clean_msg.content = stripped
                filtered.append(clean_msg)
                skip_next_ai = False
            else:
                filtered.append(msg)
                skip_next_ai = False
        # 如果是ai类型的消息
        elif msg_type == "ai":
            tool_calls = getattr(msg, "tool_calls", None)
            # 且不是toll_calls类型的ai消息
            if not tool_calls:
                # 判断是否要跳过，不跳过的话添加进最终要返回的集合
                if skip_next_ai:
                    skip_next_ai = False
                    continue
                filtered.append(msg)

    return filtered


def detect_correction(messages: list[Any]) -> bool:
    """Detect explicit user corrections in recent conversation turns."""
    # 查看最近6条消息中的human消息
    recent_user_msgs = [msg for msg in messages[-6:] if getattr(msg, "type", None) == "human"]

    # 遍历这些消息
    for msg in recent_user_msgs:
        # 提取消息的内容
        content = extract_message_text(msg).strip()
        # 正则匹配否定的模版，如果消息内容里面出现了这些否定词，表示用户在纠正ai，返回True
        if content and any(pattern.search(content) for pattern in _CORRECTION_PATTERNS):
            return True

    # 否则返回False
    return False


def detect_reinforcement(messages: list[Any]) -> bool:
    """Detect explicit positive reinforcement signals in recent conversation turns."""
    # 查看最新的6条消息中的human消息
    recent_user_msgs = [msg for msg in messages[-6:] if getattr(msg, "type", None) == "human"]

    # 遍历消息
    for msg in recent_user_msgs:
        # 提取消息内容
        content = extract_message_text(msg).strip()
        # 正则匹配强化的模版，如果消息内容出现了这些肯定词，表示用户在肯定ai，返回True
        if content and any(pattern.search(content) for pattern in _REINFORCEMENT_PATTERNS):
            return True

    return False

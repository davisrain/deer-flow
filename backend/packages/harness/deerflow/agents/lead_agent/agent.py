"""Lead agent factory.

INVARIANT — tracing callback placement
======================================

Tracing callbacks (Langfuse, LangSmith) are attached at the **graph
invocation root** in :func:`_make_lead_agent` (see the
``build_tracing_callbacks()`` block that appends to ``config["callbacks"]``).
Every ``create_chat_model(...)`` call inside this module — and inside any
middleware reachable from this graph (e.g. ``TitleMiddleware``) — MUST pass
``attach_tracing=False``.

Forgetting that flag emits duplicate spans (one rooted at the graph, one at
the model) AND prevents the Langfuse handler's ``propagate_attributes``
path from firing, so ``session_id`` / ``user_id`` never reach the trace.
The four current sites are: bootstrap agent, default agent, summarization
middleware, and the async path inside ``TitleMiddleware``. Any new in-graph
``create_chat_model`` call must add to this list and pass the flag.
"""

from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig

from deerflow.agents.lead_agent.prompt import apply_prompt_template
from deerflow.agents.memory.summarization_hook import memory_flush_hook
from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware
from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware
from deerflow.agents.middlewares.safety_finish_reason_middleware import SafetyFinishReasonMiddleware
from deerflow.agents.middlewares.subagent_limit_middleware import SubagentLimitMiddleware
from deerflow.agents.middlewares.summarization_middleware import BeforeSummarizationHook, DeerFlowSummarizationMiddleware
from deerflow.agents.middlewares.title_middleware import TitleMiddleware
from deerflow.agents.middlewares.todo_middleware import TodoMiddleware
from deerflow.agents.middlewares.token_usage_middleware import TokenUsageMiddleware
from deerflow.agents.middlewares.tool_error_handling_middleware import build_lead_runtime_middlewares
from deerflow.agents.middlewares.view_image_middleware import ViewImageMiddleware
from deerflow.agents.thread_state import ThreadState
from deerflow.config.agents_config import load_agent_config, validate_agent_name
from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.models import create_chat_model
from deerflow.skills.tool_policy import filter_tools_by_skill_allowed_tools
from deerflow.skills.types import Skill
from deerflow.tracing import build_tracing_callbacks

logger = logging.getLogger(__name__)


def _get_runtime_config(config: RunnableConfig) -> dict:
    """Merge legacy configurable options with LangGraph runtime context."""
    cfg = dict(config.get("configurable", {}) or {})
    context = config.get("context", {}) or {}
    if isinstance(context, dict):
        cfg.update(context)
    return cfg


def _resolve_model_name(requested_model_name: str | None = None, *, app_config: AppConfig | None = None) -> str:
    """Resolve a runtime model name safely, falling back to default if invalid. Returns None if no models are configured."""
    app_config = app_config or get_app_config()
    default_model_name = app_config.models[0].name if app_config.models else None
    if default_model_name is None:
        raise ValueError("No chat models are configured. Please configure at least one model in config.yaml.")

    if requested_model_name and app_config.get_model_config(requested_model_name):
        return requested_model_name

    if requested_model_name and requested_model_name != default_model_name:
        logger.warning(f"Model '{requested_model_name}' not found in config; fallback to default model '{default_model_name}'.")
    return default_model_name


def _create_summarization_middleware(*, app_config: AppConfig | None = None) -> DeerFlowSummarizationMiddleware | None:
    """Create and configure the summarization middleware from config."""
    resolved_app_config = app_config or get_app_config()
    config = resolved_app_config.summarization

    if not config.enabled:
        return None

    # Prepare trigger parameter
    # 从配置文件中获取触发summary的条件
    # 支持三种类型：tokens、messages、fraction
    trigger = None
    if config.trigger is not None:
        if isinstance(config.trigger, list):
            trigger = [t.to_tuple() for t in config.trigger]
        else:
            trigger = config.trigger.to_tuple()

    # Prepare keep parameter
    # 从配置文件中获取最终保留的数据
    keep = config.keep.to_tuple()

    # Prepare model parameter.
    # Bind "middleware:summarize" tag so RunJournal identifies these LLM calls
    # as middleware rather than lead_agent (SummarizationMiddleware is a
    # LangChain built-in, so we tag the model at creation time).
    # attach_tracing=False because the graph-level RunnableConfig (set in
    # ``_make_lead_agent``) already carries tracing callbacks; binding them
    # again at the model level would emit duplicate spans and break
    # ``session_id`` / ``user_id`` propagation.
    # 可以专门配置用于摘要的轻量模型
    if config.model_name:
        model = create_chat_model(name=config.model_name, thinking_enabled=False, app_config=resolved_app_config, attach_tracing=False)
    # 如果没有配置，使用默认模型
    else:
        model = create_chat_model(thinking_enabled=False, app_config=resolved_app_config, attach_tracing=False)
    model = model.with_config(tags=["middleware:summarize"])

    # Prepare kwargs
    kwargs = {
        "model": model,
        "trigger": trigger,
        "keep": keep,
    }

    if config.trim_tokens_to_summarize is not None:
        kwargs["trim_tokens_to_summarize"] = config.trim_tokens_to_summarize

    if config.summary_prompt is not None:
        kwargs["summary_prompt"] = config.summary_prompt

    hooks: list[BeforeSummarizationHook] = []
    if resolved_app_config.memory.enabled:
        hooks.append(memory_flush_hook)

    # The logic below relies on two assumptions holding true: this factory is
    # the sole entry point for DeerFlowSummarizationMiddleware, and the runtime
    # config is not expected to change after startup.
    skills_container_path = resolved_app_config.skills.container_path or "/mnt/skills"

    return DeerFlowSummarizationMiddleware(
        **kwargs,
        skills_container_path=skills_container_path,
        skill_file_read_tool_names=config.skill_file_read_tool_names,
        before_summarization=hooks,
        preserve_recent_skill_count=config.preserve_recent_skill_count,
        preserve_recent_skill_tokens=config.preserve_recent_skill_tokens,
        preserve_recent_skill_tokens_per_skill=config.preserve_recent_skill_tokens_per_skill,
    )


def _create_todo_list_middleware(is_plan_mode: bool) -> TodoMiddleware | None:
    """Create and configure the TodoList middleware.

    Args:
        is_plan_mode: Whether to enable plan mode with TodoList middleware.

    Returns:
        TodoMiddleware instance if plan mode is enabled, None otherwise.
    """
    if not is_plan_mode:
        return None

    # Custom prompts matching DeerFlow's style
    system_prompt = """
<todo_list_system>
You have access to the `write_todos` tool to help you manage and track complex multi-step objectives.

**CRITICAL RULES:**
- Mark todos as completed IMMEDIATELY after finishing each step - do NOT batch completions
- Keep EXACTLY ONE task as `in_progress` at any time (unless tasks can run in parallel)
- Update the todo list in REAL-TIME as you work - this gives users visibility into your progress
- DO NOT use this tool for simple tasks (< 3 steps) - just complete them directly

**When to Use:**
This tool is designed for complex objectives that require systematic tracking:
- Complex multi-step tasks requiring 3+ distinct steps
- Non-trivial tasks needing careful planning and execution
- User explicitly requests a todo list
- User provides multiple tasks (numbered or comma-separated list)
- The plan may need revisions based on intermediate results

**When NOT to Use:**
- Single, straightforward tasks
- Trivial tasks (< 3 steps)
- Purely conversational or informational requests
- Simple tool calls where the approach is obvious

**Best Practices:**
- Break down complex tasks into smaller, actionable steps
- Use clear, descriptive task names
- Remove tasks that become irrelevant
- Add new tasks discovered during implementation
- Don't be afraid to revise the todo list as you learn more

**Task Management:**
Writing todos takes time and tokens - use it when helpful for managing complex problems, not for simple requests.
</todo_list_system>
"""

    tool_description = """Use this tool to create and manage a structured task list for complex work sessions.

**IMPORTANT: Only use this tool for complex tasks (3+ steps). For simple requests, just do the work directly.**

## When to Use

Use this tool in these scenarios:
1. **Complex multi-step tasks**: When a task requires 3 or more distinct steps or actions
2. **Non-trivial tasks**: Tasks requiring careful planning or multiple operations
3. **User explicitly requests todo list**: When the user directly asks you to track tasks
4. **Multiple tasks**: When users provide a list of things to be done
5. **Dynamic planning**: When the plan may need updates based on intermediate results

## When NOT to Use

Skip this tool when:
1. The task is straightforward and takes less than 3 steps
2. The task is trivial and tracking provides no benefit
3. The task is purely conversational or informational
4. It's clear what needs to be done and you can just do it

## How to Use

1. **Starting a task**: Mark it as `in_progress` BEFORE beginning work
2. **Completing a task**: Mark it as `completed` IMMEDIATELY after finishing
3. **Updating the list**: Add new tasks, remove irrelevant ones, or update descriptions as needed
4. **Multiple updates**: You can make several updates at once (e.g., complete one task and start the next)

## Task States

- `pending`: Task not yet started
- `in_progress`: Currently working on (can have multiple if tasks run in parallel)
- `completed`: Task finished successfully

## Task Completion Requirements

**CRITICAL: Only mark a task as completed when you have FULLY accomplished it.**

Never mark a task as completed if:
- There are unresolved issues or errors
- Work is partial or incomplete
- You encountered blockers preventing completion
- You couldn't find necessary resources or dependencies
- Quality standards haven't been met

If blocked, keep the task as `in_progress` and create a new task describing what needs to be resolved.

## Best Practices

- Create specific, actionable items
- Break complex tasks into smaller, manageable steps
- Use clear, descriptive task names
- Update task status in real-time as you work
- Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
- Remove tasks that are no longer relevant
- **IMPORTANT**: When you write the todo list, mark your first task(s) as `in_progress` immediately
- **IMPORTANT**: Unless all tasks are completed, always have at least one task `in_progress` to show progress

Being proactive with task management demonstrates thoroughness and ensures all requirements are completed successfully.

**Remember**: If you only need a few tool calls to complete a task and it's clear what to do, it's better to just do the task directly and NOT use this tool at all.
"""

    return TodoMiddleware(system_prompt=system_prompt, tool_description=tool_description)


# ThreadDataMiddleware must be before SandboxMiddleware to ensure thread_id is available
# UploadsMiddleware should be after ThreadDataMiddleware to access thread_id
# DanglingToolCallMiddleware patches missing ToolMessages before model sees the history
# SummarizationMiddleware should be early to reduce context before other processing
# TodoListMiddleware should be before ClarificationMiddleware to allow todo management
# TitleMiddleware generates title after first exchange
# MemoryMiddleware queues conversation for memory update (after TitleMiddleware)
# ViewImageMiddleware should be before ClarificationMiddleware to inject image details before LLM
# ToolErrorHandlingMiddleware should be before ClarificationMiddleware to convert tool exceptions to ToolMessages
# ClarificationMiddleware should be last to intercept clarification requests after model calls
def build_middlewares(
    config: RunnableConfig,
    model_name: str | None,
    agent_name: str | None = None,
    custom_middlewares: list[AgentMiddleware] | None = None,
    *,
    app_config: AppConfig | None = None,
    deferred_setup=None,
):
    """Build the lead-agent middleware chain based on runtime configuration.

    Public entry point for the lead agent's full middleware composition. Used by
    ``make_lead_agent`` and by the embedded ``DeerFlowClient`` (a lead-agent variant
    that needs the identical chain). Keep this name stable: it is imported across a
    module boundary, so renames/signature changes ripple into ``client.py``.

    Args:
        config: Runtime configuration containing configurable options like is_plan_mode.
        model_name: Resolved runtime model name; gates vision-only middleware.
        agent_name: If provided, MemoryMiddleware will use per-agent memory storage.
        custom_middlewares: Optional list of custom middlewares to inject into the chain.
        app_config: Explicit AppConfig; falls back to ``get_app_config()`` when omitted.
        deferred_setup: Optional deferred-MCP-tool setup that attaches
            ``DeferredToolFilterMiddleware`` when ``tool_search`` is enabled.

    Returns:
        List of middleware instances.
    """
    resolved_app_config = app_config or get_app_config()
    # 构建一些主要的middleware
    # ThreadDataMiddleware 创建thread_id维度的数据目录
    # SandboxMiddleware 创建sandbox
    # DanglingToolCallMiddleware 修复缺失和ToolMessage
    # LLMErrorHandlingMiddleware 处理llm返回异常：重试、熔断、降级
    # SandboxAuditMiddleware 对bash命令做审计
    # ToolErrorHandlingMiddleware 处理tool调用异常，降级为ToolMessage，并且给task工具调用添加subagent的执行状态
    middlewares = build_lead_runtime_middlewares(app_config=resolved_app_config, lazy_init=True)

    # Always inject current date (and optionally memory) as <system-reminder> into the
    # first HumanMessage to keep the system prompt fully static for prefix-cache reuse.
    from deerflow.agents.middlewares.dynamic_context_middleware import DynamicContextMiddleware

    # 将时间和memory等动态信息拆分到一个特殊的HumanMessage中，作为一个<system-reminder>，以达到将system prompt静态化的效果，以命中prefix cache，节省token。
    # 以前的版本currentdate和memory是在SystemPrompt中的
    middlewares.append(DynamicContextMiddleware(agent_name=agent_name, app_config=resolved_app_config))

    # Add summarization middleware if enabled
    # 在每次调用 LLM 前检测对话历史是否过长，过长则用 LLM 把旧消息压缩成摘要，同时确保 skill 内容、日期记忆 reminder 不被误删，并在压缩前把历史刷入记忆系统。
    summarization_middleware = _create_summarization_middleware(app_config=resolved_app_config)
    if summarization_middleware is not None:
        middlewares.append(summarization_middleware)

    # Add TodoList middleware if plan mode is enabled
    # 根据是否plan mode决定是否要添加TodoListMiddleware
    cfg = _get_runtime_config(config)
    is_plan_mode = cfg.get("is_plan_mode", False)
    # _create_todo_list_middleware 在 plan mode 下创建 TodoMiddleware，
    # 它做三件事：被摘要截断后重新注入TodoList提醒、有未完成任务时拦截模型的提前退出并强制续约（最多 2 次）、run 切换时清理跨 run 的提醒队列防止污染。
    todo_list_middleware = _create_todo_list_middleware(is_plan_mode)
    if todo_list_middleware is not None:
        middlewares.append(todo_list_middleware)

    # Add TokenUsageMiddleware when token_usage tracking is enabled
    if resolved_app_config.token_usage.enabled:
        # TokenUsageMiddleware 在每次模型返回后：打 token 日志、给 AIMessage 打上结构化的"这一步做了什么"归因标记（供前端渲染步骤卡片）、
        # 并把已完成子 agent 的 token 消耗回写到派发它的 AIMessage 上。
        middlewares.append(TokenUsageMiddleware())

    # Add TitleMiddleware
    # 根据第一个HumanMessage和AIMessage调用llm生成title，如果是同步或者没有配置title的llm，用user_msg兜底
    # 前期是titleConfig是enable的
    middlewares.append(TitleMiddleware(app_config=resolved_app_config))

    # Add MemoryMiddleware (after TitleMiddleware)
    # 给 agent 构建长期记忆——让它在下一次对话时还能记得用户的偏好、背景和历史上下文，而不是每次都从零开始。
    #
    # after_agent 之后消息存到哪里
    # 整个链路是：
    #
    # after_agent
    #   └─ queue.add(...)
    #        └─ 内存队列（list[ConversationContext]）
    #             └─ threading.Timer 30秒后触发
    #                  └─ MemoryUpdater.update_memory()
    #                       └─ LLM.invoke(MEMORY_UPDATE_PROMPT)  ← 调 LLM 提取记忆
    #                            └─ _apply_updates()
    #                                 └─ memory.json 文件
    #                                      路径: .deer-flow/users/{user_id}/memory.json
    # 最终落地在本地 JSON 文件，按用户隔离存储。
    middlewares.append(MemoryMiddleware(agent_name=agent_name, memory_config=resolved_app_config.memory))

    # Add ViewImageMiddleware only if the current model supports vision.
    # Use the resolved runtime model_name from make_lead_agent to avoid stale config values.
    model_config = resolved_app_config.get_model_config(model_name) if model_name else None
    # 模型是否支持图片输入
    if model_config is not None and model_config.supports_vision:
        # 在 LLM 调用前，把 view_image 工具读取到的图片 base64 数据注入到对话里，让 LLM 真正能"看到"图片内容。
        #
        # 为什么需要这个中间件
        # view_image 工具执行后，图片数据存在 state.viewed_images 里，但 ToolMessage 里只有文字内容（"图片已加载"之类）。LLM 收到的是文字结果，看不到图片本身。
        #
        # 这个中间件在下次调用 LLM 前把实际图片内容（base64）构造成一条 HumanMessage 注入进去，LLM 才能真正分析图片。
        #
        # 触发条件（_should_inject_image_message）
        # 四个条件全部满足才注入：
        #
        # 最后一条 AIMessage 包含 view_image tool_call
        # 该 AIMessage 的所有 tool_call（不只是 view_image）都已有对应 ToolMessage
        # state.viewed_images 里有图片数据
        # 那条 AIMessage 后面还没有注入过图片消息（防重复）
        # 注入的消息格式
        # HumanMessage(
        #     content=[
        #         {"type": "text", "text": "Here are the images you've viewed:"},
        #         {"type": "text", "text": "\n- **/mnt/user-data/uploads/chart.png** (image/png)"},
        #         {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0K..."}},
        #         # 多张图片继续追加...
        #     ],
        #     additional_kwargs={"hide_from_ui": True}  # 不显示在前端
        # )
        # 用多模态的 image_url 格式，让支持 vision 的 LLM 直接看到图片像素内容。hide_from_ui=True 保证这条消息只是给 LLM 看的内部上下文，不出现在用户界面上。
        #
        # 完整时序
        # 用户: "帮我分析这张图表"
        # Turn 1:
        #   AIMessage(tool_calls=[view_image("/mnt/user-data/uploads/chart.png")])
        #   ToolMessage("图片已读取，base64 存入 viewed_images")
        # Turn 2（before_model 触发）:
        #   ViewImageMiddleware 检测到上轮有已完成的 view_image
        #   → 注入 HumanMessage(content=[text + image_url base64])
        #   → LLM 调用时收到图片内容
        #   → AIMessage("这张图表显示了...")
        middlewares.append(ViewImageMiddleware())

    # Hide deferred tool schemas from model binding until tool_search promotes them.
    # The deferred set + catalog hash come from the build-time setup (assembled
    # after tool-policy filtering); promotion is read from graph state.
    if deferred_setup is not None and deferred_setup.deferred_names:
        from deerflow.agents.middlewares.deferred_tool_filter_middleware import DeferredToolFilterMiddleware
        # 作用是在model调用时，将deferred且没有promote的工具剔除，不让llm看到。让llm使用tool_search来搜索deferred tools
        # 在tool调用时，校验llm是否直接调用了deferred但没有promote的工具，返回一个错误的ToolMessage
        middlewares.append(DeferredToolFilterMiddleware(deferred_setup.deferred_names, deferred_setup.catalog_hash))

    # Add SubagentLimitMiddleware to truncate excess parallel task calls
    subagent_enabled = cfg.get("subagent_enabled", False)
    if subagent_enabled:
        max_concurrent_subagents = cfg.get("max_concurrent_subagents", 3)
        # 限制subagent的task tool call的调用数量
        middlewares.append(SubagentLimitMiddleware(max_concurrent=max_concurrent_subagents))

    # LoopDetectionMiddleware — detect and break repetitive tool call loops
    loop_detection_config = resolved_app_config.loop_detection
    if loop_detection_config.enabled:
        # LoopDetectionMiddleware 是一个 P0 级安全守卫，防止 agent 陷入无限工具调用循环直到 LangGraph 递归限制崩溃
        # 处理方式分为两种：添加HumanMessage告警或者直接停止agent loop
        # 判断方式分为两种：一种是对tool_calls计算hash，一种是统计同一个工具的调用频率
        middlewares.append(LoopDetectionMiddleware.from_config(loop_detection_config))

    # Inject custom middlewares before ClarificationMiddleware
    if custom_middlewares:
        middlewares.extend(custom_middlewares)

    # SafetyFinishReasonMiddleware — suppress tool execution when the provider
    # safety-terminated the response. Registered after custom middlewares so
    # that LangChain's reverse-order after_model dispatch runs Safety first;
    # cleared tool_calls then flow through Loop/Subagent accounting without
    # firing extra alarms. See safety_finish_reason_middleware.py docstring.
    safety_config = resolved_app_config.safety_finish_reason
    if safety_config.enabled:
        # SafetyFinishReasonMiddleware 解决的是一个特定的 provider 行为问题：模型因安全拦截而中途截断响应时，仍可能携带半截的 tool_calls，如果任由 LangChain 执行，会导致参数不完整的工具被调用，然后循环失败。
        #
        # 核心问题场景（issue #3028）
        # 模型生成 write_file(content="# 报告\n\n## 第一章...")
        #   ↓ 安全过滤触发，生成中断
        # AIMessage {
        #   tool_calls: [write_file(content="# 报告\n\n## 第一章...【截断】")]
        #   finish_reason: "content_filter"   ← 安全信号
        # }
        #   ↓ 没有本中间件时
        # LangChain ToolRouter 看到 tool_calls 非空 → 去执行
        #   → 写入截断的文件
        #   → agent 看到残缺文件，尝试修复
        #   → 再次触发安全过滤
        #   → 无限循环
        # 处理流程
        # after_model 触发
        #     ↓
        # 最后一条消息是 AIMessage 且有 tool_calls？
        #     ↓ 是
        # 逐个 detector 检测 → 有命中？
        #     ↓ 是
        # _build_suppressed_message：
        #   1. 清空 tool_calls（结构化 + additional_kwargs 原始元数据）
        #   2. 向 content 追加用户可读解释文字
        #   3. 保留原始 finish_reason（content_filter/refusal/SAFETY 不改写）
        #   4. 在 additional_kwargs["safety_termination"] 写入可观测性记录
        #     ↓
        # _emit_event → 向 SSE 推送 safety_termination 事件（前端消除 "工具启动中..." 占位）
        # _record_audit_event → 写入 RunJournal 持久化（只记录工具名/id，不记录参数内容）
        #     ↓
        # 返回改写后的 AIMessage，tool_calls 已清空 → 工具不会被执行
        # 三个内置 Detector
        # Detector	检测字段	触发值	覆盖 Provider
        # OpenAICompatibleContentFilterDetector	finish_reason	content_filter（可扩展）	OpenAI、Azure、Moonshot、DeepSeek、vLLM 等
        # AnthropicRefusalDetector	stop_reason	refusal	Claude
        # GeminiSafetyDetector	finish_reason	SAFETY/BLOCKLIST/PROHIBITED_CONTENT 等8种	Gemini/Vertex AI
        # 检测字段同时查 response_metadata 和 additional_kwargs，兼容新旧版本 LangChain 适配器的不同存放位置。
        middlewares.append(SafetyFinishReasonMiddleware.from_config(safety_config))

    # ClarificationMiddleware should always be last
    middlewares.append(ClarificationMiddleware())
    return middlewares


def _available_skill_names(agent_config, is_bootstrap: bool) -> set[str] | None:
    if is_bootstrap:
        return {"bootstrap"}
    if agent_config and agent_config.skills is not None:
        return set(agent_config.skills)
    return None


def _load_enabled_skills_for_tool_policy(available_skills: set[str] | None, *, app_config: AppConfig) -> list[Skill]:
    try:
        from deerflow.agents.lead_agent.prompt import get_enabled_skills_for_config

        skills = get_enabled_skills_for_config(app_config)
    except Exception:
        logger.exception("Failed to load skills for allowed-tools policy")
        raise

    if available_skills is None:
        return skills
    return [skill for skill in skills if skill.name in available_skills]


def make_lead_agent(config: RunnableConfig):
    """LangGraph graph factory; keep the signature compatible with LangGraph Server."""
    runtime_config = _get_runtime_config(config)
    runtime_app_config = runtime_config.get("app_config")
    return _make_lead_agent(config, app_config=runtime_app_config or get_app_config())


def _make_lead_agent(config: RunnableConfig, *, app_config: AppConfig):
    # Lazy import to avoid circular dependency
    from deerflow.tools import get_available_tools
    from deerflow.tools.builtins import setup_agent, update_agent
    from deerflow.tools.builtins.tool_search import assemble_deferred_tools

    cfg = _get_runtime_config(config)
    resolved_app_config = app_config

    thinking_enabled = cfg.get("thinking_enabled", True)
    reasoning_effort = cfg.get("reasoning_effort", None)
    requested_model_name: str | None = cfg.get("model_name") or cfg.get("model")
    is_plan_mode = cfg.get("is_plan_mode", False)
    subagent_enabled = cfg.get("subagent_enabled", False)
    max_concurrent_subagents = cfg.get("max_concurrent_subagents", 3)
    is_bootstrap = cfg.get("is_bootstrap", False)
    agent_name = validate_agent_name(cfg.get("agent_name"))

    agent_config = load_agent_config(agent_name) if not is_bootstrap else None
    available_skills = _available_skill_names(agent_config, is_bootstrap)
    # Custom agent model from agent config (if any), or None to let _resolve_model_name pick the default
    agent_model_name = agent_config.model if agent_config and agent_config.model else None

    # Final model name resolution: request → agent config → global default, with fallback for unknown names
    model_name = _resolve_model_name(requested_model_name or agent_model_name, app_config=resolved_app_config)

    model_config = resolved_app_config.get_model_config(model_name)

    if model_config is None:
        raise ValueError("No chat model could be resolved. Please configure at least one model in config.yaml or provide a valid 'model_name'/'model' in the request.")
    if thinking_enabled and not model_config.supports_thinking:
        logger.warning(f"Thinking mode is enabled but model '{model_name}' does not support it; fallback to non-thinking mode.")
        thinking_enabled = False

    logger.info(
        "Create Agent(%s) -> thinking_enabled: %s, reasoning_effort: %s, model_name: %s, is_plan_mode: %s, subagent_enabled: %s, max_concurrent_subagents: %s",
        agent_name or "default",
        thinking_enabled,
        reasoning_effort,
        model_name,
        is_plan_mode,
        subagent_enabled,
        max_concurrent_subagents,
    )

    # Inject run metadata for LangSmith trace tagging
    if "metadata" not in config:
        config["metadata"] = {}

    config["metadata"].update(
        {
            "agent_name": agent_name or "default",
            "model_name": model_name or "default",
            "thinking_enabled": thinking_enabled,
            "reasoning_effort": reasoning_effort,
            "is_plan_mode": is_plan_mode,
            "subagent_enabled": subagent_enabled,
            "tool_groups": agent_config.tool_groups if agent_config else None,
            "available_skills": sorted(available_skills) if available_skills is not None else None,
        }
    )

    # Inject tracing callbacks at the graph invocation root so a single LangGraph
    # run produces one trace with all node / LLM / tool calls as child spans,
    # AND so the Langfuse handler sees ``on_chain_start(parent_run_id=None)`` and
    # actually propagates ``langfuse_session_id`` / ``langfuse_user_id`` from
    # ``config["metadata"]`` onto the trace. Without root-level attachment the
    # model is a nested observation and the handler strips ``langfuse_*`` keys.
    tracing_callbacks = build_tracing_callbacks()
    if tracing_callbacks:
        existing = config.get("callbacks") or []
        if not isinstance(existing, list):
            existing = list(existing)
        config["callbacks"] = [*existing, *tracing_callbacks]

    skills_for_tool_policy = _load_enabled_skills_for_tool_policy(available_skills, app_config=resolved_app_config)

    if is_bootstrap:
        # Special bootstrap agent with minimal prompt for initial custom agent creation flow
        raw_tools = get_available_tools(model_name=model_name, subagent_enabled=subagent_enabled, app_config=resolved_app_config) + [setup_agent]
        filtered = filter_tools_by_skill_allowed_tools(raw_tools, skills_for_tool_policy)
        final_tools, setup = assemble_deferred_tools(filtered, enabled=resolved_app_config.tool_search.enabled)
        return create_agent(
            model=create_chat_model(name=model_name, thinking_enabled=thinking_enabled, app_config=resolved_app_config, attach_tracing=False),
            tools=final_tools,
            middleware=build_middlewares(config, model_name=model_name, app_config=resolved_app_config, deferred_setup=setup),
            system_prompt=apply_prompt_template(
                subagent_enabled=subagent_enabled,
                max_concurrent_subagents=max_concurrent_subagents,
                available_skills=set(["bootstrap"]),
                app_config=resolved_app_config,
                deferred_names=setup.deferred_names,
            ),
            state_schema=ThreadState,
        )

    # Custom agents can update their own SOUL.md / config via update_agent.
    # The default agent (no agent_name) does not see this tool.
    extra_tools = [update_agent] if agent_name else []
    # Default lead agent (unchanged behavior)
    raw_tools = get_available_tools(model_name=model_name, groups=agent_config.tool_groups if agent_config else None, subagent_enabled=subagent_enabled, app_config=resolved_app_config)
    filtered = filter_tools_by_skill_allowed_tools(raw_tools + extra_tools, skills_for_tool_policy)
    final_tools, setup = assemble_deferred_tools(filtered, enabled=resolved_app_config.tool_search.enabled)
    return create_agent(
        model=create_chat_model(name=model_name, thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort, app_config=resolved_app_config, attach_tracing=False),
        tools=final_tools,
        middleware=build_middlewares(config, model_name=model_name, agent_name=agent_name, app_config=resolved_app_config, deferred_setup=setup),
        system_prompt=apply_prompt_template(
            subagent_enabled=subagent_enabled,
            max_concurrent_subagents=max_concurrent_subagents,
            agent_name=agent_name,
            available_skills=set(agent_config.skills) if agent_config and agent_config.skills is not None else None,
            app_config=resolved_app_config,
            deferred_names=setup.deferred_names,
        ),
        state_schema=ThreadState,
    )

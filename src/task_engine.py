"""
MEL AI AGENT - Task Engine
===========================
Small deterministic execution core for multi-step runtime tasks (research,
file work, calendar). Not a planner: workflows (agent_runtime.py) decide the
steps in plain code; this module only enforces the rules every tool call
must obey:

- Unknown tools are rejected; specialists may only call tools listed in
  their AgentDefinition.allowed_tools (multi_agent.py), checked per call.
- Consequential tools never run without an explicit approval, and an
  approval is consumed before the tool runs so a repeated "yes" can't fire
  it twice. Idempotency keys block duplicate writes within a session.
- Each task has a hard tool-call budget (default 5), per-call timeouts, and
  retries for read-only tools only (writes are never auto-retried).
- Tools can't re-enter the engine for the same task (no recursion).
- Cancellation is cooperative: checked before every call.

State is per session_id and deliberately compact (summaries and source
lists, never raw documents) so follow-ups stay cheap.
"""

import asyncio
import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger("task_engine")

DEFAULT_MAX_TOOL_CALLS = 5
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_READ_RETRIES = 1


class TaskStatus(str, Enum):
    PROPOSED = "proposed"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Risk(str, Enum):
    READ = "read"                    # no side effects
    WRITE_NEW = "write_new"          # creates new local files inside approved roots, never overwrites
    CONSEQUENTIAL = "consequential"  # external/irreversible changes -- approval required


class TaskEngineError(Exception):
    pass


class UnauthorizedToolError(TaskEngineError):
    pass


class ApprovalRequiredError(TaskEngineError):
    pass


class BudgetExceededError(TaskEngineError):
    pass


class TaskCancelledError(TaskEngineError):
    pass


class DuplicateWriteError(TaskEngineError):
    pass


class ToolExecutionError(TaskEngineError):
    pass


class InvalidToolResponseError(ToolExecutionError):
    pass


@dataclass
class ToolSpec:
    name: str
    handler: Callable[[dict], Awaitable[Any]]
    risk: Risk = Risk.READ
    description: str = ""
    timeout: float | None = None
    # Optional result check; return False (or raise) to reject a malformed response.
    validate: Callable[[Any], bool] | None = None


@dataclass
class AgentTask:
    id: str
    session_id: str
    kind: str
    specialist: str
    goal: str
    plan: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PROPOSED
    steps: list[dict] = field(default_factory=list)
    tool_calls: int = 0
    cancelled: bool = False
    error: str = ""
    result_summary: str = ""
    finished_at: float = 0.0
    _in_tool: bool = False

    def describe(self) -> str:
        return f"{self.kind} task {self.id} is {self.status.value}"


@dataclass
class PendingApproval:
    task_id: str
    tool: str
    params: dict
    description: str
    idempotency_key: str
    created_at: float = field(default_factory=time.time)


@dataclass
class SessionState:
    session_id: str
    tasks: dict[str, AgentTask] = field(default_factory=dict)
    recent: deque = field(default_factory=lambda: deque(maxlen=10))
    active_task_id: str | None = None
    pending: PendingApproval | None = None
    completed_writes: dict[str, Any] = field(default_factory=dict)
    # Compact working memory for follow-ups
    research: dict | None = None
    last_summary: dict | None = None
    last_file: str | None = None

    @property
    def active_task(self) -> AgentTask | None:
        return self.tasks.get(self.active_task_id) if self.active_task_id else None


class ExecutionPolicy:
    """Specialist -> allowed tool names. Built from the existing specialist
    registry so multi_agent.AgentDefinition.allowed_tools is the single
    source of truth, enforced at call time rather than only at selection."""

    def __init__(self, permissions: dict[str, set[str]]):
        self.permissions = {k: set(v) for k, v in permissions.items()}

    @classmethod
    def from_registry(cls, registry) -> "ExecutionPolicy":
        return cls({a.name: set(a.allowed_tools) for a in registry.all()})

    def check(self, specialist: str, tool_name: str):
        if tool_name not in self.permissions.get(specialist, set()):
            raise UnauthorizedToolError(f"Specialist '{specialist}' is not allowed to use '{tool_name}'")


class TaskEngine:
    def __init__(self, policy: ExecutionPolicy, tools: list[ToolSpec] | None = None,
                 max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
                 default_timeout: float = DEFAULT_TIMEOUT_S,
                 read_retries: int = DEFAULT_READ_RETRIES):
        self.policy = policy
        self.tools: dict[str, ToolSpec] = {}
        self.max_tool_calls = max_tool_calls
        self.default_timeout = default_timeout
        self.read_retries = read_retries
        self._sessions: dict[str, SessionState] = {}
        self._ids = itertools.count(1)
        for spec in tools or []:
            self.register(spec)

    # ── registry / sessions ──
    def register(self, spec: ToolSpec):
        self.tools[spec.name] = spec

    def session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    # ── task lifecycle ──
    def start_task(self, session_id: str, kind: str, specialist: str, goal: str,
                   plan: list[str] | None = None) -> AgentTask:
        state = self.session(session_id)
        active = state.active_task
        if active and active.status == TaskStatus.RUNNING:
            raise TaskEngineError(f"Another task is still running ({active.id})")
        task = AgentTask(id=f"t{next(self._ids)}", session_id=session_id, kind=kind,
                         specialist=specialist, goal=goal[:200], plan=list(plan or []),
                         status=TaskStatus.RUNNING)
        state.tasks[task.id] = task
        state.recent.append(task.id)
        state.active_task_id = task.id
        # Bound memory: forget tasks that fell out of the recent window
        for tid in list(state.tasks):
            if tid not in state.recent:
                del state.tasks[tid]
        logger.info(f"Task {task.id} started: kind={kind} specialist={specialist}")
        return task

    def finish(self, task: AgentTask, status: TaskStatus, summary: str = "", error: str = ""):
        if task.status == TaskStatus.CANCELLED:
            return
        task.status = status
        task.result_summary = summary[:300]
        task.error = error[:300]
        task.finished_at = time.time()
        logger.info(f"Task {task.id} -> {status.value}")

    @staticmethod
    def check_cancelled(task: AgentTask):
        if task.cancelled:
            raise TaskCancelledError(f"Task {task.id} was cancelled")

    def cancel(self, session_id: str) -> AgentTask | None:
        state = self.session(session_id)
        task = state.active_task
        if not task or task.status not in (TaskStatus.RUNNING, TaskStatus.AWAITING_APPROVAL,
                                           TaskStatus.PROPOSED):
            return None
        task.cancelled = True
        task.status = TaskStatus.CANCELLED
        if state.pending and state.pending.task_id == task.id:
            state.pending = None
        return task

    # ── tool execution ──
    async def run_tool(self, task: AgentTask, name: str, params: dict,
                       idempotency_key: str | None = None, approved: bool = False) -> Any:
        if task.cancelled:
            raise TaskCancelledError(f"Task {task.id} was cancelled")
        if task._in_tool:
            raise TaskEngineError("Recursive tool invocation is not allowed")
        spec = self.tools.get(name)
        if spec is None:
            raise UnauthorizedToolError(f"Unrecognized tool '{name}'")
        self.policy.check(task.specialist, name)
        if spec.risk == Risk.CONSEQUENTIAL and not approved:
            raise ApprovalRequiredError(f"'{name}' requires approval")

        state = self.session(task.session_id)
        if idempotency_key and idempotency_key in state.completed_writes:
            raise DuplicateWriteError(f"Already done: {idempotency_key}")

        attempts = 1 + (self.read_retries if spec.risk == Risk.READ else 0)
        timeout = spec.timeout or self.default_timeout
        last_error = ""
        for attempt in range(attempts):
            if task.cancelled:
                raise TaskCancelledError(f"Task {task.id} was cancelled")
            if task.tool_calls >= self.max_tool_calls:
                raise BudgetExceededError(f"Tool budget of {self.max_tool_calls} calls exhausted")
            task.tool_calls += 1
            task._in_tool = True
            try:
                result = await asyncio.wait_for(spec.handler(params), timeout=timeout)
                if spec.validate is not None and not spec.validate(result):
                    raise InvalidToolResponseError(f"'{name}' returned an invalid response")
            except InvalidToolResponseError as e:
                last_error = str(e)
                task.steps.append({"tool": name, "status": "invalid"})
                continue
            except asyncio.TimeoutError:
                last_error = f"'{name}' timed out after {timeout:.0f}s"
                task.steps.append({"tool": name, "status": "timeout"})
                continue
            except TaskEngineError:
                raise
            except (PermissionError, FileNotFoundError, KeyError, ValueError) as e:
                # Deterministic failures (boundary violations, bad params): retrying can't help
                task.steps.append({"tool": name, "status": "error"})
                raise ToolExecutionError(f"'{name}' failed: {e}") from e
            except Exception as e:
                last_error = f"'{name}' failed: {type(e).__name__}: {e}"
                task.steps.append({"tool": name, "status": "error"})
                continue
            finally:
                task._in_tool = False
            task.steps.append({"tool": name, "status": "ok"})
            if idempotency_key:
                state.completed_writes[idempotency_key] = result
            return result

        raise ToolExecutionError(last_error or f"'{name}' failed")

    # ── approvals ──
    def propose(self, task: AgentTask, tool: str, params: dict, description: str,
                idempotency_key: str) -> PendingApproval:
        """Park a consequential call until the user confirms. Validates
        tool + permission now so a request that could never run isn't
        offered for approval."""
        if tool not in self.tools:
            raise UnauthorizedToolError(f"Unrecognized tool '{tool}'")
        self.policy.check(task.specialist, tool)
        state = self.session(task.session_id)
        if idempotency_key in state.completed_writes:
            raise DuplicateWriteError(f"Already done: {idempotency_key}")
        pending = PendingApproval(task_id=task.id, tool=tool, params=dict(params),
                                  description=description, idempotency_key=idempotency_key)
        state.pending = pending
        task.status = TaskStatus.AWAITING_APPROVAL
        return pending

    async def resolve_approval(self, session_id: str, approved: bool) -> tuple[AgentTask | None, Any]:
        """Returns (task, result). The pending entry is cleared *before*
        the tool runs, so a duplicate confirmation finds nothing to run."""
        state = self.session(session_id)
        pending = state.pending
        if pending is None:
            return None, None
        state.pending = None
        task = state.tasks.get(pending.task_id)
        if task is None or task.cancelled:
            return task, None
        if not approved:
            task.status = TaskStatus.CANCELLED
            task.cancelled = True
            return task, None
        task.status = TaskStatus.RUNNING
        result = await self.run_tool(task, pending.tool, pending.params,
                                     idempotency_key=pending.idempotency_key, approved=True)
        return task, result

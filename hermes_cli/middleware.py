"""Hermes middleware contract helpers.

Observer hooks report what happened. Middleware can change what happens by rewriting a request or
wrapping the actual execution callback. Agent-loop call sites and plugins share this vocabulary.
"""

from __future__ import annotations

import logging
import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

OBSERVER_SCHEMA_VERSION = "hermes.observer.v1"
MIDDLEWARE_SCHEMA_VERSION = "hermes.middleware.v1"

TOOL_REQUEST_MIDDLEWARE = "tool_request"
TOOL_EXECUTION_MIDDLEWARE = "tool_execution"
LLM_REQUEST_MIDDLEWARE = "llm_request"
LLM_EXECUTION_MIDDLEWARE = "llm_execution"
FINAL_OUTPUT_MIDDLEWARE = "final_output"

VALID_MIDDLEWARE: set[str] = {
    TOOL_REQUEST_MIDDLEWARE, TOOL_EXECUTION_MIDDLEWARE, LLM_REQUEST_MIDDLEWARE, LLM_EXECUTION_MIDDLEWARE,
    FINAL_OUTPUT_MIDDLEWARE,
}


@dataclass(frozen=True)
class FinalOutputAllow:
    response: str


@dataclass(frozen=True)
class FinalOutputDrop:
    reason_code: str | None = None


@dataclass
class ProtectedTextTurn:
    session_id: str
    turn_id: str
    decisions: Dict[str, FinalOutputAllow | FinalOutputDrop] = field(default_factory=dict)
    authorized_body: str | None = None
    disposition: str | None = None
    fragments: List[str] = field(default_factory=list)


class ProtectedTurnViolation(Exception):
    """A protected request cannot safely proceed to the provider or transcript."""


def supports_protected_text_turn() -> bool:
    return True


def supports_final_output_gate() -> bool:
    return True


def protected_turn(agent: Any) -> ProtectedTextTurn | None:
    state = getattr(agent, "_protected_text_turn", None)
    if (isinstance(state, ProtectedTextTurn)
            and state.session_id == (getattr(agent, "session_id", None) or "")
            and state.turn_id == (getattr(agent, "_current_turn_id", None) or "")):
        return state
    return None


def protected_failure_result(agent: Any, messages: Any, api_calls: int, reason: str) -> Dict[str, Any]:
    """A body-free terminal result. A failure is never an assistant message."""
    state = protected_turn(agent)
    if state is not None:
        state.disposition = "failed_protected_mode"
        state.fragments.clear()
    return {
        "final_response": None, "messages": messages, "api_calls": api_calls,
        "completed": False, "failed": True, "output_disposition": "failed_protected_mode",
        "failure_reason": reason, "error": reason,
    }


def protected_request_has_tools(request: Any) -> bool:
    if not isinstance(request, dict):
        return True
    if request.get("tools") or request.get("functions"):
        return True
    choice = request.get("tool_choice", request.get("function_call"))
    return choice not in (None, "none", "None")


def protected_mode_compatible(agent: Any) -> bool:
    """Reject modes that can emit an attempted answer before final authorization."""
    from agent.verification_stop import verify_on_stop_enabled
    from agent.kanban_stop import kanban_stop_nudge_enabled

    return (
        getattr(agent, "api_mode", None) not in {"codex_app_server", "codex_responses"}
        and not verify_on_stop_enabled()
        and not kanban_stop_nudge_enabled()
        and not bool(getattr(agent, "_turn_file_mutation_paths", None))
    )


def run_final_output_policies(agent: Any, candidate: str) -> FinalOutputAllow | FinalOutputDrop:
    """Authorize the final, transformed body. Unlike execution middleware, failure denies."""
    from hermes_cli.plugins import _delivery_manager

    state = protected_turn(agent)
    if state is None:
        return FinalOutputDrop("protected_turn_stale")
    if not isinstance(candidate, str):
        return FinalOutputDrop("invalid_candidate")
    try:
        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    except Exception:
        logger.error("protected final-output candidate could not be hashed")
        return FinalOutputDrop("invalid_candidate")
    cached = state.decisions.get(digest)
    if cached is not None:
        return cached
    try:
        callbacks = list(_delivery_manager()._middleware.get(FINAL_OUTPUT_MIDDLEWARE, ()))
    except BaseException:
        logger.error("protected final-output policy registry unavailable")
        decision = FinalOutputDrop("policy_registry_error")
        state.decisions[digest] = decision
        state.disposition = "dropped"
        return decision
    if not callbacks:
        decision: FinalOutputAllow | FinalOutputDrop = FinalOutputDrop("policy_required")
    else:
        body = candidate
        decision = FinalOutputAllow(body)
        for callback in callbacks:
            try:
                result = callback(**middleware_payload(
                    response=body, session_id=state.session_id, turn_id=state.turn_id,
                    platform=getattr(agent, "platform", "") or "",
                    model=getattr(agent, "model", "") or "",
                    provider=getattr(agent, "provider", "") or "",
                ))
                if isinstance(result, FinalOutputDrop):
                    decision = result
                    break
                if not isinstance(result, FinalOutputAllow) or not isinstance(result.response, str):
                    decision = FinalOutputDrop("invalid_policy_result")
                    break
                body = result.response
                decision = FinalOutputAllow(body)
            except BaseException:
                logger.error("protected final-output policy failed")
                decision = FinalOutputDrop("policy_error")
                break
    state.decisions[digest] = decision
    state.disposition = "allowed" if isinstance(decision, FinalOutputAllow) else "dropped"
    if isinstance(decision, FinalOutputAllow):
        state.authorized_body = decision.response
    return decision


@dataclass
class RequestMiddlewareResult:
    """Result of applying request middleware to a mutable payload."""

    payload: Any
    original_payload: Any
    changed: bool = False
    trace: List[Dict[str, Any]] = field(default_factory=list)


def observer_payload(**kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    return kwargs


def middleware_payload(**kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    kwargs.setdefault("middleware_schema_version", MIDDLEWARE_SCHEMA_VERSION)
    return kwargs


def _safe_copy(payload: Any) -> Any:
    """Deep-copy a request payload, tolerating non-deepcopyable members.

    An LLM request can carry clients/callbacks/file handles; a hard ``deepcopy`` failure would
    otherwise abort the whole request-middleware pass.
    """
    try:
        return deepcopy(payload)
    except Exception as exc:  # pragma: no cover - exercised via fallback test
        logger.debug("deepcopy failed for request payload (%s); using shallow copy", exc)
        return dict(payload) if isinstance(payload, dict) else payload


def _apply_request_chain(
    kind: str, payload_key: str, trace: List[Dict[str, Any]], original: Any, **kwargs: Any
) -> RequestMiddlewareResult:
    """Feed ``kwargs[payload_key]`` through every ``kind`` middleware; each may return ``{payload_key: {...}}``."""
    from hermes_cli.plugins import invoke_middleware

    current = kwargs[payload_key]
    for result in invoke_middleware(kind, **middleware_payload(**kwargs)):
        if not isinstance(result, dict):
            continue
        next_payload = result.get(payload_key)
        if not isinstance(next_payload, dict):
            continue
        current = _safe_copy(next_payload)
        entry = {
            key: value
            for key in ("source", "reason", "name")
            if isinstance(value := result.get(key), str) and value
        }
        trace.append(entry or {"source": "plugin"})
    return RequestMiddlewareResult(
        payload=current, original_payload=original, changed=bool(trace), trace=trace,
    )


def apply_llm_request_middleware(request: Dict[str, Any], **context: Any) -> RequestMiddlewareResult:
    """Apply registered LLM request middleware; ``{"request": {...}}`` replaces the provider kwargs."""
    from hermes_cli.plugins import has_middleware

    if not has_middleware(LLM_REQUEST_MIDDLEWARE):
        return RequestMiddlewareResult(payload=request, original_payload=request)

    original_request = _safe_copy(request)
    return _apply_request_chain(
        LLM_REQUEST_MIDDLEWARE, "request", [], original_request,
        request=_safe_copy(original_request), original_request=original_request, **context,
    )


def apply_tool_request_middleware(
    tool_name: str, args: Dict[str, Any], **context: Any
) -> RequestMiddlewareResult:
    """Apply registered tool request middleware; ``{"args": {...}}`` replaces the effective tool
    arguments before hooks, guardrails, approvals, and execution see them."""
    original_args = _safe_copy(args)
    current_args = _safe_copy(original_args)
    trace: List[Dict[str, Any]] = []

    session_id = str(context.get("session_id") or "")
    skip_relay = bool(context.pop("skip_relay", False))
    if session_id and not skip_relay:
        from agent import relay_runtime

        relay_args = relay_runtime.apply_tool_request_intercepts(
            session_id=session_id, tool_name=tool_name, args=current_args)
        if relay_args != current_args:
            current_args = _safe_copy(relay_args)
            trace.append({"source": "nemo_relay"})

    from hermes_cli.plugins import has_middleware

    if not has_middleware(TOOL_REQUEST_MIDDLEWARE):
        return RequestMiddlewareResult(
            payload=args if not trace else current_args, original_payload=args,
            changed=bool(trace), trace=trace,
        )
    return _apply_request_chain(
        TOOL_REQUEST_MIDDLEWARE, "args", trace, original_args,
        tool_name=tool_name, args=current_args, original_args=original_args, **context,
    )


def run_llm_execution_middleware(
    request: Dict[str, Any], next_call: Callable[[Dict[str, Any]], Any], **context: Any) -> Any:
    """Run provider execution through registered LLM execution middleware."""
    return _run_execution_chain(
        LLM_EXECUTION_MIDDLEWARE, next_call,
        request=request, original_request=context.pop("original_request", request), **context)


def run_tool_execution_middleware(
    tool_name: str, args: Dict[str, Any], next_call: Callable[[Dict[str, Any]], Any], **context: Any,
) -> Any:
    """Run tool execution through registered tool execution middleware."""
    return _run_execution_chain(
        TOOL_EXECUTION_MIDDLEWARE, next_call,
        tool_name=tool_name, args=args, original_args=context.pop("original_args", args), **context)


class _DownstreamExecutionError(Exception):
    """Marks an exception raised BELOW a middleware frame so the frame's own failure handling
    (skip-and-continue) doesn't swallow it."""

    def __init__(self, original: BaseException) -> None:
        super().__init__(str(original))
        self.original = original


def _run_execution_chain(kind: str, terminal_call: Callable[[Any], Any], **kwargs: Any) -> Any:
    from hermes_cli.plugins import _delivery_manager

    payload_key = "request" if "request" in kwargs else "args"
    manager = _delivery_manager()
    callbacks = list(manager._middleware.get(kind, []))
    if not callbacks:
        return terminal_call(kwargs[payload_key])

    def call_at(index: int, payload: Any) -> Any:
        if index >= len(callbacks):
            return terminal_call(payload)

        callback = callbacks[index]
        next_called = False
        next_succeeded = False
        next_result: Any = None

        def next_call(next_payload: Any = None) -> Any:
            nonlocal next_called, next_succeeded, next_result
            # Single-use per frame: a second call would re-run the downstream provider/tool, so it
            # is a contract violation, not a retry.
            if next_called:
                raise RuntimeError(
                    f"Middleware '{kind}' callback "
                    f"{getattr(callback, '__name__', repr(callback))} called "
                    "next_call() more than once; downstream execution is single-use"
                )
            next_called = True
            try:
                next_result = call_at(index + 1, payload if next_payload is None else next_payload)
                next_succeeded = True
                return next_result
            except Exception as exc:
                raise _DownstreamExecutionError(exc) from exc

        call_kwargs = middleware_payload(**kwargs)
        call_kwargs[payload_key] = payload
        call_kwargs["next_call"] = next_call
        try:
            return callback(**call_kwargs)
        except _DownstreamExecutionError as exc:
            raise exc.original
        except Exception as exc:
            # Runs once per tool/LLM call: a mis-declared callback fails identically every time,
            # so it goes through the manager's warn-once reporter (#111922).
            manager._report_hook_failure(kind, callback, call_kwargs, exc, surface="Middleware")
            if next_succeeded:
                return next_result
            if next_called:
                raise
            return call_at(index + 1, payload)

    return call_at(0, kwargs[payload_key])


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

API_EXECUTION_MIDDLEWARE = LLM_EXECUTION_MIDDLEWARE

API_REQUEST_MIDDLEWARE = LLM_REQUEST_MIDDLEWARE

def apply_api_request_middleware(
    request: Dict[str, Any],
    **context: Any,
) -> RequestMiddlewareResult:
    """Compatibility wrapper for older ``api_request`` naming."""
    return apply_llm_request_middleware(request, **context)

def run_api_execution_middleware(
    request: Dict[str, Any],
    next_call: Callable[[Dict[str, Any]], Any],
    **context: Any,
) -> Any:
    """Compatibility wrapper for older ``api_execution`` naming."""
    return run_llm_execution_middleware(request, next_call, **context)
# ---- END PLUGIN-COMPAT ----

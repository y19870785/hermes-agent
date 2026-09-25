"""Protected text turns authorize model prose before it becomes a reply or history."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from threading import Event, Thread

import pytest

from hermes_cli.middleware import (
    FINAL_OUTPUT_MIDDLEWARE, LLM_REQUEST_MIDDLEWARE, FinalOutputAllow, FinalOutputDrop, ProtectedTextTurn,
    protected_failure_result, run_final_output_policies,
)


@pytest.fixture()
def policy_manager(monkeypatch):
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    monkeypatch.setattr(manager, "_middleware", {**manager._middleware, FINAL_OUTPUT_MIDDLEWARE: []})
    monkeypatch.setattr(manager, "_discovered", True)
    return manager


@pytest.fixture()
def loop_agent(monkeypatch):
    from run_agent import AIAgent

    tool = {"type": "function", "function": {
        "name": "test_tool", "description": "test", "parameters": {"type": "object", "properties": {}}
    }}
    with (
        patch("model_tools.get_tool_definitions", return_value=[tool]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._has_stream_consumers = lambda: True
    monkeypatch.setattr(agent, "_save_trajectory", lambda *_a, **_k: None)
    monkeypatch.setattr(agent, "_cleanup_task_resources", lambda *_a, **_k: None)
    return agent


def _response(text, *, tool_calls=None, finish_reason="stop"):
    from tests.agent.test_run_agent import _mock_response
    return _mock_response(
        content=text, finish_reason="tool_calls" if tool_calls else finish_reason,
        tool_calls=tool_calls,
    )


def _run(agent, policy_manager, answer, policy, *, provider_response=None):
    policy_manager._middleware[FINAL_OUTPUT_MIDDLEWARE] = [policy]
    agent.client.chat.completions.create.return_value = provider_response or _response(answer)
    snapshots = []
    persistence_spies = (
        patch.object(agent, "_flush_messages_to_session_db", side_effect=lambda rows, *_a: snapshots.append(
            ("flush", [dict(row) for row in rows]))),
        patch.object(agent, "_persist_session", side_effect=lambda rows, *_a: snapshots.append(
            ("persist", [dict(row) for row in rows]))),
    )
    with persistence_spies[0], persistence_spies[1]:
        result = agent.run_conversation("test input", protected_text_turn=True)
    return result, snapshots


def test_formal_entrypoint_rejects_untyped_opt_in(loop_agent):
    with pytest.raises(TypeError, match="protected_text_turn"):
        loop_agent.run_conversation("test input", protected_text_turn="yes")


def test_policy_drop_is_body_free_before_persistence(loop_agent, policy_manager):
    secret = "FORK1B_DROP_SECRET"
    result, snapshots = _run(loop_agent, policy_manager, secret, lambda **_k: FinalOutputDrop("stale"))
    assert result["output_disposition"] == "dropped"
    assert result["final_response"] is None
    assert all(secret not in str(rows) for _, rows in snapshots)
    assert secret not in str(result["messages"])
    assert all(row.get("content") != secret for row in result["messages"])
    assert loop_agent.client.chat.completions.create.call_count == 1
    assert not loop_agent.client.chat.completions.create.call_args.kwargs.get("tools")
    assert getattr(loop_agent, "_protected_text_turn", None) is None


def test_policy_allow_commits_only_authorized_body(loop_agent, policy_manager):
    result, snapshots = _run(
        loop_agent, policy_manager, "RAW_SECRET", lambda **_k: FinalOutputAllow("APPROVED")
    )
    assert result["output_disposition"] == "allowed"
    assert result["final_response"] == "APPROVED"
    assert result["messages"][-1]["content"] == "APPROVED"
    assert any(rows[-1].get("content") == "APPROVED" for _, rows in snapshots)
    assert all("RAW_SECRET" not in str(rows) for _, rows in snapshots)


def test_policy_rewrite_is_not_reauthorized_by_normal_finalizer(loop_agent, policy_manager):
    seen = []
    def policy(response, **_kwargs):
        seen.append(response)
        return FinalOutputAllow("APPROVED") if response == "RAW_SECRET" else FinalOutputDrop("unexpected_retry")
    result, snapshots = _run(loop_agent, policy_manager, "RAW_SECRET", policy)
    assert seen == ["RAW_SECRET"]
    assert result["output_disposition"] == "allowed"
    assert result["final_response"] == "APPROVED"
    assert any(rows[-1].get("content") == "APPROVED" for _, rows in snapshots)


@pytest.mark.parametrize("policy", [lambda **_k: None, lambda **_k: "", lambda **_k: (_ for _ in ()).throw(RuntimeError("deny"))])
def test_invalid_or_failing_policy_denies(loop_agent, policy_manager, policy):
    result, snapshots = _run(loop_agent, policy_manager, "FORK1B_DROP_SECRET", policy)
    assert result["final_response"] is None
    assert result["output_disposition"] == "dropped"
    assert "FORK1B_DROP_SECRET" not in str(result["messages"])
    assert all("FORK1B_DROP_SECRET" not in str(rows) for _, rows in snapshots)


def test_candidate_digest_and_drop_monotonic(policy_manager):
    agent = SimpleNamespace(session_id="s", _current_turn_id="t", platform="cli", model="m", provider="p")
    agent._protected_text_turn = ProtectedTextTurn("s", "t")
    calls = []
    def decide(**kwargs):
        calls.append(kwargs["response"])
        return FinalOutputDrop("stale") if kwargs["response"] == "A" else FinalOutputAllow("B")
    policy_manager._middleware[FINAL_OUTPUT_MIDDLEWARE] = [decide]
    assert isinstance(run_final_output_policies(agent, "A"), FinalOutputDrop)
    assert isinstance(run_final_output_policies(agent, "A"), FinalOutputDrop)
    assert isinstance(run_final_output_policies(agent, "B"), FinalOutputAllow)
    assert calls == ["A", "B"]


def test_rewritten_body_is_a_new_candidate_if_later_recovered(policy_manager):
    agent = SimpleNamespace(session_id="s", _current_turn_id="t", platform="cli", model="m", provider="p")
    agent._protected_text_turn = ProtectedTextTurn("s", "t")
    calls = []
    def decide(**kwargs):
        calls.append(kwargs["response"])
        return FinalOutputAllow("B") if kwargs["response"] == "A" else FinalOutputDrop("changed")
    policy_manager._middleware[FINAL_OUTPUT_MIDDLEWARE] = [decide]
    assert run_final_output_policies(agent, "A") == FinalOutputAllow("B")
    assert run_final_output_policies(agent, "B") == FinalOutputDrop("changed")
    assert calls == ["A", "B"]


def test_request_middleware_cannot_reinsert_tools(loop_agent, policy_manager):
    policy_manager._middleware[LLM_REQUEST_MIDDLEWARE] = [
        lambda request, **_k: {"request": {**request, "tools": [{"type": "function", "function": {"name": "escape"}}]}}
    ]
    result, snapshots = _run(loop_agent, policy_manager, "FORK1B_DROP_SECRET", lambda **_k: FinalOutputAllow("ok"))
    assert result["output_disposition"] == "failed_protected_mode"
    assert result["final_response"] is None
    loop_agent.client.chat.completions.create.assert_not_called()
    assert all("FORK1B_DROP_SECRET" not in str(rows) for _, rows in snapshots)


def test_unexpected_provider_tool_calls_never_execute(loop_agent, policy_manager):
    from tests.agent.test_run_agent import _mock_tool_call
    response = _response("FORK1B_INTERIM_SECRET", tool_calls=[_mock_tool_call("test_tool")])
    with patch.object(loop_agent, "_execute_tool_calls") as tools:
        result, snapshots = _run(
            loop_agent, policy_manager, "unused", lambda **_k: FinalOutputAllow("ok"),
            provider_response=response,
        )
    assert result["output_disposition"] == "failed_protected_mode"
    assert result["final_response"] is None
    tools.assert_not_called()
    assert all("tool_calls" not in str(rows) for _, rows in snapshots)
    assert "FORK1B_INTERIM_SECRET" not in str(result["messages"])


def test_protected_turn_uses_core_non_streaming_then_restores_ordinary(loop_agent, policy_manager):
    from agent.turn_api_call import _should_stream
    assert _should_stream(loop_agent) is True
    with patch.object(loop_agent, "_interruptible_streaming_api_call", side_effect=AssertionError("streamed")) as stream:
        result, _ = _run(loop_agent, policy_manager, "ok", lambda **_k: FinalOutputAllow("ok"))
    assert result["output_disposition"] == "allowed"
    stream.assert_not_called()
    assert _should_stream(loop_agent) is True


def test_stream_delivery_defense_blocks_text_and_reasoning_after_identity_drift(loop_agent):
    loop_agent.stream_delta_callback = MagicMock()
    loop_agent.reasoning_callback = MagicMock()
    loop_agent._protected_text_turn = ProtectedTextTurn("old-session", "old-turn")
    loop_agent._fire_stream_delta("FORK1B_INTERIM_SECRET")
    loop_agent._fire_reasoning_delta("FORK1B_INTERIM_SECRET")
    loop_agent.stream_delta_callback.assert_not_called()
    loop_agent.reasoning_callback.assert_not_called()


def test_incompatible_mode_refuses_before_provider(loop_agent, policy_manager, monkeypatch):
    monkeypatch.setattr("agent.conversation_loop.protected_mode_compatible", lambda _agent: False)
    result, _ = _run(loop_agent, policy_manager, "unused", lambda **_k: FinalOutputAllow("ok"))
    assert result["output_disposition"] == "failed_protected_mode"
    loop_agent.client.chat.completions.create.assert_not_called()


def test_truncation_continuation_is_internal_until_final_allow(loop_agent, policy_manager):
    loop_agent.client.chat.completions.create.side_effect = [
        _response("FORK1B_TRUNCATION_SECRET", finish_reason="length"),
        _response("done", finish_reason="stop"),
    ]
    seen = []
    policy_manager._middleware[FINAL_OUTPUT_MIDDLEWARE] = [
        lambda response, **_k: (seen.append(response), FinalOutputAllow(response))[1]
    ]
    with patch.object(loop_agent, "_interruptible_streaming_api_call", side_effect=AssertionError("streamed")) as stream:
        result = loop_agent.run_conversation("test input", protected_text_turn=True)
    assert result["output_disposition"] == "allowed"
    assert "FORK1B_TRUNCATION_SECRET" in result["final_response"]
    assert len(seen) >= 1
    assert len([row for row in result["messages"] if row.get("role") == "assistant"]) == 1
    assert loop_agent.client.chat.completions.create.call_count == 2
    assert all(not call.kwargs.get("tools") for call in loop_agent.client.chat.completions.create.call_args_list)
    stream.assert_not_called()


def test_truncation_failure_keeps_fragments_out_of_history(loop_agent, policy_manager):
    loop_agent.client.chat.completions.create.side_effect = [
        _response("FORK1B_TRUNCATION_SECRET", finish_reason="length") for _ in range(4)
    ]
    policy_manager._middleware[FINAL_OUTPUT_MIDDLEWARE] = [lambda response, **_k: FinalOutputAllow(response)]
    result = loop_agent.run_conversation("test input", protected_text_turn=True)
    assert result["output_disposition"] == "failed_protected_mode"
    assert result["final_response"] is None
    assert "FORK1B_TRUNCATION_SECRET" not in str(result["messages"])


def test_failure_surfaces_emit_only_core_status(loop_agent):
    from gateway.run import _normalize_empty_agent_response
    result = protected_failure_result(loop_agent, [{"role": "user", "content": "input"}], 1, "final_output_dropped")
    gateway_status = _normalize_empty_agent_response(result, "", history_len=0)
    assert gateway_status and "FORK1B_DROP_SECRET" not in gateway_status
    assert result["final_response"] is None


def test_missing_final_policy_refuses_before_provider(loop_agent, policy_manager):
    policy_manager._middleware[FINAL_OUTPUT_MIDDLEWARE] = []
    loop_agent.client.chat.completions.create.return_value = _response("FORK1B_DROP_SECRET")
    result = loop_agent.run_conversation("input", protected_text_turn=True)
    assert result["output_disposition"] == "failed_protected_mode"
    loop_agent.client.chat.completions.create.assert_not_called()


def test_late_external_policy_change_drops_completed_response(loop_agent, policy_manager):
    entered, release = Event(), Event()
    allowed = [True]
    def provider(**_kwargs):
        entered.set()
        assert release.wait(5)
        return _response("FORK1B_STALE_SECRET")
    loop_agent.client.chat.completions.create.side_effect = provider
    policy_manager._middleware[FINAL_OUTPUT_MIDDLEWARE] = [
        lambda **_k: FinalOutputAllow("ok") if allowed[0] else FinalOutputDrop("stale")
    ]
    result_box = []
    worker = Thread(target=lambda: result_box.append(loop_agent.run_conversation("input", protected_text_turn=True)))
    worker.start()
    try:
        assert entered.wait(5)
        allowed[0] = False
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    result = result_box[0]
    assert result["output_disposition"] == "dropped"
    assert "FORK1B_STALE_SECRET" not in str(result["messages"])
    assert result["final_response"] is None


def test_session_change_during_provider_call_fails_closed(loop_agent, policy_manager):
    def provider(**_kwargs):
        loop_agent.session_id = "replacement-session"
        return _response("FORK1B_STALE_SECRET")
    loop_agent.client.chat.completions.create.side_effect = provider
    policy_manager._middleware[FINAL_OUTPUT_MIDDLEWARE] = [lambda **_k: FinalOutputAllow("ok")]
    result = loop_agent.run_conversation("input", protected_text_turn=True)
    assert result["output_disposition"] == "failed_protected_mode"
    assert result["final_response"] is None
    assert "FORK1B_STALE_SECRET" not in str(result["messages"])


def test_policy_registry_error_drops_without_candidate_logging(policy_manager, caplog):
    agent = SimpleNamespace(session_id="s", _current_turn_id="t", platform="cli", model="m", provider="p")
    agent._protected_text_turn = ProtectedTextTurn("s", "t")
    with patch("hermes_cli.plugins._delivery_manager", side_effect=RuntimeError("FORK1B_DROP_SECRET")):
        decision = run_final_output_policies(agent, "FORK1B_DROP_SECRET")
    assert decision == FinalOutputDrop("policy_registry_error")
    assert "FORK1B_DROP_SECRET" not in caplog.text


def test_recovery_policy_internal_exception_never_returns_raw_body(policy_manager, monkeypatch):
    from agent.turn_finalizer import finalize_turn
    from tests.agent.test_turn_finalizer_final_response_persistence import FakeAgent
    agent = FakeAgent()
    agent._current_turn_id = "turn"
    agent._protected_text_turn = ProtectedTextTurn(agent.session_id, "turn")
    monkeypatch.setattr("hermes_cli.middleware.run_final_output_policies", lambda *_a: (_ for _ in ()).throw(RuntimeError("failure")))
    result = finalize_turn(
        agent, final_response="FORK1B_STALE_SECRET", api_call_count=1,
        interrupted=False, failed=False, messages=[{"role": "user", "content": "input"}],
        conversation_history=[], effective_task_id="task", turn_id="turn",
        user_message="input", original_user_message="input", _should_review_memory=False,
        _turn_exit_reason="fallback_prior_turn_content", current_turn_user_idx=0,
    )
    assert result["final_response"] is None
    assert result["output_disposition"] == "dropped"
    assert "FORK1B_STALE_SECRET" not in str(result["messages"])
    assert "FORK1B_STALE_SECRET" not in str(agent.persisted_messages)


def test_failed_turn_notice_is_not_in_next_protected_provider_projection(loop_agent, policy_manager):
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND, FAILED_TURN_NOTICE
    policy_manager._middleware[FINAL_OUTPUT_MIDDLEWARE] = [lambda response, **_k: FinalOutputAllow(response)]
    loop_agent.client.chat.completions.create.return_value = _response("ok")
    prior = [
        {"role": "user", "content": "earlier input"},
        {"role": "assistant", "content": FAILED_TURN_NOTICE, "display_kind": FAILED_TURN_DISPLAY_KIND},
    ]
    result = loop_agent.run_conversation("new input", conversation_history=prior, protected_text_turn=True)
    assert result["output_disposition"] == "allowed"
    wire = loop_agent.client.chat.completions.create.call_args.kwargs.get("messages")
    assert FAILED_TURN_NOTICE not in str(wire)


def test_failed_turn_closure_never_repairs_protected_drop(loop_agent):
    from agent.conversation_loop import _close_durable_failed_turn
    rows = [{"role": "user", "content": "input"}]
    db = SimpleNamespace(latest_conversation_role=lambda _sid: "user")
    loop_agent._session_db = db
    loop_agent.session_id = "s"
    with patch.object(loop_agent, "_flush_messages_to_session_db") as flush:
        _close_durable_failed_turn(loop_agent, {
            "messages": rows, "failed": True, "completed": False,
            "output_disposition": "dropped",
        })
    assert rows == [{"role": "user", "content": "input"}]
    flush.assert_not_called()

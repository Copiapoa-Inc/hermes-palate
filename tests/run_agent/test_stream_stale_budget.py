"""Regression tests for the per-call cumulative stream-stall budget (#1290)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest


def _make_anthropic_agent(*, thinking_callback=None):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="claude-test",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        thinking_callback=thinking_callback,
    )
    agent.api_mode = "anthropic_messages"
    agent._anthropic_client = MagicMock()
    agent._anthropic_api_key = "test-anthropic-key"
    agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client
    return agent


def _completed_message():
    return SimpleNamespace(
        content=[],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def test_stale_budget_defaults_to_240_and_honors_override(monkeypatch):
    from agent.chat_completion_helpers import _derive_stream_stale_budget

    monkeypatch.delenv("HERMES_STREAM_STALE_BUDGET", raising=False)
    assert _derive_stream_stale_budget(180.0) == 240.0

    monkeypatch.setenv("HERMES_STREAM_STALE_BUDGET", "75")
    assert _derive_stream_stale_budget(10.0) == 75.0

    assert _derive_stream_stale_budget(10.0, configured_budget=120.0) == 120.0


def test_stale_budget_inherits_larger_per_attempt_timeout(monkeypatch):
    from agent.chat_completion_helpers import _derive_stream_stale_budget

    monkeypatch.setenv("HERMES_STREAM_STALE_BUDGET", "240")

    assert _derive_stream_stale_budget(300.0) == 300.0


def test_cross_turn_stale_breaker_remains_a_secondary_cap(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "5")
    agent = _make_anthropic_agent()
    agent._consecutive_stale_streams = 5

    with pytest.raises(RuntimeError, match="5 consecutive stale attempts"):
        agent._interruptible_streaming_api_call({})

    agent._anthropic_client.messages.stream.assert_not_called()


def test_always_stalled_stream_aborts_near_cumulative_budget(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.4")
    monkeypatch.setenv("HERMES_STREAM_STALE_BUDGET", "0.9")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "3")
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "50")

    notices: list[str] = []
    agent = _make_anthropic_agent(thinking_callback=notices.append)
    releases: list[threading.Event] = []
    worker_exits: list[threading.Event] = []

    def _stream_side_effect(*args, **kwargs):
        release = threading.Event()
        worker_exit = threading.Event()
        releases.append(release)
        worker_exits.append(worker_exit)
        cm = MagicMock()
        stream = MagicMock()

        def _blocking_gen():
            try:
                release.wait(timeout=5.0)
                raise httpx.ConnectError("connection dropped after abort")
                yield
            finally:
                worker_exit.set()

        stream.__iter__ = MagicMock(return_value=_blocking_gen())
        cm.__enter__ = MagicMock(return_value=stream)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    agent._anthropic_client.messages.stream.side_effect = _stream_side_effect
    agent._abort_request_anthropic_client = lambda *a, **k: releases[-1].set()

    started = time.time()
    agent._last_activity_ts = started - 5.0
    with pytest.raises(RuntimeError, match="Provider has been unresponsive.*consecutive stale attempts"):
        agent._interruptible_streaming_api_call({})
    elapsed = time.time() - started

    assert 0.9 <= elapsed < 1.8
    assert 2 <= len(releases) < 4
    assert all(worker_exit.wait(timeout=0.2) for worker_exit in worker_exits)
    assert agent.get_activity_summary()["seconds_since_activity"] >= 5.8
    assert any("reconnecting" in notice for notice in notices)


def test_silent_transport_failures_share_the_same_budget(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.4")
    monkeypatch.setenv("HERMES_STREAM_STALE_BUDGET", "0.65")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "3")
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "50")

    agent = _make_anthropic_agent()
    attempts = 0

    def _stream_side_effect(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        cm = MagicMock()
        stream = MagicMock()

        def _silent_failure():
            time.sleep(0.25)
            raise httpx.ConnectError("silent connection drop")
            yield

        stream.__iter__ = MagicMock(return_value=_silent_failure())
        cm.__enter__ = MagicMock(return_value=stream)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    agent._anthropic_client.messages.stream.side_effect = _stream_side_effect

    started = time.time()
    with pytest.raises(RuntimeError, match="Provider has been unresponsive"):
        agent._interruptible_streaming_api_call({})

    assert 0.65 <= time.time() - started < 1.5
    assert attempts == 3


def test_progressing_stream_can_outlive_stale_budget(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.35")
    monkeypatch.setenv("HERMES_STREAM_STALE_BUDGET", "0.4")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "3")

    agent = _make_anthropic_agent()
    cm = MagicMock()
    stream = MagicMock()

    def _progressing_gen():
        for _ in range(6):
            time.sleep(0.1)
            yield SimpleNamespace(type="ping")

    stream.__iter__ = MagicMock(return_value=_progressing_gen())
    stream.get_final_message = MagicMock(return_value=_completed_message())
    cm.__enter__ = MagicMock(return_value=stream)
    cm.__exit__ = MagicMock(return_value=False)
    agent._anthropic_client.messages.stream.return_value = cm

    started = time.time()
    response = agent._interruptible_streaming_api_call({})

    assert time.time() - started > 0.4
    assert response.stop_reason == "end_turn"
    assert agent._consecutive_stale_streams == 0


def test_attempt_tracker_rejects_accounting_from_superseded_attempt():
    from agent.chat_completion_helpers import _StreamAttemptTracker

    tracker = _StreamAttemptTracker(started_at=0.0)
    first_attempt = tracker.start_attempt(now=0.0)
    snapshot = tracker.snapshot(now=5.0)
    assert snapshot.attempt_id == first_attempt
    assert snapshot.elapsed == 5.0

    second_attempt = tracker.start_attempt(now=5.0)

    assert tracker.account_stall(first_attempt, now=10.0) is None
    assert tracker.record_progress(first_attempt, now=10.0) is None
    assert tracker.account_stall(second_attempt, now=7.0) == 2.0


def test_late_anthropic_event_after_budget_abort_cannot_touch_activity(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.3")
    monkeypatch.setenv("HERMES_STREAM_STALE_BUDGET", "0.3")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "3")
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "50")

    agent = _make_anthropic_agent()
    release = threading.Event()
    late_event_seen = threading.Event()
    worker_exit = threading.Event()
    cm = MagicMock()
    stream = MagicMock()

    def _late_event_gen():
        release.wait(timeout=5.0)
        late_event_seen.set()
        yield SimpleNamespace(type="ping")
        time.sleep(0.4)
        yield SimpleNamespace(type="ping")

    stream.__iter__ = MagicMock(return_value=_late_event_gen())
    stream.get_final_message = MagicMock(return_value=_completed_message())
    cm.__enter__ = MagicMock(return_value=stream)
    cm.__exit__ = MagicMock(side_effect=lambda *a: worker_exit.set() or False)
    agent._anthropic_client.messages.stream.return_value = cm
    agent._abort_request_anthropic_client = lambda *a, **k: release.set()
    agent._last_activity_ts = time.time() - 5.0

    with pytest.raises(RuntimeError, match="Provider has been unresponsive"):
        agent._interruptible_streaming_api_call({})

    assert late_event_seen.wait(timeout=0.2)
    assert worker_exit.is_set()
    assert agent.get_activity_summary()["seconds_since_activity"] >= 5.2


def test_late_openai_chunk_after_budget_abort_cannot_touch_activity(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.3")
    monkeypatch.setenv("HERMES_STREAM_STALE_BUDGET", "0.3")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "3")
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "50")

    agent = _make_anthropic_agent()
    agent.api_mode = "chat_completions"
    request_client = MagicMock()
    release = threading.Event()
    late_chunk_seen = threading.Event()
    worker_exit = threading.Event()

    def _late_chunk_gen():
        try:
            release.wait(timeout=5.0)
            late_chunk_seen.set()
            yield SimpleNamespace(choices=[], model=None, usage=None)
            time.sleep(0.4)
            yield SimpleNamespace(choices=[], model=None, usage=None)
        finally:
            worker_exit.set()

    request_client.chat.completions.create.return_value = _late_chunk_gen()
    agent._create_request_openai_client = lambda *a, **k: request_client
    agent._abort_request_openai_client = lambda *a, **k: release.set()
    agent._close_request_openai_client = lambda *a, **k: worker_exit.set()
    activity: list[str] = []
    original_touch = agent._touch_activity

    def _record_activity(desc):
        activity.append(desc)
        original_touch(desc)

    agent._touch_activity = _record_activity

    with pytest.raises(RuntimeError, match="Provider has been unresponsive"):
        agent._interruptible_streaming_api_call({})

    assert late_chunk_seen.wait(timeout=0.2)
    assert worker_exit.is_set()
    assert "receiving stream response" not in activity


@pytest.mark.parametrize(
    ("stale_timeout", "expected"),
    [(float("inf"), True), (1800.0, False), (1800.1, True), (60.0, False)],
)
def test_initial_wait_refresh_policy(stale_timeout, expected):
    from agent.chat_completion_helpers import _initial_wait_refreshes_activity

    assert _initial_wait_refreshes_activity(stale_timeout) is expected


def test_disabled_stale_timeout_initial_wait_refreshes_activity(monkeypatch):
    from agent import chat_completion_helpers as helpers

    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "inf")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    monkeypatch.setattr(helpers, "_STREAM_WAIT_NOTICE_INTERVAL", 0.05)

    notices: list[str] = []
    agent = _make_anthropic_agent(thinking_callback=notices.append)
    cm = MagicMock()
    stream = MagicMock()

    def _slow_empty_stream():
        time.sleep(0.4)
        return
        yield

    stream.__iter__ = MagicMock(return_value=_slow_empty_stream())
    stream.get_final_message = MagicMock(return_value=_completed_message())
    cm.__enter__ = MagicMock(return_value=stream)
    cm.__exit__ = MagicMock(return_value=False)
    agent._anthropic_client.messages.stream.return_value = cm
    agent._last_activity_ts = time.time() - 60.0

    response = agent._interruptible_streaming_api_call({})

    assert response.stop_reason == "end_turn"
    assert any("waiting on" in notice for notice in notices)
    assert agent.get_activity_summary()["seconds_since_activity"] < 1.0

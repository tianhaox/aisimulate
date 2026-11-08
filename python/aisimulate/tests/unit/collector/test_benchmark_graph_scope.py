# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check native-call ordering for the actual helper, without importing CUDA."""

import ast
import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
COLLECTOR = Path(__file__).resolve().parents[3] / "collector"
SOURCE = COLLECTOR / "helper.py"


def trace_benchmark(
    monkeypatch, *, source=SOURCE, failure=None, explicit=None, eager=False, fallback=False, tracing=False, **extra
):
    """Execute a helper source snapshot; retain graph references like a traceback."""
    from collector.helper import TRACE_CHILD_DIR_ENV, TRACE_MEASURED_RANGE, TRACE_TIMED_RANGE

    events, graphs = [], []
    state = {"stream": "caller", "replays": 0, "fault_fired": False}
    error = RuntimeError(f"{failure} failed")

    def fault(stage):
        if failure == stage and not state["fault_fired"]:
            state["fault_fired"] = True
            events.append(f"raise:{stage}")
            raise error

    class Graph:
        def __init__(self):
            events.append("graph:create")
            fault("graph_create")
            self.resets = 0
            graphs.append(self)

        def replay(self):
            state["replays"] += 1
            events.append("graph:replay")
            fault("replay")
            if state["replays"] == 3:
                fault("timed_replay")

        def reset(self):
            assert state["stream"] == "caller"
            events.append("graph:reset")
            fault("reset")
            self.resets += 1

    @contextmanager
    def capture(graph):
        events.append("capture:begin")
        state["stream"] = "capture"
        fault("capture_begin")
        yield
        events.append("capture:end")
        fault("capture_end")
        state["stream"] = "caller"

    def current_stream():
        events.append("stream:get")
        fault("current_stream")
        return state["stream"]

    def set_stream(stream):
        events.append(f"stream:set:{stream}")
        state["stream"] = stream

    def kernel():
        events.append("kernel")
        fault("initial_warmup")
        if state["stream"] == "capture":
            fault("capture_body")

    class Event:
        def __init__(self, **kwargs):
            events.append("event:create")

        def record(self):
            events.append("event:record")

        def elapsed_time(self, end):
            events.append("event:elapsed")
            return 12.0

    cuda = SimpleNamespace(
        is_available=lambda: True,
        CUDAGraph=Graph,
        graph=capture,
        current_stream=current_stream,
        set_stream=set_stream,
        synchronize=lambda: events.append("synchronize"),
        empty_cache=lambda: events.append("empty_cache"),
        Event=Event,
    )

    @contextmanager
    def record_function(name):
        tag = {TRACE_TIMED_RANGE: "timed", TRACE_MEASURED_RANGE: "measured"}[name]
        events.append(f"{tag}:enter")
        yield
        events.append(f"{tag}:exit")

    profiler = SimpleNamespace(record_function=record_function)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, profiler=profiler))
    if tracing:
        monkeypatch.setenv(TRACE_CHILD_DIR_ENV, "/tmp/children")
    else:
        monkeypatch.delenv(TRACE_CHILD_DIR_ENV, raising=False)
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "benchmark_with_power")
    namespace = {
        "contextmanager": contextmanager,
        "get_device_module": lambda device=None: cuda,
        "logging": logging,
        "os": os,
        "time": time,
        "worker_heartbeat": lambda: events.append("heartbeat"),
        "_cold_l2_delta_ms": lambda device, dev_mod, one_call, repeat_n: (events.append("cold_l2"), one_call(), 0.5)[2],
        "TRACE_CHILD_DIR_ENV": TRACE_CHILD_DIR_ENV,
        "TRACE_TIMED_RANGE": TRACE_TIMED_RANGE,
        "TRACE_MEASURED_RANGE": TRACE_MEASURED_RANGE,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    kwargs = {
        "device": SimpleNamespace(index=0),
        "kernel_func": kernel,
        "num_warmups": 2,
        "num_runs": 3,
        "measure_power": False,
        "power_min_duration": 1.0,
        "use_cuda_graph": not eager,
        "allow_graph_fail": fallback,
    }
    if explicit is not None:
        kwargs["explicit_graph_cleanup"] = explicit
    kwargs.update(extra)
    result, caught = None, None
    try:
        with namespace["benchmark_with_power"](**kwargs) as result:
            events.append("caller:body")
            fault("caller")
    except RuntimeError as exc:
        caught = {"type": type(exc).__name__, "message": str(exc), "original_error": exc is error}
    return {
        "events": events,
        # phase_seconds are wall-clock readings that differ between otherwise identical runs.
        "result": {k: v for k, v in result.items() if k != "phase_seconds"} if result is not None else None,
        "error": caught,
        "stream": state["stream"],
        "resets": [graph.resets for graph in graphs],
    }


@pytest.mark.parametrize("explicit", [False, True])
def test_success_preserves_timing_body_and_scopes_cleanup(monkeypatch, explicit):
    observed = trace_benchmark(monkeypatch, explicit=explicit)
    assert observed["error"] is None
    assert observed["result"]["latency_ms"] == 4.0
    assert observed["result"]["used_cuda_graph"] is True
    assert observed["events"].count("kernel") == 3  # two eager warmups + capture
    assert observed["events"].count("graph:replay") == 5  # two warmups + three runs
    assert observed["resets"] == [int(explicit)]
    assert observed["events"][-3:] == (
        ["caller:body", "graph:reset", "empty_cache"] if explicit else ["caller:body", "synchronize", "empty_cache"]
    )
    assert ("stream:set:caller" in observed["events"]) is explicit


@pytest.mark.parametrize("eager", [False, True])
def test_a_trace_marks_exactly_the_timed_work(monkeypatch, eager):
    """probe.trace_case keeps the kernels inside this range: the warmups, capture and timed runs,
    not the caller's module build before it nor its body after; a collection never opens it."""
    events = trace_benchmark(monkeypatch, eager=eager, tracing=True)["events"]
    assert events.count("timed:enter") == events.count("timed:exit") == 1
    assert events.index("timed:enter") < events.index("kernel")
    assert events.index("event:elapsed") < events.index("timed:exit") < events.index("caller:body")
    assert "timed:enter" not in trace_benchmark(monkeypatch, eager=eager)["events"]
    failed = trace_benchmark(monkeypatch, failure="capture_body", tracing=True)
    assert failed["error"]["original_error"] and failed["events"].count("timed:exit") == 1


@pytest.mark.parametrize("eager", [False, True])
def test_the_measured_range_holds_exactly_the_timed_runs(monkeypatch, eager):
    """Inside the timed range, the runs latency_ms averages and nothing before them: the gates
    check its auxiliary kernels, and a warmup's one-time work is not paid per call."""
    events = trace_benchmark(monkeypatch, eager=eager, tracing=True)["events"]
    assert events.count("measured:enter") == events.count("measured:exit") == 1
    inside = events[events.index("measured:enter") + 1 : events.index("measured:exit")]
    assert inside[0] == "event:record" and inside[-2:] == ["event:record", "synchronize"]
    assert inside.count("kernel" if eager else "graph:replay") == 3  # num_runs, no warmup
    assert events.index("timed:enter") < events.index("measured:enter") < events.index("measured:exit")
    assert events.index("measured:exit") < events.index("timed:exit")
    assert "measured:enter" not in trace_benchmark(monkeypatch, eager=eager)["events"]
    # a timed run that raises still closes the range
    failed = trace_benchmark(monkeypatch, failure="timed_replay", tracing=True)
    assert failed["events"].count("measured:enter") == failed["events"].count("measured:exit") == 1


@pytest.mark.parametrize("failure", ["capture_begin", "capture_body", "capture_end"])
@pytest.mark.parametrize("explicit", [False, True])
def test_capture_failure_lifetime_is_opt_in(monkeypatch, failure, explicit):
    observed = trace_benchmark(monkeypatch, failure=failure, explicit=explicit)
    assert observed["error"]["original_error"] is True
    assert observed["resets"] == [int(explicit)]
    assert ("empty_cache" in observed["events"]) is explicit
    assert observed["stream"] == ("caller" if explicit else "capture")
    assert "graph:replay" not in observed["events"]


@pytest.mark.parametrize("failure", ["capture_begin", "capture_body", "capture_end"])
@pytest.mark.parametrize("explicit", [False, True])
def test_existing_opted_fallback_keeps_its_measurement_policy(monkeypatch, failure, explicit):
    observed = trace_benchmark(monkeypatch, failure=failure, explicit=explicit, fallback=True)
    assert observed["error"] is None
    assert observed["result"]["used_cuda_graph"] is False
    assert observed["resets"] == [int(explicit)]
    assert "graph:replay" not in observed["events"]
    assert observed["events"].index("empty_cache") < observed["events"].index("event:record")


@pytest.mark.parametrize("failure", ["caller", "replay", "timed_replay"])
@pytest.mark.parametrize("explicit", [False, True])
def test_cleanup_after_capture_keeps_caller_and_replay_errors(monkeypatch, failure, explicit):
    observed = trace_benchmark(monkeypatch, failure=failure, explicit=explicit)
    assert observed["error"]["original_error"] is True
    assert observed["resets"] == [int(explicit)]
    assert observed["events"][-2:] == (["graph:reset", "empty_cache"] if explicit else ["synchronize", "empty_cache"])


def test_default_is_identical_to_explicit_false(monkeypatch):
    assert trace_benchmark(monkeypatch) == trace_benchmark(monkeypatch, explicit=False)


def test_eager_warmups_can_be_fewer_than_replay_warmups(monkeypatch):
    observed = trace_benchmark(monkeypatch, num_eager_warmups=1)
    assert observed["events"].count("kernel") == 2  # one eager warmup + capture
    assert observed["events"].count("graph:replay") == 5  # replay warmups stay at num_warmups (2) + three runs
    assert observed["result"]["used_cuda_graph"] is True
    zero = trace_benchmark(monkeypatch, num_eager_warmups=0)
    assert zero["events"].count("kernel") == 1  # capture only
    default = trace_benchmark(monkeypatch)
    assert default["events"].count("kernel") == 3  # num_warmups eager passes + capture, unchanged


def test_a_minimum_warmup_time_keeps_replaying_before_the_timed_runs(monkeypatch):
    default = trace_benchmark(monkeypatch)
    assert default == trace_benchmark(monkeypatch, min_warmup_seconds=0.0)
    warmed = trace_benchmark(monkeypatch, min_warmup_seconds=0.02)
    extra = warmed["events"].count("graph:replay") - default["events"].count("graph:replay")
    assert extra > 0 and extra % 8 == 0  # whole batches of replays
    # The extra replays come before the timed block, which still holds exactly num_runs (3).
    ev = warmed["events"]
    start = ev.index("event:record")
    assert ev[start + 1 : start + 5] == ["graph:replay"] * 3 + ["event:record"]
    assert ev[start - 3] == "synchronize"  # warmup ends synced, then the two timing events
    eager = trace_benchmark(monkeypatch, eager=True, min_warmup_seconds=0.02)
    assert eager["events"].count("kernel") > trace_benchmark(monkeypatch, eager=True)["events"].count("kernel")


def test_every_measurement_beats_the_stall_watchdog_before_touching_the_kernel(monkeypatch):
    graph = trace_benchmark(monkeypatch)
    assert graph["events"].count("heartbeat") == 1
    assert graph["events"].index("heartbeat") < graph["events"].index("kernel")
    eager = trace_benchmark(monkeypatch, eager=True)
    assert eager["events"].count("heartbeat") == 1


def test_eager_does_not_touch_graph_lifetime_in_either_mode(monkeypatch):
    default = trace_benchmark(monkeypatch, eager=True)
    opted = trace_benchmark(monkeypatch, eager=True, explicit=True)
    assert default == opted
    assert default["resets"] == []
    # eager: num_warmups (2) + num_runs (3); the helper no longer repeats the eager warmup after the capture step
    assert default["events"].count("kernel") == 5


def test_only_vllm_mla_module_binds_explicit_graph_cleanup():
    path = COLLECTOR / "vllm/collect_mla_module.py"
    tree = ast.parse(path.read_text())
    binding = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "benchmark_with_power" for target in node.targets)
    )
    calls = []

    def shared_benchmark(**kwargs):
        calls.append(kwargs)
        return "native-context"

    namespace = {"partial": partial, "_benchmark_with_power": shared_benchmark}
    exec(compile(ast.Module(body=[binding], type_ignores=[]), str(path), "exec"), namespace)
    kernel = object()
    assert namespace["benchmark_with_power"](kernel_func=kernel, num_runs=30) == "native-context"
    assert calls == [{"explicit_graph_cleanup": True, "kernel_func": kernel, "num_runs": 30}]
    assert namespace["_benchmark_with_power"] is shared_benchmark


def test_cold_l2_adds_the_per_call_cost_of_a_cleared_l2_after_the_timed_block(monkeypatch):
    """Serving reads a layer's weights from HBM: other layers run between its calls. With cold_l2
    the batched time gains the cold-minus-warm per-call difference, measured after the timed
    block on the same graph (the fake reports 0.5 ms; the fake events report 12 ms)."""
    warm = trace_benchmark(monkeypatch)
    assert "cold_l2" not in warm["events"] and warm["result"]["cold_l2_delta_ms"] is None
    cold = trace_benchmark(monkeypatch, cold_l2=True)
    ev = cold["events"]
    assert cold["result"]["latency_ms"] == 12.0 / 3 + 0.5 and cold["result"]["cold_l2_delta_ms"] == 0.5
    # after the measured block's closing timing event, replaying the same graph
    assert ev.index("cold_l2") > max(i for i, e in enumerate(ev) if e == "event:record")
    assert ev[ev.index("cold_l2") + 1] == "graph:replay"
    # the capture and the measured block are unchanged
    assert ev[: ev.index("cold_l2")] == warm["events"][: ev.index("cold_l2")]


def test_cold_l2_delta_is_cold_minus_warm_medians_and_never_negative():
    import types

    from collector import helper

    clock = [0.0]

    class _Ev:
        def __init__(self, **kw):
            self.t = None

        def record(self):
            self.t = clock[0]

        def elapsed_time(self, other):
            return other.t - self.t

    state = {"cleared": False}
    buf = types.SimpleNamespace(sum=lambda: state.update(cleared=True))
    dev = types.SimpleNamespace(Event=_Ev, synchronize=lambda: None)

    def one_call():
        clock[0] += 2.0 if state["cleared"] else 1.5
        state["cleared"] = False

    orig = helper._l2_flush_buffer
    helper._l2_flush_buffer = lambda device, dev_mod: buf
    try:
        assert helper._cold_l2_delta_ms(None, dev, one_call, repeat_n=1, calls=5) == 0.5
        faster = lambda: clock.__setitem__(0, clock[0] + (1.0 if state.pop("cleared", False) else 1.5))
        state["cleared"] = False
        assert helper._cold_l2_delta_ms(None, dev, faster, repeat_n=1, calls=5) == 0.0
    finally:
        helper._l2_flush_buffer = orig

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native observation must preserve execution while exposing every live DP rank."""
import json

import pytest

from aisimulate import _runtime
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.runner import EngineReplayRunnerFactory, _materialize_engine_execution_spec
from aisimulate.sweeper.replay import ReplayOutputRequirements


def specification():
    return prediction_to_replay_spec(CorePredictionConfig.model_validate({
        "engine": {
            "mode": "aggregated", "backend": "vllm", "model": "example/model",
            "hardware": "h200_sxm", "context_length": 1024,
            "workers": {"aggregated": {
                "parallelism": {"tensor": 1, "attention_data": 2, "moe_expert": 2},
                "scheduler": {"max_batched_tokens": 8, "max_sequences": 2},
                "kv_cache": {"block_size": 4, "capacity": {"type": "fixed", "blocks": 100}},
                "timing": {"type": "fixed", "prefill_ms": 2, "decode_ms": 2},
            }},
        },
        "traffic": {"source": {"type": "synthetic", "input_tokens": 16, "output_tokens": 16},
                    "load": {"type": "concurrency", "concurrency": 4}, "stop": {"requests": 4}},
    }))


def test_native_telemetry_preserves_requests_and_reports_rank_cache():
    runner = EngineReplayRunnerFactory().create(0)
    base = runner.run(specification(), output_requirements=ReplayOutputRequirements(
        include_raw_report=True, capture_per_request=True)).metadata["native_report"]
    observed = runner.run(specification(), output_requirements=ReplayOutputRequirements(
        capture_telemetry=True, capture_per_request=True,
        telemetry_sample_interval_ms=0.5)).metadata["native_report"]
    # Synthetic requests receive fresh UUIDs and are reported in UUID order.
    # Compare identical authored sessions without their transport UUID.
    def canonical(rows):
        return [{k: v for k, v in r.items() if k != "uuid"}
                for r in sorted(rows, key=lambda r: r["session_id"])]
    assert canonical(base["per_request"]) == canonical(observed["per_request"])
    assert base["duration_ms"] == observed["duration_ms"]
    frames = observed["telemetry"]
    assert len(frames) > 2
    assert frames[0]["sampled_at_ms"] == 0
    assert frames[-1]["sampled_at_ms"] == observed["duration_ms"]
    assert [f["sampled_at_ms"] for f in frames] == sorted(f["sampled_at_ms"] for f in frames)
    for f in frames:
        assert {r["dp_rank"] for r in f["decode_scheduler_metrics"]} == {0, 1}
        for r in f["decode_scheduler_metrics"]:
            assert r["active_blocks"] + r["inactive_blocks"] <= r["total_blocks"]
            assert r["physical_cache_usage"] == pytest.approx(
                (r["active_blocks"] + r["inactive_blocks"]) / r["total_blocks"])
    assert max(r["active_blocks"] for f in frames for r in f["decode_scheduler_metrics"]) > 0
    assert all(r["active_blocks"] == 0 for r in frames[-1]["decode_scheduler_metrics"])
    assert any(r["inactive_blocks"] > 0 for r in frames[-1]["decode_scheduler_metrics"])


def test_native_telemetry_accepts_flat_payload_and_rejects_invalid_interval():
    payload = _materialize_engine_execution_spec(specification(), trace_block_size=4, record_per_request=True)
    flat = payload.get("spec", payload).copy()
    flat["requests"] = [{"id": "one", "arrival_time_ms": 0, "input_tokens": 8, "output_tokens": 8}]
    flat["telemetry_sample_interval_ms"] = 1
    out = json.loads(_runtime.run_replay_json(json.dumps(flat)))
    assert out["completed_requests"] == 1 and out["telemetry"]
    for interval in [0, -1, True, "1"]:
        flat["telemetry_sample_interval_ms"] = interval
        with pytest.raises(RuntimeError, match="telemetry"):
            _runtime.run_replay_json(json.dumps(flat))


def test_native_telemetry_streams_without_retaining_frames(tmp_path):
    target = tmp_path / "telemetry.jsonl"
    runner = EngineReplayRunnerFactory().create(0)
    report = runner.run(specification(), output_requirements=ReplayOutputRequirements(
        capture_telemetry=True, telemetry_sample_interval_ms=0.5,
        telemetry_output_path=str(target),
    )).metadata["native_report"]
    frames = [json.loads(line) for line in target.read_text().splitlines()]
    assert report["telemetry"] == []
    assert report["telemetry_stream_samples"] == len(frames)
    assert report["telemetry_output_path"] == str(target)
    assert len(frames) > 2
    assert frames[-1]["completed_requests"] == report["completed_requests"] == 4
    assert {r[1] for r in frames[-1]["decode"]} == {0, 1}
    assert all(r[2] == 0 for r in frames[-1]["decode"])
    # Never silently replace an existing stream.
    with pytest.raises(RuntimeError, match="creating telemetry output"):
        runner.run(specification(), output_requirements=ReplayOutputRequirements(
            capture_telemetry=True, telemetry_output_path=str(target)))


def test_telemetry_path_requires_enabled_capture():
    with pytest.raises(ValueError, match="telemetry_output_path"):
        ReplayOutputRequirements(telemetry_output_path="unused.jsonl")

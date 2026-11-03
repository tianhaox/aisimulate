# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 decode step through vLLM's own engine: a validation probe, not a perf-model table.

Why it exists: the dsv41_module component tables (collector/sglang/collect_dsv41_module.py)
time every component eagerly (``used_cuda_graph=False``), while serving decodes inside CUDA
graphs. Their B200 decode-attention rows are flat from KV 3 to 2049 (0.85-0.94 ms per layer)
and GB200 reads about 1.6x slower than B200 on the same GPU silicon, which points at host launch
cost rather than kernels. This op measures what serving runs and nothing else: vLLM builds
DeepseekV41ForCausalLM through its own loader (dummy weights, the first N layers), allocates its
own KV cache and captures its own decode graphs; the op only feeds token ids and times steps.
The step-time difference between two layer counts is the per-layer decode cost, with graphs and
eager.

One engine per (checkpoint, TP, layer count, mode) times each [batch, kv_len] point:
``generate`` with 1 and with 1 + ``steps`` output tokens for ``batch`` random prompts of
``kv_len`` tokens (prefix caching off, EOS ignored); the step time is their difference over
``steps``, the median of three attempts after one warm-up. KV grows by ``steps`` tokens during
the measured steps; rows record the starting length.

Checkpoint-config overrides, and why the measured layers stay as served:
- ``num_hidden_layers``: the first N layers, in checkpoint order.
- ``engram_layer_ids`` / ``engram_num_embeddings`` emptied: the Engram tables (layers 1 and 14)
  are hundreds of GB; layer 1 is in every layer count, so the per-layer difference excludes it.
- The vision tower is stubbed (image limit 0), as text-only serving does
  (vllm/models/deepseek_v41/nvidia/vl_model.py, DeepseekV41ForCausalLM.__init__ @v0.33.0).

The engine runs in a child process: vLLM starts its own engine-core process, which a collector
worker cannot host. The child writes one JSON result that the worker logs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version as get_version
from pathlib import Path

from collector.helper import _AIC_MODEL_CONFIG_DIR, get_sm_version, log_perf, worker_heartbeat
from collector.op_plans import plan_cases

__compat__ = "vllm==0.33.0"

_PKG_ROOT = Path(__file__).resolve().parents[2]  # python/aisimulate
_ATTEMPTS = 3
_CHILD_TIMEOUT_S = 3600


def get_dsv41_layer_step_test_cases():
    return plan_cases("dsv41_layer_step", "vllm", get_sm_version())


def run_dsv41_layer_step_worker(
    model_path: str,
    tp_size: int,
    layers: int,
    mode: str,
    points,
    steps: int,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    """collect.py worker for one engine row (op_plans._vllm_dsv41_layer_step_rows)."""
    if tp_size != 1:
        raise ValueError(f"tp_size={tp_size}: the probe engine owns one GPU; plan TP1")
    if mode not in ("graph", "eager"):
        raise ValueError(f"unknown mode {mode!r}")
    gpu_id = int(str(device).split(":")[-1]) if ":" in str(device) else 0
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible[gpu_id] if gpu_id < len(visible) else str(gpu_id)
    with tempfile.TemporaryDirectory(prefix="dsv41_step_") as tmp:
        spec = {
            "model_path": model_path,
            "tp_size": tp_size,
            "layers": int(layers),
            "mode": mode,
            "points": [[int(b), int(kv)] for b, kv in points],
            "steps": int(steps),
            "workdir": tmp,
        }
        spec_path, out_path = Path(tmp) / "spec.json", Path(tmp) / "result.json"
        spec_path.write_text(json.dumps(spec))
        log_path = Path(tmp) / "child.log"
        cmd = [sys.executable, "-m", "collector.vllm.collect_dsv41_layer_step", "--spec", str(spec_path)]
        cmd += ["--out", str(out_path)]
        with log_path.open("wb") as log:
            proc = subprocess.Popen(cmd, env=env, cwd=str(_PKG_ROOT), stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + _CHILD_TIMEOUT_S
            while True:
                try:
                    proc.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        proc.kill()
                        proc.wait()
                        break
                    worker_heartbeat()
        print(log_path.read_text(errors="replace")[-20000:])
        result = json.loads(out_path.read_text()) if out_path.exists() else {"points": []}

    logged = 0
    for row in result.get("points", []):
        if row.get("step_ms") is None:
            continue
        logged += log_perf(
            item_list=[
                {
                    "model": model_path,
                    "tp_size": tp_size,
                    "layers": layers,
                    "mode": mode,
                    "batch_size": row["batch_size"],
                    "kv_len": row["kv_len"],
                    "decode_steps": steps,
                    "step_ms": f"{row['step_ms']:.4f}",
                    "step_ms_attempts": json.dumps([round(x, 4) for x in row["attempts_ms"]]),
                    "used_cuda_graph": mode == "graph",
                    "cudagraph_mode": result.get("cudagraph_mode"),
                }
            ],
            framework="VLLM",
            version=result.get("vllm_version") or get_version("vllm"),
            device_name=result.get("device_name") or "unknown",
            op_name="dsv41_layer_step",
            kernel_source="vllm.LLM decode step (DeepseekV41ForCausalLM)",
            perf_filename=perf_filename,
        )
    failed = [row for row in result.get("points", []) if row.get("error")]
    if proc.returncode != 0 or failed or logged < len(points):
        reason = result.get("error") or (failed[0]["error"] if failed else f"child exit {proc.returncode}")
        raise RuntimeError(
            f"dsv41_layer_step {mode} layers={layers}: {logged}/{len(points)} points logged; first failure: {reason}"
        )


def _model_dir(model_path: str, layers: int, workdir: str) -> str:
    """The packaged checkpoint config with the first ``layers`` layers and no Engram tables."""
    src = Path(_AIC_MODEL_CONFIG_DIR) / f"{model_path.replace('/', '--')}_config.json"
    config = json.loads(src.read_text())
    text = config["text_config"]
    if layers > text["num_hidden_layers"]:
        raise ValueError(f"{model_path} has {text['num_hidden_layers']} layers, asked for {layers}")
    text.update(num_hidden_layers=layers, engram_layer_ids=[], engram_num_embeddings=[])
    out = Path(workdir) / "model"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(config, indent=1))
    return str(out)


def _time_generate(llm, sampling_params_cls, prompts: list[dict], tokens: int) -> float:
    params = sampling_params_cls(
        max_tokens=tokens, min_tokens=tokens, ignore_eos=True, temperature=0.0, detokenize=False
    )
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    produced = {len(o.outputs[0].token_ids) for o in outputs}
    if produced != {tokens}:
        raise RuntimeError(f"asked for {tokens} tokens per request, engine produced {sorted(produced)}")
    return elapsed


def _child(spec: dict) -> dict:
    import random

    import torch
    from vllm import LLM, SamplingParams

    points, steps = spec["points"], spec["steps"]
    result: dict = {"points": [], "vllm_version": get_version("vllm"), "device_name": torch.cuda.get_device_name(0)}
    llm = LLM(
        model=_model_dir(spec["model_path"], spec["layers"], spec["workdir"]),
        load_format="dummy",
        tensor_parallel_size=spec["tp_size"],
        enforce_eager=spec["mode"] == "eager",
        max_model_len=max(kv for _b, kv in points) + steps + 64,
        max_num_seqs=max(b for b, _kv in points),
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        limit_mm_per_prompt={"image": 0},
        gpu_memory_utilization=0.85,
        seed=0,
    )
    config = llm.llm_engine.vllm_config.compilation_config
    result["cudagraph_mode"] = str(getattr(config, "cudagraph_mode", None))
    rng = random.Random(0)
    for batch, kv_len in points:
        row = {"batch_size": batch, "kv_len": kv_len}
        try:

            def prompts():
                return [
                    {"prompt_token_ids": [rng.randrange(1000, 120000) for _ in range(kv_len)]} for _ in range(batch)
                ]

            _time_generate(llm, SamplingParams, prompts(), 1 + steps)  # warm-up: graphs, JIT, allocator
            attempts = []
            for _ in range(_ATTEMPTS):
                short = _time_generate(llm, SamplingParams, prompts(), 1)
                long = _time_generate(llm, SamplingParams, prompts(), 1 + steps)
                attempts.append((long - short) / steps * 1e3)
            row.update(step_ms=statistics.median(attempts), attempts_ms=attempts)
        except Exception as exc:  # record the point's failure; the other points still run
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        print(json.dumps(row), flush=True)
        result["points"].append(row)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    try:
        result = _child(spec)
    except Exception as exc:
        result = {"points": [], "error": f"{type(exc).__name__}: {exc}"[:2000]}
        Path(args.out).write_text(json.dumps(result))
        raise
    Path(args.out).write_text(json.dumps(result))
    return 0 if all("error" not in row for row in result["points"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

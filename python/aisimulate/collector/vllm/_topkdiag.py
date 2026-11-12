# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only (not for merge): what vLLM's prefill indexer top-k sees and costs at one point.

ops.top_k_per_row_prefill is wrapped once per process. While a point is armed, its first call prints one
TOPKDIAG line: score statistics inside three rows' windows, the top-k time on the scores as they are and on
the same scores scaled to a peak of 10 (the kernel bins scores as fp16 first).
"""

import json
import time

import torch

_STATE = {"label": None, "done": True}


def _stats(logits, ks, ke, row):
    w = logits[row, int(ks[row]) : int(ke[row])].float()
    f = w[torch.isfinite(w)]
    out = {"row": row, "n": int(w.numel()), "finite": int(f.numel())}
    if f.numel():
        out.update(
            unique=int(torch.unique(f).numel()),
            min=float(f.min()),
            max=float(f.max()),
            abs_median=float(f.abs().median()),
            zeros=int((f == 0).sum()),
        )
    return out


def install() -> None:
    from vllm import _custom_ops as ops

    if getattr(ops, "_aic_topk_diag", False):
        return
    orig = ops.top_k_per_row_prefill

    def wrapper(logits, ks, ke, topk_indices, num_rows, stride0, stride1, topk_tokens, *args, **kwargs):
        if _STATE["done"] or _STATE["label"] is None:
            return orig(logits, ks, ke, topk_indices, num_rows, stride0, stride1, topk_tokens, *args, **kwargs)
        _STATE["done"] = True
        try:
            rows = sorted({0, num_rows // 2, num_rows - 1})
            rec = {
                "label": _STATE["label"],
                "shape": list(logits.shape),
                "num_rows": int(num_rows),
                "topk": int(topk_tokens),
                "rows": [_stats(logits, ks, ke, r) for r in rows],
            }

            def timed(x):
                torch.cuda.synchronize()
                t = time.perf_counter()
                orig(x, ks, ke, topk_indices, num_rows, x.stride(0), x.stride(1), topk_tokens, *args, **kwargs)
                torch.cuda.synchronize()
                return round((time.perf_counter() - t) * 1e3, 3)

            timed(logits)
            rec["topk_ms"] = timed(logits)
            peak = max((max(abs(r.get("min", 0.0)), abs(r.get("max", 0.0))) for r in rec["rows"]), default=0.0)
            if peak > 0:
                scaled = logits * (10.0 / peak)
                timed(scaled)
                rec["topk_ms_peak10"] = timed(scaled)
                rec["scale"] = 10.0 / peak
                del scaled
            print("TOPKDIAG " + json.dumps(rec), flush=True)
        except Exception as e:  # never fail the point
            print(f"TOPKDIAG failed: {e!r}", flush=True)
        return orig(logits, ks, ke, topk_indices, num_rows, stride0, stride1, topk_tokens, *args, **kwargs)

    ops.top_k_per_row_prefill = wrapper
    ops._aic_topk_diag = True


def arm(label: str, seq_len: int, prefix_len: int) -> None:
    """Arm the next prefill top-k call of a selected context point."""
    if prefix_len in (2048, 3072, 4096, 8192, 16384, 32768) and seq_len in (64, 512, 1024):
        install()
        _STATE["label"], _STATE["done"] = label, False
    else:
        _STATE["label"], _STATE["done"] = None, True

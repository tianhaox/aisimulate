# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only (not for merge): what SGLang's C4 indexer top-k sees and costs at selected CSA points.

Two prints, both to stdout, which the parent echoes into the job log:

* ``TOPKDIAG readback``, once per runner after the first history write: one written history
  key read back from the indexer pool.
* ``TOPKDIAG {...}``, once per armed point: the indexer logits inside one query row's candidate
  window and the top-k wall time on those logits, as they are and scaled to a peak of 10 (the
  kernel bins scores as fp16 first).

Nothing here runs inside the timed region. The point line comes from one extra eager forward
after the measured iterations, with the top-k function wrapped for that forward only.
"""

import json
import sys
import time

import torch

_CONTEXT_ISL = (64, 512, 1024)
_CONTEXT_PREFIX = (2048, 4096, 8192, 32768)
_GENERATION_PAST = (2048, 8192, 32768)
# The two top-k functions run_topk_transform calls on the sgl-kernel backend: v2 when
# DSATopKBackend.should_use_topk_v2(), else v1 (C4IndexerBackendMixin.forward_c4_indexer,
# srt/layers/attention/dsv4/indexer.py:948-998 at v0.5.23). The logits they receive are the
# ones run_paged_indexer just computed with deep_gemm.fp8_paged_mqa_logits (:1083-1101).
_TOPK_FUNCTIONS = ("topk_transform_paged_v2", "topk_transform_paged")

_STATE = {"readback": False}


def armed(mode: str, attn_kind: str, batch_size: int, seq_len: int, prefix_len: int) -> bool:
    """CSA, batch 8: context at a few (isl, prefix) points, generation at a few past-KV lengths."""
    if attn_kind != "csa" or batch_size != 8:
        return False
    if mode == "context":
        return seq_len in _CONTEXT_ISL and prefix_len in _CONTEXT_PREFIX
    return seq_len in _GENERATION_PAST


def readback(model_runner, attention_module, forward_batch, history_length: int) -> None:
    """Print one written history key, read back from the pool, once per runner.

    The caller runs this right after _write_index_history returned, so the store call itself
    raised nothing. The key's bytes are located as the store kernel lays them out
    (fused_store_indexer_cache, kernels/jit/csrc/deepseek_v4/store.cuh:147 at v0.5.23): 128
    value bytes at offset * 128 in the key's page, one fp32 scale at page_size * 128 + offset * 4.
    """
    indexer = getattr(attention_module, "indexer", None)
    keys = history_length // 4
    if _STATE["readback"] or indexer is None or keys <= 0:
        return
    _STATE["readback"] = True
    # An asynchronous failure of the store kernel surfaces here and fails the point.
    torch.cuda.synchronize()
    try:
        pool = model_runner.token_to_kv_pool
        page_size = int(pool.get_index_k_page_size(attention_module.compress_ratio))
        head_dim = int(indexer.head_dim)
        buffer = pool.get_index_k_with_scale_buffer(attention_module.layer_id)
        row = forward_batch.req_pool_indices.long()[0]
        tokens = model_runner.req_to_token_pool.req_to_token[row]
        record = {
            "store_ok": True,
            "layer_id": int(attention_module.layer_id),
            "page_size": page_size,
            "buffer_shape": list(buffer.shape),
            "keys_per_request": keys,
            "slots": [],
        }
        # The first and the last history key of the first request.
        for key_index in sorted({0, keys - 1}):
            slot = int(tokens[4 * key_index + 3]) // 4
            page, offset = divmod(slot, page_size)
            key = buffer[page, offset * head_dim : (offset + 1) * head_dim]
            scale = buffer[page, page_size * head_dim + offset * 4 : page_size * head_dim + offset * 4 + 4]
            record["slots"].append(
                {
                    "key": key_index,
                    "slot": slot,
                    "key_nonzero_bytes": int(torch.count_nonzero(key)),
                    "scale_nonzero_bytes": int(torch.count_nonzero(scale)),
                    "scale": float(scale.clone().view(torch.float32)[0]),
                }
            )
        print("TOPKDIAG readback " + json.dumps(record), flush=True)
    except Exception as exc:  # the readback itself never fails the point
        print(f"TOPKDIAG readback failed: {exc!r}", flush=True)


def _measure(orig, name, scores, seq_lens, args, kwargs, history_keys: int) -> dict:
    lengths = seq_lens.reshape(-1)
    row = int(lengths.argmax())  # the query row with the most candidates
    candidates = int(lengths[row])
    window = scores[row, :candidates].float()
    finite = window[torch.isfinite(window)]
    record = {
        "topk_fn": name,
        "topk": int(args[1].shape[-1]),  # out_page_indices
        "logits_shape": list(scores.shape),
        "rows": int(lengths.numel()),
        "row": row,
        "candidates": candidates,
        "finite": int(finite.numel()),
    }
    peak = 0.0
    if finite.numel():
        low, high = float(finite.min()), float(finite.max())
        peak = max(abs(low), abs(high))
        record.update(
            mean=float(finite.mean()),
            std=float(finite.std()) if finite.numel() > 1 else 0.0,
            min=low,
            max=high,
            zeros=int((finite == 0).sum()),
            zeros_in_history=int((window[: min(history_keys, candidates)] == 0).sum()),
            distinct=int(torch.unique(finite).numel()),
        )

    def timed() -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        orig(scores, seq_lens, *args, **kwargs)
        torch.cuda.synchronize()
        return round((time.perf_counter() - start) * 1e3, 4)

    timed()
    record["topk_ms"] = timed()
    if peak > 0:
        # In place, so the scaled scores keep the layout the kernel was handed.
        saved = scores.clone()
        try:
            scores.mul_(10.0 / peak)
            timed()
            record["topk_ms_peak10"] = timed()
            record["scale"] = 10.0 / peak
        finally:
            scores.copy_(saved)
    return record


def run_point(kernel_func, label: dict, history_keys: int) -> None:
    """Run the point's forward once more, eagerly, and print what its first top-k call saw."""
    from sglang.kernels.ops.attention.dsv4 import topk as topk_ops

    seen: dict = {}
    patched = []

    def wrap(name):
        orig = getattr(topk_ops, name)

        def wrapper(scores, seq_lens, *args, **kwargs):
            if not seen:
                seen["topk_fn"] = name
                try:
                    seen.update(_measure(orig, name, scores, seq_lens, args, kwargs, history_keys))
                except Exception as exc:
                    seen["error"] = repr(exc)
            return orig(scores, seq_lens, *args, **kwargs)

        # Every SGLang module that imported the function holds its own reference.
        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").startswith("sglang.") and vars(module).get(name) is orig:
                setattr(module, name, wrapper)
                patched.append((module, name, orig))

    try:
        for name in _TOPK_FUNCTIONS:
            wrap(name)
        with torch.no_grad():
            kernel_func()
        torch.cuda.synchronize()
    except Exception as exc:  # the diagnostic never fails the point
        seen.setdefault("error", repr(exc))
    finally:
        for module, name, orig in patched:
            setattr(module, name, orig)
    if not seen:
        seen["error"] = f"the forward called none of {_TOPK_FUNCTIONS}"
    print("TOPKDIAG " + json.dumps({**label, **seen}), flush=True)

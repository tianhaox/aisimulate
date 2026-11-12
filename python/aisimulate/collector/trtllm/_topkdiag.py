# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only (not for merge): what the DSA indexer's prefill top-k sees and costs at one context point.

TRT-LLM dsa_context rows at prefix 2048 (= index_topk) are 2-2.7x the prefix-3072 rows for bf16/fp8_block GEMMs
and not for nvfp4. One TOPKDIAG line per selected point records, from one eager forward, the time spent in the
indexer's MQA logits and in its top-k, and what the logits and the gathered K cache look like.
"""

import json
import time
import weakref

import torch

_PREFIXES = (1536, 2048, 3072, 4096, 8192, 32768)
_SEQ_LENS = (64, 512, 1024)


def _fill_prefix_k(k_data, k_scale):
    """Give the never-written prefix rows of the gathered K (all-zero bytes) distinct valid content: each takes
    the first half of one written row and the second half of another, and the first one's scale."""
    kb = k_data.view(torch.uint8).reshape(k_data.shape[0], -1)
    zero = (kb == 0).all(dim=1)
    src = (~zero).nonzero().flatten()
    count = int(zero.sum())
    if count == 0 or src.numel() == 0:
        return count, int(src.numel())
    a = src[torch.randint(src.numel(), (count,), device=kb.device)]
    b = src[torch.randint(src.numel(), (count,), device=kb.device)]
    half = kb.shape[1] // 2
    kb[zero] = torch.cat([kb[a, :half], kb[b, half:]], dim=1)
    ks = k_scale.reshape(k_scale.shape[0], -1)
    ks[zero] = ks[a]
    return count, int(src.numel())


def _window_stats(logits, row_starts, row_ends, row):
    lo = int(row_starts[row]) if row_starts is not None else 0
    hi = int(row_ends[row]) if row_ends is not None else logits.shape[-1]
    w = logits[row, lo:hi].float()
    finite = torch.isfinite(w)
    f = w[finite]
    return {
        "row": row,
        "n": int(w.numel()),
        "finite": int(finite.sum()),
        "nan": int(torch.isnan(w).sum()),
        "unique": int(torch.unique(f).numel()) if f.numel() else 0,
        "min": float(f.min()) if f.numel() else None,
        "max": float(f.max()) if f.numel() else None,
        "std": float(f.std()) if f.numel() > 1 else None,
    }


def run(attn_module, position_ids, hidden_states, attn_metadata, model_config, label, seq_len, prefix_len):
    if prefix_len not in _PREFIXES or seq_len not in _SEQ_LENS:
        return
    from tensorrt_llm._torch.utils import get_model_extra_attrs, model_extra_attrs

    indexer = next((m for m in attn_module.modules() if type(m).__name__ == "Indexer"), None)
    if indexer is None:
        print("TOPKDIAG " + json.dumps({"label": label, "error": "no Indexer module"}), flush=True)
        return
    rec = {
        "label": label,
        "use_fp4": bool(getattr(indexer, "use_fp4", False)),
        "top_k_type": type(indexer.top_k).__name__,
        "index_topk": int(getattr(indexer, "index_topk", -1)),
        "skip_indexer_ctx": bool(getattr(attn_metadata, "skip_indexer_for_ctx_reqs", False)),
        "enable_gvr_topk": bool(getattr(attn_metadata, "enable_gvr_topk", False)),
        "use_self_sampling_topk": bool(getattr(attn_metadata, "use_self_sampling_topk", False)),
        "use_cute_dsl_topk": bool(getattr(attn_metadata, "use_cute_dsl_topk", False)),
        "chunks": None
        if getattr(attn_metadata, "indexer_prefill_chunks", None) is None
        else len(attn_metadata.indexer_prefill_chunks),
        "passes": [],
    }
    orig_mqa, orig_topk = indexer._call_mqa_logits, indexer.top_k.forward
    cur = {}

    def mqa(q_data, k_data, k_scale, *args, **kwargs):
        if cur.get("fill"):
            cur["filled_rows"], cur["source_rows"] = _fill_prefix_k(k_data, k_scale)
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = orig_mqa(q_data, k_data, k_scale, *args, **kwargs)
        torch.cuda.synchronize()
        cur["mqa_ms"] = cur.get("mqa_ms", 0.0) + (time.perf_counter() - t) * 1e3
        cur["mqa_calls"] = cur.get("mqa_calls", 0) + 1
        if "k" not in cur:
            kb = k_data.reshape(-1)[:65536]
            ks = k_scale.reshape(-1)[:4096]
            cur["k"] = {
                "k_shape": list(k_data.shape),
                "k_dtype": str(k_data.dtype),
                "k_nonzero_frac": round(float((kb.view(torch.uint8) != 0).float().mean()), 4),
                "k_unique_bytes": int(torch.unique(kb.view(torch.uint8)).numel()),
                "scale_dtype": str(k_scale.dtype),
                "scale_unique": int(torch.unique(ks.reshape(-1).view(torch.uint8)).numel()),
                "q_shape": list(q_data.shape),
            }
        return out

    def topk(logits, output_indices, *args, **kwargs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = orig_topk(logits, output_indices, *args, **kwargs)
        torch.cuda.synchronize()
        cur["topk_ms"] = cur.get("topk_ms", 0.0) + (time.perf_counter() - t) * 1e3
        cur["topk_calls"] = cur.get("topk_calls", 0) + 1
        if "logits" not in cur:
            rs, re = kwargs.get("row_starts"), kwargs.get("row_ends")
            rows = sorted({0, logits.shape[0] // 2, logits.shape[0] - 1})
            cur["logits"] = {
                "shape": list(logits.shape),
                "dtype": str(logits.dtype),
                "rows": [_window_stats(logits, rs, re, r) for r in rows],
            }
        return out

    indexer._call_mqa_logits = mqa
    indexer.top_k.forward = topk
    try:
        with model_extra_attrs(model_config.extra_attrs):
            get_model_extra_attrs()["attention_metadata"] = weakref.ref(attn_metadata)
            with torch.inference_mode():
                for i in range(6):
                    cur.clear()
                    # Passes 0-2 as collected (prefix K never written); passes 3-5 with the prefix K filled.
                    cur["fill"] = i >= 3
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    attn_module.forward(position_ids, hidden_states, attn_metadata)
                    torch.cuda.synchronize()
                    cur["forward_ms"] = (time.perf_counter() - t) * 1e3
                    rec["passes"].append(
                        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cur.items()}
                    )
    finally:
        del indexer._call_mqa_logits
        del indexer.top_k.forward
    print("TOPKDIAG " + json.dumps(rec), flush=True)

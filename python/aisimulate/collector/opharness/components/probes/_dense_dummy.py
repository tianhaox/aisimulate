# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense low-bit dummy weights for the vLLM and SGLang serving probes.

The probes load a checkpoint-free dummy model (load_format=dummy). The two frameworks' dummy
initializers draw floats from U(-1e-3, 1e-3), which rounds FP8 storage to zero or the smallest
subnormal, and leave integer storage (packed FP4, MXFP4, INT4/INT8) untouched (vllm
model_executor/model_loader/weight_utils.py initialize_single_dummy_weight, sglang
srt/model_loader/weight_utils.py initialize_dummy_weights). Real checkpoints are dense, and the
grouped GEMMs time 8-23% faster on zero-like weights from 2048 tokens (B200, DeepSeek-V3 MoE
shapes) and pick other tactics; the collector times dense weights (collector/helper.py
fill_dense_weight_). A probe whose kernel times serve as the timing reference therefore runs
dense low-bit weights too.

``densify(named_tensors)`` refills, after the framework's dummy initializer and before its
post-load processing (repacking, scale conversion), every weight matrix whose storage is FP8 or
an integer: FP8 from N(0, 64) clamped to +-448, 8-bit integers uniform over all bytes, 32-bit
integers uniform over all bits (every bit pattern of these storages is a finite value). Only
tensors named as weight matrices are touched: scales, zero points and the integer index tensors
(g_idx, permutations, expert maps) keep the framework's values. 16/32-bit float weights keep
the framework's U(-1e-3, 1e-3) draw, whose bits are already dense.
"""

from __future__ import annotations

_CHUNK = 1 << 27
_MATRIX_SUFFIXES = ("weight", "weight_packed", "qweight")
_NOT_MATRIX = ("scale", "zero", "g_idx", "perm", "bias", "alpha", "map")


def is_weight_matrix(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return leaf.endswith(_MATRIX_SUFFIXES) and not any(s in leaf for s in _NOT_MATRIX)


def _fill(tensor) -> bool:
    import torch

    if tensor.is_floating_point():
        if tensor.element_size() >= 2:
            return False
        flat = tensor.view(-1)
        for start in range(0, flat.numel(), _CHUNK):
            part = flat[start : start + _CHUNK]
            values = torch.randn(part.shape, dtype=torch.float32, device=part.device).mul_(64.0).clamp_(-448.0, 448.0)
            part.copy_(values.to(part.dtype))
            del values
    elif tensor.dtype == torch.uint8:
        tensor.random_(0, 256)
    elif tensor.dtype == torch.int8:
        tensor.random_(-128, 128)
    elif tensor.dtype == torch.int32:
        tensor.random_(-(2**31), 2**31 - 1)
    else:
        return False
    return True


def densify(named_tensors) -> dict:
    """Refill the low-bit weight matrices of ``named_tensors`` ((name, tensor) pairs) in place;
    returns {"tensors": n, "bytes": total} for the probe record."""
    import torch

    n = total = 0
    with torch.no_grad():
        for name, tensor in named_tensors:
            if tensor is None or tensor.device.type == "meta" or not is_weight_matrix(name):
                continue
            data = tensor.data if hasattr(tensor, "data") else tensor
            if _fill(data):
                n += 1
                total += data.numel() * data.element_size()
    return {"tensors": n, "bytes": total}


def install_vllm(rec: dict) -> None:
    """Wrap vLLM's DummyModelLoader.load_weights: the framework's dummy draw, then densify.
    BaseModelLoader.load_model runs process_weights_after_loading after load_weights
    (vllm model_executor/model_loader/base_loader.py), so the post-load repacking sees the
    dense values. In-process engine only (the probe keeps EngineCore in-process)."""
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

    original = DummyModelLoader.load_weights

    def load_weights(self, model, model_config):
        original(self, model, model_config)
        named = [*model.named_parameters(), *model.named_buffers()]
        stats = densify(named)
        acc = rec.setdefault("probe_dense_weights", {"tensors": 0, "bytes": 0})
        acc["tensors"] += stats["tensors"]
        acc["bytes"] += stats["bytes"]

    DummyModelLoader.load_weights = load_weights


def install_sglang(rec: dict) -> None:
    """Wrap the initialize_dummy_weights SGLang's DummyModelLoader calls (bound in
    srt/model_loader/loader.py), so post_load_weights and the quant methods'
    process_weights_after_loading run on the dense values."""
    import sglang.srt.model_loader.loader as loader

    original = loader.initialize_dummy_weights

    def initialize_dummy_weights(model, *args, **kwargs):
        original(model, *args, **kwargs)
        stats = densify(model.state_dict().items())
        acc = rec.setdefault("probe_dense_weights", {"tensors": 0, "bytes": 0})
        acc["tensors"] += stats["tensors"]
        acc["bytes"] += stats["bytes"]

    loader.initialize_dummy_weights = initialize_dummy_weights

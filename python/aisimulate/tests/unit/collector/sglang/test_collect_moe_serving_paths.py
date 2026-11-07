# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SGLang MoE collector builds and drives the layer the way serving does.

No GPU or SGLang install here: the collector's functions are AST-extracted and
run against recording fakes. Each test pins one serving contract of the
pinned SGLang release that the timed path depends on:

* the runner pin goes through ``get_flags().moe.override`` and the MXFP4
  precision pin through ``get_exec().moe.override``, restored on exit;
* the parallel-topology patch replaces only bindings a module has, and restores
  them;
* TopK is bound to the built experts (``TopK.bind_runner_backend``) with no
  explicit output format, as a model binds it, which is what picks bypassed
  versus materialized routing;
* FlashInfer runners are tuned like serving's warm-up (``autotune_v2`` inside
  the tuning pass), other runners are not tuned;
* the row's kernel source comes from the runner FusedMoE resolved.
"""

import ast
import contextlib
import sys
import types
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
SOURCE_PATH = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_moe.py"


def _load(*names: str, namespace: dict | None = None) -> dict:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
        )
    ]
    loaded = {"contextlib": contextlib, "SimpleNamespace": SimpleNamespace, "Callable": Callable, **(namespace or {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE_PATH), "exec"), loaded)
    return loaded


class _RecordingOverride:
    def __init__(self, **values):
        self.__dict__.update(values)
        self.calls = []

    @contextlib.contextmanager
    def override(self, **kwargs):
        self.calls.append(dict(kwargs))
        saved = {name: getattr(self, name) for name in kwargs}
        self.__dict__.update(kwargs)
        try:
            yield self
        finally:
            self.__dict__.update(saved)


class TestPins:
    @pytest.mark.parametrize("backend", ["triton", "flashinfer_trtllm", "flashinfer_mxfp4"])
    def test_runner_pin_goes_through_moe_flags_and_restores_on_error(self, backend):
        moe_flags = _RecordingOverride(runner_backend=None)
        pin = _load(
            "_pin_moe_runner_backend",
            namespace={"get_flags": lambda: SimpleNamespace(moe=moe_flags), "MoeRunnerBackend": object},
        )["_pin_moe_runner_backend"]

        with pytest.raises(RuntimeError, match="boom"), pin(backend):
            assert moe_flags.runner_backend == backend
            raise RuntimeError("boom")
        assert moe_flags.runner_backend is None
        assert moe_flags.calls == [{"runner_backend": backend}]

    def test_mxfp4_precision_pin_goes_through_the_exec_bag(self):
        exec_moe = _RecordingOverride(flashinfer_mxfp4_moe_precision="default")
        pin = _load(
            "_pin_flashinfer_mxfp4_moe_precision",
            namespace={"get_exec": lambda: SimpleNamespace(moe=exec_moe)},
        )["_pin_flashinfer_mxfp4_moe_precision"]

        with pin(None):
            assert exec_moe.calls == []
        with pin("bf16"):
            assert exec_moe.flashinfer_mxfp4_moe_precision == "bf16"
        assert exec_moe.flashinfer_mxfp4_moe_precision == "default"


def test_parallel_patch_replaces_only_existing_bindings_and_restores():
    original_parallel = object()
    with_both = types.SimpleNamespace(get_parallel=lambda: original_parallel, is_allocation_symmetric=lambda: True)
    symmetric_only = types.SimpleNamespace(is_allocation_symmetric=lambda: True)
    neither = types.SimpleNamespace()
    patch = _load(
        "_patch_framework_moe_parallel",
        namespace={"_PARALLEL_READERS": (with_both, symmetric_only, neither)},
    )["_patch_framework_moe_parallel"]
    before = (with_both.get_parallel, with_both.is_allocation_symmetric, symmetric_only.is_allocation_symmetric)

    with pytest.raises(RuntimeError, match="boom"), patch(moe_tp_size=2, moe_ep_size=4):
        parallel = with_both.get_parallel()
        assert (parallel.moe_tp_size, parallel.moe_ep_size, parallel.moe_ep_rank) == (2, 4, 0)
        assert parallel.tp_group is None
        assert with_both.is_allocation_symmetric() is False
        assert symmetric_only.is_allocation_symmetric() is False
        assert not hasattr(symmetric_only, "get_parallel") and not vars(neither)
        raise RuntimeError("boom")

    assert (with_both.get_parallel, with_both.is_allocation_symmetric, symmetric_only.is_allocation_symmetric) == before


# ---------------------------------------------------------------------------
# _benchmark_moe_layer against recording fakes
# ---------------------------------------------------------------------------
class _FakeLayer:
    def __init__(self, record, **kwargs):
        self.record = record
        self.kwargs = kwargs
        self.runner_backend = SimpleNamespace(value=kwargs["_backend"])
        self.routing_method_type = "resolved-routing"
        self.should_fuse_routed_scaling_factor_in_topk = True
        self.quant_method = SimpleNamespace(
            process_weights_after_loading=lambda layer: record.append(("process_weights", layer))
        )

    def to(self, _device):
        return self

    def named_parameters(self):
        return []

    def parameters(self):
        return []

    def __call__(self, hidden_states, topk_output):
        self.record.append(("forward", topk_output))


class _FakeTopK:
    def __init__(self, record, **kwargs):
        self.record = record
        record.append(("topk_init", kwargs))

    def bind_runner_backend(self, runner_backend, *, routing_method_type=None):
        self.record.append(("bind", runner_backend.value, routing_method_type))

    def __call__(self, hidden_states, router_logits):
        return "topk-output"


def _run_layer_benchmark(monkeypatch, *, moe_backend, capability_major, routing_method_type=None, num_tokens=(4, 8)):
    record = []
    tuned = []
    v2_contexts = []

    @contextlib.contextmanager
    def autotune_v2(**kwargs):
        v2_contexts.append(kwargs)
        yield

    monkeypatch.setitem(sys.modules, "flashinfer", types.SimpleNamespace(autotune_v2=autotune_v2))

    @contextlib.contextmanager
    def flashinfer_autotuned(num_tokens, tune_fn, *, label=""):
        tuned.append((num_tokens, label))
        tune_fn()
        yield True

    @contextlib.contextmanager
    def benchmark_with_power(*, kernel_func, **_kwargs):
        kernel_func()
        yield {"latency_ms": 5.0, "power_stats": None}

    @contextlib.contextmanager
    def no_op(*_args, **_kwargs):
        yield

    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        randn=lambda *shape, **_kwargs: SimpleNamespace(shape=shape, to=lambda **_k: None),
        zeros=lambda *shape, **_kwargs: "zeros",
        no_grad=contextlib.nullcontext,
        cuda=SimpleNamespace(get_device_capability=lambda _device: (capability_major, 0), empty_cache=lambda: None),
    )
    loaded = _load(
        "_benchmark_moe_layer",
        "_SERVING_AUTOTUNED_MOE_RUNNERS",
        namespace={
            "torch": fake_torch,
            "gc": SimpleNamespace(collect=lambda: None),
            "FusedMoE": lambda **kwargs: _FakeLayer(record, _backend=moe_backend, **kwargs),
            "TopK": lambda **kwargs: _FakeTopK(record, **kwargs),
            "MoeRunnerBackend": lambda value: value,
            "RoutingMethodType": {"Renormalize": "Renormalize", "DeepSeekV3": "DeepSeekV3"},
            "_quant_config": lambda *_args, **_kwargs: None,
            "_kernel_source": lambda *_args, **_kwargs: f"sglang_{moe_backend}",
            "_mxfp4_activation_precision": lambda _moe_type: "default",
            "_ensure_writable_flashinfer_cubin_dir": lambda: None,
            "_pin_moe_runner_backend": no_op,
            "_pin_flashinfer_mxfp4_moe_precision": no_op,
            "_patch_framework_moe_parallel": no_op,
            "_fmoe_kernels_mod": SimpleNamespace(_B_DESC_CACHE=SimpleNamespace(clear=lambda: None)),
            "balanced_logits": lambda *_args: SimpleNamespace(to=lambda **_kwargs: "logits"),
            "power_law_logits_v3": lambda *_args: SimpleNamespace(to=lambda **_kwargs: "logits"),
            "benchmark_with_power": benchmark_with_power,
            "flashinfer_autotuned": flashinfer_autotuned,
        },
    )
    rows = []
    source = loaded["_benchmark_moe_layer"](
        moe_type="bfloat16",
        moe_backend=moe_backend,
        num_tokens_list=list(num_tokens),
        on_result=lambda n, latency, _power, kernel_source: rows.append((n, latency, kernel_source)),
        hidden_size=64,
        inter_size=128,
        topk=2,
        num_experts=8,
        moe_tp_size=1,
        moe_ep_size=2,
        distributed="balanced",
        power_law_alpha=0.0,
        activation="silu",
        is_gated=True,
        has_bias=False,
        gemm1_alpha=None,
        gemm1_clamp_limit=None,
        swiglu_limit=None,
        scoring_func="softmax",
        routing_method_type=routing_method_type,
        routed_scaling_factor=None,
        renormalize=True,
        has_correction_bias=False,
        num_expert_group=None,
        topk_group=None,
        apply_router_weight_on_input=False,
        is_fp4_experts=False,
        int4_group_size=128,
        device="cuda:0",
    )
    return source, record, tuned, v2_contexts, rows


@pytest.mark.parametrize(
    ("routing_method_type", "bound_routing"),
    [("Renormalize", "resolved-routing"), (None, None)],
)
def test_topk_is_bound_to_the_built_experts_like_a_model(monkeypatch, routing_method_type, bound_routing):
    source, record, _tuned, _v2, rows = _run_layer_benchmark(
        monkeypatch, moe_backend="flashinfer_trtllm", capability_major=10, routing_method_type=routing_method_type
    )

    events = [event[0] for event in record]
    # One build for the sweep: weights processed once, TopK built and bound once, then forwards.
    assert events.count("process_weights") == 1 and events.count("topk_init") == 1
    assert events.index("process_weights") < events.index("topk_init") < events.index("bind")
    (topk_kwargs,) = [event[1] for event in record if event[0] == "topk_init"]
    assert "output_format" not in topk_kwargs
    assert [event[1:] for event in record if event[0] == "bind"] == [("flashinfer_trtllm", bound_routing)]
    assert source == "sglang_flashinfer_trtllm"
    # Five router inputs per timed call: the row is per forward.
    assert rows == [(4, 1.0, "sglang_flashinfer_trtllm"), (8, 1.0, "sglang_flashinfer_trtllm")]


def test_flashinfer_runners_are_tuned_like_serving_warmup(monkeypatch):
    _source, _record, tuned, v2_contexts, _rows = _run_layer_benchmark(
        monkeypatch, moe_backend="flashinfer_trtllm", capability_major=10
    )

    assert tuned == [(4, "sglang_flashinfer_trtllm"), (8, "sglang_flashinfer_trtllm")]
    assert v2_contexts == [
        {"mode": "tune", "persistent_cache": False, "tuning_buckets": (4,)},
        {"mode": "tune", "persistent_cache": False, "tuning_buckets": (8,)},
    ]


@pytest.mark.parametrize(
    ("moe_backend", "capability_major"),
    [("triton", 10), ("marlin", 9), ("flashinfer_cutlass", 8)],
)
def test_runners_serving_does_not_tune_are_timed_untuned(monkeypatch, moe_backend, capability_major):
    _source, _record, tuned, v2_contexts, rows = _run_layer_benchmark(
        monkeypatch, moe_backend=moe_backend, capability_major=capability_major
    )

    assert tuned == [] and v2_contexts == []
    assert [row[0] for row in rows] == [4, 8]


# ---------------------------------------------------------------------------
# _kernel_source
# ---------------------------------------------------------------------------
def _kernel_source_fn():
    return _load(
        "_kernel_source",
        "_mxfp4_activation_precision",
        "_KERNEL_SOURCE_BY_RUNNER",
        "_KERNEL_SOURCE_BY_MXFP4_LEAF",
        "_KERNEL_SOURCE_BY_FP4_EXPERT_METHOD",
    )["_kernel_source"]


def _layer(method_name: str, runner: str | None, **method_attrs):
    method_cls = type(method_name, (), {})
    method = method_cls()
    method.__dict__.update(method_attrs)
    scheme = method_attrs.pop("scheme", None)
    return SimpleNamespace(
        quant_method=method,
        runner_backend=None if runner is None else SimpleNamespace(value=runner),
        scheme=scheme,
    )


@pytest.mark.parametrize(
    ("moe_type", "method", "backend", "expected"),
    [
        ("bfloat16", "UnquantizedFusedMoEMethod", "triton", "sglang_fused_moe_triton"),
        ("bfloat16", "UnquantizedFusedMoEMethod", "flashinfer_trtllm", "sglang_flashinfer_trtllm_moe"),
        ("bfloat16", "UnquantizedFusedMoEMethod", "flashinfer_cutlass", "sglang_flashinfer_cutlass_moe"),
        ("fp8_block", "Fp8MoEMethod", "flashinfer_trtllm", "sglang_flashinfer_trtllm_moe"),
        ("nvfp4", "ModelOptNvFp4FusedMoEMethod", "flashinfer_cutlass", "sglang_flashinfer_cutlass_moe"),
        ("w4a16_mxfp4", "Mxfp4MoEMethod", "marlin", "sglang_marlin_moe"),
    ],
)
def test_kernel_source_comes_from_the_resolved_runner(moe_type, method, backend, expected):
    layer = _layer(method, backend)
    assert _kernel_source_fn()(layer, moe_type=moe_type, moe_backend=backend, is_fp4_experts=False) == expected


def test_kernel_source_rejects_a_runner_other_than_the_pinned_one():
    layer = _layer("Fp8MoEMethod", "triton")
    with pytest.raises(RuntimeError, match="requested_backend=flashinfer_trtllm.*runner_backend=triton"):
        _kernel_source_fn()(layer, moe_type="fp8_block", moe_backend="flashinfer_trtllm", is_fp4_experts=False)


@pytest.mark.parametrize(
    ("leaf", "moe_type", "precision", "expected"),
    [
        ("trtllm_sm100", "w4a8_mxfp4_mxfp8", "default", "sglang_mxfp4_flashinfer_trtllm_moe"),
        ("trtllm_sm100", "w4a16_mxfp4", "bf16", "sglang_mxfp4_flashinfer_trtllm_moe"),
        # SGLang names the SM120 leaf cutlass_sm120 (mxfp4.py:426-427).
        ("cutlass_sm120", "w4a8_mxfp4_mxfp8", "default", "sglang_flashinfer_cutlass_moe"),
        ("cutlass_sm90", "w4a16_mxfp4", "bf16", "sglang_flashinfer_cutlass_moe"),
    ],
)
def test_mxfp4_leaf_and_precision(leaf, moe_type, precision, expected):
    layer = _layer("Mxfp4MoEMethod", "flashinfer_mxfp4", _fi_kernel=leaf, flashinfer_mxfp4_moe_precision=precision)
    assert (
        _kernel_source_fn()(layer, moe_type=moe_type, moe_backend="flashinfer_mxfp4", is_fp4_experts=False) == expected
    )


def test_mxfp4_precision_mismatch_fails_closed():
    layer = _layer(
        "Mxfp4MoEMethod", "flashinfer_mxfp4", _fi_kernel="trtllm_sm100", flashinfer_mxfp4_moe_precision="default"
    )
    with pytest.raises(RuntimeError, match="activation precision is 'default', expected 'bf16'"):
        _kernel_source_fn()(layer, moe_type="w4a16_mxfp4", moe_backend="flashinfer_mxfp4", is_fp4_experts=False)


@pytest.mark.parametrize(
    ("backend", "scheme", "ok"),
    [
        ("flashinfer_trtllm", "CompressedTensorsMxInt4MoE", True),
        ("marlin", "CompressedTensorsWNA16MoE", True),
        ("flashinfer_trtllm", "CompressedTensorsWNA16MoE", False),
    ],
)
def test_int4_scheme_identity(backend, scheme, ok):
    layer = _layer("CompressedTensorsFusedMoEMethod", backend)
    layer.scheme = type(scheme, (), {})()
    kernel_source = _kernel_source_fn()
    if ok:
        assert kernel_source(layer, moe_type="int4_wo", moe_backend=backend, is_fp4_experts=False).startswith("sglang_")
    else:
        with pytest.raises(RuntimeError, match="INT4-WO built scheme"):
            kernel_source(layer, moe_type="int4_wo", moe_backend=backend, is_fp4_experts=False)


def test_dsv4_fp4_expert_methods_label_by_method():
    layer = _layer("Mxfp4FlashinferTrtllmMoEMethod", "flashinfer_mxfp4", flashinfer_mxfp4_moe_precision="default")
    assert (
        _kernel_source_fn()(layer, moe_type="w4a8_mxfp4_mxfp8", moe_backend="flashinfer_mxfp4", is_fp4_experts=True)
        == "sglang_mxfp4_flashinfer_trtllm_moe"
    )

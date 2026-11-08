# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-surface smoke tests for the three sglang collector modules
(AIC-1762 Task 4c/4d follow-up).

First real-GPU 0.5.17 run: ``collect_moe.py`` died at MODULE IMPORT on every
system with ``AttributeError: module 'sglang.srt.server_args' has no
attribute '_global_server_args'``. The collector classified it and exited 0,
so the jobs looked green with no moe parquet. This was a NON-import
attribute access (a bare module-attribute read inside a module-level ``if``
guard) that the earlier per-symbol audit pattern (import statements +
explicitly-exercised APIs) never covered -- and the ``sys.modules`` fakes
used elsewhere in this test suite for capability-probe functions
(``unittest.mock.MagicMock``-based) could not have caught it either, because
``MagicMock`` auto-creates any attribute access instead of raising.

These tests use STRICT fakes instead: every ``sglang.*`` module is a plain
``types.ModuleType`` populated with only the exact names the collector
touches at import time (module-level, not inside function bodies) --
``types.ModuleType`` raises ``AttributeError`` on anything not explicitly
set, the same failure mode the real framework produces for a genuinely
absent attribute. Each collector's surface is the one of the SGLang release
it is pinned to; a pin bump replaces it.

``torch``/``pkg_resources``/``sgl_kernel`` are faked permissively
(``MagicMock``) -- they are unrelated, always-absent-in-this-venv
dependencies, not the subject of this bug class; strictness is scoped to
the sglang surface specifically, per the actual incident.
"""

import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[4]
COLLECTOR_DIR = REPO_ROOT / "collector"


# ---------------------------------------------------------------------------
# Shared strict-fake-tree infrastructure
# ---------------------------------------------------------------------------
def _build_module_tree(specs: dict[str, dict]) -> dict[str, types.ModuleType]:
    """Build a strict fake module tree: every dotted path in `specs`, plus
    every implied parent prefix, becomes a plain ``types.ModuleType``
    (raises ``AttributeError`` on anything not explicitly set below).
    Verified empirically that pure ``sys.modules`` injection --
    no ``__path__``, no parent-attribute wiring -- is sufficient for both
    ``import a.b.c as x`` and ``from a.b.c import y`` to resolve.
    """
    modules: dict[str, types.ModuleType] = {}

    def ensure(path: str) -> types.ModuleType:
        if path not in modules:
            modules[path] = types.ModuleType(path)
        return modules[path]

    for path, attrs in specs.items():
        parts = path.split(".")
        for i in range(1, len(parts) + 1):
            ensure(".".join(parts[:i]))
        leaf = modules[path]
        for name, value in attrs.items():
            setattr(leaf, name, value)
    return modules


class _Dummy:
    """Stand-in for an sglang class the collector only constructs/references,
    never subclasses or introspects at import time."""

    def __init__(self, *args, **kwargs):
        pass


class _ServerArgsDummy:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _dummy(*args, **kwargs):
    return None


def _raise_unset_server_args():
    raise ValueError("Global server args is not set yet!")


def _install_fake_sglang(monkeypatch, specs: dict[str, dict], *, lenient_extras: tuple[str, ...] = ()) -> None:
    # Tests in the same process may already have imported a real or a
    # differently-shaped fake SGLang tree. Purge the whole namespace so
    # parent-package cache entries cannot leak across cases.
    for name in tuple(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, name)
    for path, module in _build_module_tree(specs).items():
        monkeypatch.setitem(sys.modules, path, module)
    for name in ("torch", "pkg_resources", *lenient_extras):
        monkeypatch.setitem(sys.modules, name, MagicMock())


def _import_fresh(monkeypatch, dotted_name: str, source_path: Path) -> types.ModuleType:
    """Exec `source_path` under module name `dotted_name`, bypassing any
    cached entry, and let import-time exceptions propagate to the caller."""
    monkeypatch.delitem(sys.modules, dotted_name, raising=False)
    spec = importlib.util.spec_from_file_location(dotted_name, source_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, dotted_name, module)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# collect_moe.py -- the module that first broke this way
# ---------------------------------------------------------------------------
# The sglang 0.5.23 names collect_moe.py touches at import time: the
# runtime-context publish and pin primitives, the FusedMoE/TopK layer modules,
# the Triton runner and its config module (observed for tuned-config hits), the
# FlashInfer runner modules and the quant methods whose parallel bindings the
# collector replaces, and the quant configs it builds. The Triton MoE kernels
# live in sglang.kernels.ops.moe (the 0.5.14 srt/layers/moe/moe_runner path is
# gone), and runtime_context has no bare _global_server_args.
_published_moe_server_args: list = []


def _record_moe_publish(server_args, *, role):
    _published_moe_server_args.append((server_args, role))


SGLANG_0523_MOE_SURFACE = {
    "sglang.srt.runtime_context": {
        "get_exec": _dummy,
        "get_flags": _dummy,
        "get_server_args": _raise_unset_server_args,
        "publish": _record_moe_publish,
    },
    "sglang.srt.server_args": {"ServerArgs": _ServerArgsDummy},
    "sglang.kernels.ops.moe.fused_moe_triton_kernels": {"_B_DESC_CACHE": {}},
    "sglang.srt.layers.moe.fused_moe_triton.layer": {"FusedMoE": _Dummy},
    "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass": {},
    "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm": {},
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe": {},
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config": {
        "get_default_config": _dummy,
        "get_moe_configs": _dummy,
    },
    "sglang.srt.layers.moe.token_dispatcher.standard": {},
    "sglang.srt.layers.moe.topk": {"TopK": _Dummy, "TopKOutputFormat": _Dummy},
    "sglang.srt.layers.moe.utils": {"MoeRunnerBackend": _Dummy, "RoutingMethodType": _Dummy},
    "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors": {"CompressedTensorsConfig": _Dummy},
    "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_mxint4_moe": {},
    "sglang.srt.layers.quantization.fp8": {"Fp8Config": _Dummy},
    "sglang.srt.layers.quantization.modelopt_quant": {"ModelOptFp4Config": _Dummy},
    "sglang.srt.layers.quantization.mxfp4": {"Mxfp4Config": _Dummy},
    "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe": {},
}

_COLLECT_MOE_PATH = COLLECTOR_DIR / "sglang" / "collect_moe.py"
_COLLECT_MOE_DOTTED = "collector.sglang.collect_moe"


class TestCollectMoeImportSurface:
    def test_imports_cleanly_against_0523_shaped_sglang(self, monkeypatch):
        _install_fake_sglang(monkeypatch, SGLANG_0523_MOE_SURFACE)
        module = _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, _COLLECT_MOE_PATH)
        assert module._triton_config_namespaces() == [
            sys.modules["sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config"]
        ]

    def test_unset_server_args_are_published_as_the_scheduler_publishes_them(self, monkeypatch):
        _published_moe_server_args.clear()
        _install_fake_sglang(monkeypatch, SGLANG_0523_MOE_SURFACE)

        _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, _COLLECT_MOE_PATH)

        assert len(_published_moe_server_args) == 1
        published, role = _published_moe_server_args[0]
        assert role == "scheduler"
        assert published.args == ()
        assert published.kwargs == {
            "model_path": "dummy",
            "enable_deterministic_inference": False,
            "enable_fused_moe_sum_all_reduce": False,
            "kt_weight_path": None,
            "flashinfer_mxfp4_moe_precision": "default",
        }

    def test_a_record_already_published_in_the_process_is_kept(self, monkeypatch):
        _published_moe_server_args.clear()
        surface = dict(SGLANG_0523_MOE_SURFACE)
        surface["sglang.srt.runtime_context"] = {
            **SGLANG_0523_MOE_SURFACE["sglang.srt.runtime_context"],
            "get_server_args": lambda: object(),
        }
        _install_fake_sglang(monkeypatch, surface)

        _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, _COLLECT_MOE_PATH)

        assert _published_moe_server_args == []

    def test_import_fails_on_the_retired_kernel_location(self, monkeypatch):
        """The 0.5.14 Triton MoE kernel module location is not a fallback any more."""
        surface = {
            name: attrs
            for name, attrs in SGLANG_0523_MOE_SURFACE.items()
            if name != "sglang.kernels.ops.moe.fused_moe_triton_kernels"
        }
        surface["sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels"] = {"_B_DESC_CACHE": {}}
        _install_fake_sglang(monkeypatch, surface)
        with pytest.raises(ImportError):
            _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, _COLLECT_MOE_PATH)


# ---------------------------------------------------------------------------
# collect_gemm.py / collect_computescale.py
# ---------------------------------------------------------------------------
# The sglang 0.5.23 names each module touches at import time: the serving
# GEMM entry points and backend initializers in srt/layers/quantization
# (fp8_utils, fp4_utils, unquant, modelopt_quant's NVFP4 padding helpers), the fused-op trace in sglang.kernels.fused_op
# and the SM120 KDA NVFP4 GEMM in sglang.kernels.ops.gemm. The JIT quantization
# kernels live in sglang.kernels.ops.quantization.fp8_kernel (the 0.5.14
# srt/layers/quantization/fp8_kernel path is gone).
SGLANG_0523_GEMM_SURFACE = {
    "sglang.kernels.fused_op": {
        "clear_fused_op_trace": _dummy,
        "disable_fused_op_trace": _dummy,
        "enable_fused_op_trace": _dummy,
        "get_fused_op_trace": _dummy,
    },
    "sglang.kernels.ops.gemm": {"try_qwen3x_nvfp4_gemm": _dummy},
    "sglang.srt.layers.quantization.unquant": {
        "UnquantizedLinearMethod": _Dummy,
        "initialize_bf16_gemm_config": _dummy,
    },
    "sglang.srt.layers.quantization.fp4_utils": {
        "fp4_quantize": _dummy,
        "get_fp4_gemm_runner_backend": _dummy,
        "initialize_fp4_gemm_config": _dummy,
    },
    "sglang.srt.layers.quantization.fp8_utils": {
        "_deepgemm_block_fp8_supported": _dummy,
        "apply_fp8_linear": _dummy,
        "cutlass_fp8_supported": _dummy,
        "cutlass_w8a8_block_fp8_linear_with_fallback": _dummy,
        "deepgemm_w8a8_block_fp8_linear_with_fallback": _dummy,
        "dispatch_w8a8_block_fp8_linear": _dummy,
        "initialize_fp8_gemm_config": _dummy,
        "requant_block_scale_ue8m0_for_deepgemm": _dummy,
        "triton_w8a8_block_fp8_linear": _dummy,
    },
    "sglang.srt.layers.quantization.modelopt_quant": {
        "pad_nvfp4_activation_for_cutlass": _dummy,
        "pad_nvfp4_weight": _dummy,
        "slice_nvfp4_output": _dummy,
    },
}
SGLANG_0523_COMPUTESCALE_SURFACE = {
    "sglang.kernels.ops.quantization.fp8_kernel": {
        "sglang_per_token_quant_fp8": _dummy,
        "static_quant_fp8": _dummy,
    },
}

_COLLECT_GEMM_PATH = COLLECTOR_DIR / "sglang" / "collect_gemm.py"
_COLLECT_GEMM_DOTTED = "collector.sglang.collect_gemm"
_COLLECT_COMPUTESCALE_PATH = COLLECTOR_DIR / "sglang" / "collect_computescale.py"
_COLLECT_COMPUTESCALE_DOTTED = "collector.sglang.collect_computescale"
# collect_gemm.py does `import torch.nn.functional as F` -- needs real
# submodule entries in sys.modules, not just attribute access on a mocked
# `torch`.
_GEMM_LENIENT_EXTRAS = ("torch.nn", "torch.nn.functional")


class TestCollectGemmImportSurface:
    """The forward-looking regression guard for the module-level imports of
    the GEMM and compute_scale collectors: a name the pinned release moved or
    dropped fails here instead of at the first GPU task."""

    def test_gemm_imports_cleanly_against_0523_shaped_sglang(self, monkeypatch):
        _install_fake_sglang(monkeypatch, SGLANG_0523_GEMM_SURFACE, lenient_extras=_GEMM_LENIENT_EXTRAS)
        module = _import_fresh(monkeypatch, _COLLECT_GEMM_DOTTED, _COLLECT_GEMM_PATH)
        assert module.__compat__ == "sglang==0.5.23"

    def test_gemm_import_fails_without_the_0523_block_fp8_dispatch(self, monkeypatch):
        surface = dict(SGLANG_0523_GEMM_SURFACE)
        surface["sglang.srt.layers.quantization.fp8_utils"] = {
            name: value
            for name, value in SGLANG_0523_GEMM_SURFACE["sglang.srt.layers.quantization.fp8_utils"].items()
            if name != "dispatch_w8a8_block_fp8_linear"
        }
        _install_fake_sglang(monkeypatch, surface, lenient_extras=_GEMM_LENIENT_EXTRAS)
        with pytest.raises(ImportError):
            _import_fresh(monkeypatch, _COLLECT_GEMM_DOTTED, _COLLECT_GEMM_PATH)

    def test_computescale_imports_cleanly_against_0523_shaped_sglang(self, monkeypatch):
        _install_fake_sglang(monkeypatch, SGLANG_0523_COMPUTESCALE_SURFACE)
        module = _import_fresh(monkeypatch, _COLLECT_COMPUTESCALE_DOTTED, _COLLECT_COMPUTESCALE_PATH)
        assert module.__compat__ == "sglang==0.5.23"


# ---------------------------------------------------------------------------
# collect_attn.py / collect_attn_encoder.py
# ---------------------------------------------------------------------------
# The sglang 0.5.23 names collect_attn.py touches at import time: the dense attention
# layer and FlashAttention backend, the KV pools, the KV-index translator the runner
# builds (mem_cache/kv_index_translator.py), the forward batch and context, and the
# runtime-context parallel override (evaluated at import, as a decorator) plus the named
# buffer the FlashInfer workspace lives in.
class _NullContextDecorator(contextlib.ContextDecorator):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeParallel:
    def __init__(self):
        self.overrides: list[dict] = []

    def override(self, **kwargs):
        self.overrides.append(kwargs)
        return _NullContextDecorator()


_ATTENTION_PARALLEL = _FakeParallel()

SGLANG_0523_ATTENTION_SURFACE = {
    "sglang.srt.configs.model_config": {"AttentionArch": _Dummy},
    "sglang.srt.layers.attention.flashattention_backend": {"FlashAttentionBackend": _Dummy},
    "sglang.srt.layers.radix_attention": {"RadixAttention": _Dummy},
    "sglang.srt.mem_cache.kv_index_translator": {"KVIndexTranslator": _Dummy},
    "sglang.srt.mem_cache.memory_pool": {"MHATokenToKVPool": _Dummy, "ReqToTokenPool": _Dummy},
    "sglang.srt.model_executor.forward_batch_info": {"ForwardBatch": _Dummy, "ForwardMode": _Dummy},
    "sglang.srt.model_executor.forward_context": {"ForwardContext": _Dummy, "forward_context": _dummy},
    "sglang.srt.runtime_context": {"get_buffer": _dummy, "get_parallel": lambda: _ATTENTION_PARALLEL},
}

_COLLECT_ATTN_PATH = COLLECTOR_DIR / "sglang" / "collect_attn.py"
_COLLECT_ATTN_DOTTED = "collector.sglang.collect_attn"
_COLLECT_ATTN_ENCODER_PATH = COLLECTOR_DIR / "sglang" / "collect_attn_encoder.py"
_COLLECT_ATTN_ENCODER_DOTTED = "collector.sglang.collect_attn_encoder"
_ATTN_LENIENT_EXTRAS = ("torch.nn",)


class TestCollectAttnImportSurface:
    def test_imports_cleanly_against_0523_shaped_sglang(self, monkeypatch):
        _ATTENTION_PARALLEL.overrides.clear()
        _install_fake_sglang(monkeypatch, SGLANG_0523_ATTENTION_SURFACE, lenient_extras=_ATTN_LENIENT_EXTRAS)
        module = _import_fresh(monkeypatch, _COLLECT_ATTN_DOTTED, _COLLECT_ATTN_PATH)
        assert callable(module.run_attention_torch)
        # The kernel collector runs one attention rank: the override the decorator declares.
        assert _ATTENTION_PARALLEL.overrides == [
            {
                "attn_tp_size": 1,
                "attn_tp_rank": 0,
                "attn_cp_size": 1,
                "attn_cp_rank": 0,
                "attn_dp_size": 1,
                "attn_dp_rank": 0,
            }
        ]

    def test_import_fails_without_the_0523_kv_index_translator(self, monkeypatch):
        """The runner's translator is built from sglang's own class; there is no stand-in."""
        surface = {
            name: attrs
            for name, attrs in SGLANG_0523_ATTENTION_SURFACE.items()
            if name != "sglang.srt.mem_cache.kv_index_translator"
        }
        _install_fake_sglang(monkeypatch, surface, lenient_extras=_ATTN_LENIENT_EXTRAS)
        with pytest.raises(ImportError):
            _import_fresh(monkeypatch, _COLLECT_ATTN_DOTTED, _COLLECT_ATTN_PATH)

    def test_encoder_imports_with_zero_sglang_modules_present(self, monkeypatch):
        """collect_attn_encoder.py imports its kernels inside the per-SM lane only."""
        _install_fake_sglang(monkeypatch, {})
        _import_fresh(monkeypatch, _COLLECT_ATTN_ENCODER_DOTTED, _COLLECT_ATTN_ENCODER_PATH)


# ---------------------------------------------------------------------------
# collect_gdn.py
# ---------------------------------------------------------------------------
_COLLECT_GDN_PATH = COLLECTOR_DIR / "sglang" / "collect_gdn.py"
_COLLECT_GDN_DOTTED = "collector.sglang.collect_gdn"


class TestCollectGdnImportSurface:
    def test_imports_cleanly_with_zero_sglang_modules_present(self, monkeypatch):
        """collect_gdn.py's only sglang references at module level are
        inside `if TYPE_CHECKING:` (inert at runtime) -- every real sglang
        import is deferred to call time, inside run_gdn_torch() (already
        covered by this file's own version-conditional-import fix and gdn's
        clean real-GPU run). This test locks that structural fact: importing
        the module must not require sglang to be present AT ALL. If a
        future change moved an sglang touch to module level without a
        version guard, this test would start failing the same way
        collect_moe.py's did."""
        monkeypatch.delitem(sys.modules, "sglang", raising=False)
        _install_fake_sglang(monkeypatch, {})  # torch/pkg_resources only, no sglang.* entries
        _import_fresh(monkeypatch, _COLLECT_GDN_DOTTED, _COLLECT_GDN_PATH)

#!/usr/bin/env python3
"""Quantize the exported FP32 ONNX models.

Produces two artefact families on top of ``output/``:

  output/omnivoice_lm_int8/model.onnx          -- LM, INT8 dynamic quantization
  output/omnivoice_lm_fp16/model.onnx          -- LM, fp16 weights
  output/audio_tokenizer_encoder_int8/model.onnx
  output/audio_tokenizer_decoder_int8/model.onnx

Why dynamic INT8?
  - No calibration data needed
  - Quantizes weights to INT8 + activations to INT8 at runtime
  - 2.5-4x size reduction for transformer LMs, 1.5-2x CPU speed-up
  - Typical accuracy loss < 0.5% on logits cosine

Why FP16?
  - Lossless w.r.t. the bf16-trained checkpoint (fp32 master is over-precision)
  - 2x size reduction, big GPU speed-up; CPU is mixed
  - Useful for CUDA Execution Provider deployment

Run:
    conda activate tts
    python quantize.py [--targets lm,at_encoder,at_decoder] [--methods int8,fp16]
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from _common import (
    AT_DEC_INT8_ONNX,
    AT_DEC_INT8_OUT_DIR,
    AT_DEC_ONNX,
    AT_ENC_INT8_ONNX,
    AT_ENC_INT8_OUT_DIR,
    AT_ENC_ONNX,
    LM_FP16_ONNX,
    LM_FP16_OUT_DIR,
    LM_INT4_ONNX,
    LM_INT4_OUT_DIR,
    LM_INT8_ONNX,
    LM_INT8_OUT_DIR,
    LM_INT8HQ_ONNX,
    LM_INT8HQ_OUT_DIR,
    LM_ONNX,
    file_size,
    human_size,
    setup_logging,
)

log = setup_logging("quantize")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _purge_dir(d: Path) -> None:
    if d.exists():
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
    else:
        d.mkdir(parents=True, exist_ok=True)


def _save_external(proto: onnx.ModelProto, out_path: Path) -> None:
    """Save a (possibly large) model with a single sidecar 'model.onnx_data'."""
    _purge_dir(out_path.parent)
    onnx.save_model(
        proto,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
        size_threshold=1024,
    )


# ---------------------------------------------------------------------------
# INT8 quantization (weight-only and dynamic)
# ---------------------------------------------------------------------------
#
# We support two INT8 strategies:
#
#   weight_only (DEFAULT):
#     * Quantize every big FP32 weight initializer to INT8 (per-row,
#       symmetric, max-abs scale).
#     * Insert one ``DequantizeLinear`` per weight, then leave the rest of
#       the graph untouched — activations stay FP32 throughout.
#     * No activation outlier risk → much higher quality on transformer
#       LMs than dynamic activation quantization.
#     * Same on-disk size as ``dynamic`` (weights are still INT8).
#     * Speed:  CPU is slightly slower than ``dynamic`` (FP32 MatMul after
#       dequant) but still ahead of FP16 on CPU.
#
#   dynamic:
#     * ORT's ``quantize_dynamic`` for MatMul/Gemm/Attention.
#     * Quantizes activations on the fly using their per-tensor max-abs.
#     * Faster on CPU but very sensitive to activation outliers; produces
#       audible artefacts on Qwen3-class LMs.


def _quantize_init_int8(arr: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-slice symmetric INT8 quantization along ``axis``.

    Returns (int8_weights, fp32_scales).  ``scales`` has the same shape as
    ``arr`` collapsed to a single dim along ``axis`` — i.e. one scale per
    slice along that axis.  Choose ``axis`` = the LARGER dim of a 2-D
    weight: that maximizes the number of scales (one per output channel)
    and minimizes per-element quant noise.
    """
    arr = arr.astype(np.float32)
    other_axes = tuple(i for i in range(arr.ndim) if i != axis)
    absmax = np.max(np.abs(arr), axis=other_axes)
    scale = absmax / 127.0
    scale[scale == 0] = 1.0  # avoid div-by-0 for all-zero rows
    # Reshape scale so it broadcasts over the non-quant axes
    bcast_shape = [1] * arr.ndim
    bcast_shape[axis] = arr.shape[axis]
    q = np.round(arr / scale.reshape(bcast_shape)).clip(-128, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def _pick_quant_axis(dims: list[int]) -> int:
    """For a 2-D weight, quantize along the LARGER axis (= per-output-channel
    for MatMul B [in, out] when out > in, etc.). One scale per element of
    that axis ⇒ finer quantization."""
    return int(np.argmax(dims))


def _init_names_for_node_patterns(model: onnx.ModelProto,
                                  patterns: list[str],
                                  op_types: tuple[str, ...] = ("MatMul", "Gemm"),
                                  ) -> set[str]:
    """Return initializer names consumed by any node whose ``op_type`` is in
    ``op_types`` and whose ``node.name`` contains one of ``patterns``.

    Used to *exclude* quality-critical layers (e.g. ``audio_heads``) from
    weight-only INT8 quantization so they stay full FP32 at no extra
    interface cost — only on-disk size grows by a few tens of MB.
    """
    if not patterns:
        return set()
    init_names = {init.name for init in model.graph.initializer}
    excluded: set[str] = set()
    matched_nodes: list[str] = []
    for n in model.graph.node:
        if n.op_type not in op_types:
            continue
        if not any(pat in n.name for pat in patterns):
            continue
        matched_nodes.append(n.name)
        for inp in n.input:
            if inp in init_names:
                excluded.add(inp)
    if matched_nodes:
        log.info("    keeping FP32 (matched %d node(s)):", len(matched_nodes))
        for nm in matched_nodes:
            log.info("      %s", nm)
    return excluded


def quantize_initializers_int8(model: onnx.ModelProto,
                               size_threshold_mb: float = 1.0,
                               exclude_init_names: set[str] | None = None,
                               ) -> int:
    """In-place per-row INT8 quantization of every FP32 2-D initializer
    >= ``size_threshold_mb`` MB.  Each quantized initializer is replaced by
    ``DequantizeLinear(W_int8, scale)`` whose output keeps the original
    name's place in the graph.

    ``exclude_init_names`` lets callers protect specific weights (e.g. the
    LM output head) from quantization — they remain untouched FP32.

    Returns the number of initializers quantized.
    """
    g = model.graph
    exclude_init_names = exclude_init_names or set()

    targets = []
    for init in g.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        if init.name in exclude_init_names:
            continue
        nbytes = 4
        for d in init.dims:
            nbytes *= d
        if nbytes < size_threshold_mb * 1024 * 1024:
            continue
        if len(init.dims) != 2:
            # 1-D weights (norm scales etc.) and >2-D (Conv kernels) are
            # skipped: the former are tiny, the latter need a different
            # quant scheme.
            continue
        targets.append(init)

    if not targets:
        log.info("  [wo-int8] Nothing to quantize.")
        return 0

    total_bytes = sum(int(np.prod(t.dims)) * 4 for t in targets)
    log.info("  [wo-int8] Quantizing %d initializers (%s of fp32 weights):",
             len(targets), human_size(total_bytes))
    for init in targets:
        ax = _pick_quant_axis(list(init.dims))
        log.info("    %-50s shape=%s  axis=%d  size=%.1f MB",
                 init.name[:50], list(init.dims), ax,
                 (np.prod(init.dims) * 4) / 1024 / 1024)

    new_initializers = []
    new_nodes = []
    rename_map: dict[str, str] = {}

    for init in targets:
        arr = numpy_helper.to_array(init)
        axis = _pick_quant_axis(list(init.dims))
        q, scale = _quantize_init_int8(arr, axis=axis)

        q_name = init.name + ".int8"
        s_name = init.name + ".scale"
        dq_out = init.name + ".dq"

        new_initializers.append(numpy_helper.from_array(q, q_name))
        new_initializers.append(numpy_helper.from_array(scale, s_name))
        # DequantizeLinear  y = (x - zp) * scale, broadcast scale along ``axis``
        new_nodes.append(helper.make_node(
            "DequantizeLinear",
            inputs=[q_name, s_name],
            outputs=[dq_out],
            name=f"DequantizeLinear_{init.name.replace('.', '_').replace('/', '_')}",
            axis=axis,
        ))
        rename_map[init.name] = dq_out

    # Drop the FP32 originals
    keep_inits = [init for init in g.initializer if init.name not in rename_map]
    g.ClearField("initializer")
    g.initializer.extend(keep_inits)
    g.initializer.extend(new_initializers)

    # Rewire every consumer of the old initializer name
    for node in g.node:
        for i, inp in enumerate(node.input):
            if inp in rename_map:
                node.input[i] = rename_map[inp]

    # Append dequant nodes; ORT topologically sorts on load.
    g.node.extend(new_nodes)
    return len(targets)


def quantize_int8_weight_only(src: Path, dst: Path, name: str,
                              size_threshold_mb: float = 1.0,
                              exclude_node_patterns: list[str] | None = None,
                              ) -> None:
    """Pure weight-only INT8 (W8A32) quantization.

    Loads the FP32 source, rewrites every big 2-D FP32 initializer to
    ``DequantizeLinear(int8_weights, scale)``, leaves activations FP32.

    ``exclude_node_patterns`` keeps any MatMul / Gemm whose ``node.name``
    contains one of these substrings at FP32. Useful for the LM output
    head (``audio_heads``) where per-row INT8 noise propagates directly
    to the sampler logits.
    """
    log.info("[%s] INT8 weight-only quantization ...", name)
    log.info("  src: %s   (%s)", src, human_size(file_size(src)))
    if exclude_node_patterns:
        log.info("  Keeping FP32 for nodes matching: %s", exclude_node_patterns)
    _purge_dir(dst.parent)

    t0 = time.time()
    model = onnx.load(str(src), load_external_data=True)
    excluded_inits = _init_names_for_node_patterns(model, exclude_node_patterns or [])
    n = quantize_initializers_int8(
        model,
        size_threshold_mb=size_threshold_mb,
        exclude_init_names=excluded_inits,
    )
    _save_external(model, dst)
    dt = time.time() - t0
    log.info("  Quantized %d weights in %.1fs", n, dt)
    if excluded_inits:
        log.info("  Kept %d FP32 weights (excluded by pattern)", len(excluded_inits))
    log.info("  dst: %s   (%s)", dst, human_size(file_size(dst)))


def quantize_int8_dynamic(src: Path, dst: Path, name: str) -> None:
    """ORT dynamic INT8 quantization (weight INT8 + per-tensor dynamic
    activation INT8) followed by a weight-only pass for the embeddings
    that ORT skipped (Gather operands).

    Kept as an opt-in alternative to ``weight_only``: faster on CPU but
    noticeably worse audio quality on Qwen3-class LMs because of
    activation outliers.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    log.info("[%s] INT8 dynamic quantization ...", name)
    log.info("  src: %s   (%s)", src, human_size(file_size(src)))
    _purge_dir(dst.parent)

    t0 = time.time()
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Attention"],
        per_channel=True,
        reduce_range=False,
        use_external_data_format=True,
        extra_options={
            "EnableSubgraph": True,
            "MatMulConstBOnly": True,
            "DefaultTensorType": int(onnx.TensorProto.FLOAT),
        },
    )
    dt = time.time() - t0
    _normalize_sidecar(dst)
    log.info("  ORT dynamic-quant done in %.1fs  intermediate size: %s",
             dt, human_size(file_size(dst)))

    # Second pass: weight-only INT8 for big FP32 initializers that ORT
    # left untouched (mostly embedding tables consumed by Gather).
    log.info("  [wo-int8 patch] embeddings & remaining big fp32 weights ...")
    model = onnx.load(str(dst), load_external_data=True)
    quantize_initializers_int8(model, size_threshold_mb=1.0)
    _save_external(model, dst)

    log.info("  dst: %s   (%s)", dst, human_size(file_size(dst)))


# ---------------------------------------------------------------------------
# INT4 group-wise weight-only quantization (LLM-style)
# ---------------------------------------------------------------------------


def quantize_int4(src: Path, dst: Path, name: str,
                  block_size: int = 16, is_symmetric: bool = True,
                  accuracy_level: int = 4,
                  algo: str = "rtn",
                  exclude_node_patterns: list[str] | None = None) -> None:
    """Group-wise 4-bit weight-only quantization for MatMul nodes.

    Uses ``onnxruntime.quantization.matmul_4bits_quantizer``.  Each MatMul
    weight is split along its reduction axis into blocks of ``block_size``
    and each block gets its own (scale, zero_point).  Result op is the
    ORT-custom ``MatMulNBits`` (ORT >= 1.16).

    After the 4-bit pass we still have FP32 embedding tables (Gather data,
    not MatMul), so we run our regular weight-only INT8 pass on top to
    shrink ``embed_tokens`` / ``audio_embeddings`` and any excluded MatMul
    weights (e.g. the LM output head).

    Args:
      block_size: smaller = better quality, slightly larger model.
        16 is a good default for OmniVoice; 32/128 are more common in LLMs.
      is_symmetric: ``rtn`` only. True → simpler/faster MatMulNBits kernel;
        False → asymmetric (zero_point per block).  HQQ is always asymmetric.
      accuracy_level: ORT execution precision for MatMulNBits.
        0/1 = fp32, 4 = int8.  ``4`` is the fastest CPU path.
      algo: ``rtn`` (round-to-nearest, default, fastest) or ``hqq``
        (Half-Quadratic Quantization, ~5-10x slower but noticeably better
        accuracy on the same block_size).
      exclude_node_patterns: substrings; any MatMul whose node name contains
        one of these strings is left untouched in the 4-bit pass.
    """
    from onnxruntime.quantization.matmul_4bits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        HQQWeightOnlyQuantConfig,
        MatMul4BitsQuantizer,
        QuantFormat,
    )

    log.info("[%s] INT4 group-wise weight-only quantization ...", name)
    log.info("  src: %s   (%s)", src, human_size(file_size(src)))
    log.info("  algo=%s  block_size=%d  symmetric=%s  accuracy_level=%d  format=QOperator(MatMulNBits)",
             algo, block_size, is_symmetric, accuracy_level)
    _purge_dir(dst.parent)

    # Load with external data — quantizer will hold the unpacked tensors.
    t0 = time.time()
    proto = onnx.load(str(src), load_external_data=True)

    # Resolve exclude patterns to actual MatMul node names. Skipping the
    # output head (``audio_heads``) is critical: it directly produces the
    # logits, so even small per-block RTN noise gets multiplied through to
    # the sampler and visibly drops argmax-agreement.
    nodes_to_exclude: list[str] = []
    if exclude_node_patterns:
        for n in proto.graph.node:
            if n.op_type != "MatMul":
                continue
            for pat in exclude_node_patterns:
                if pat in n.name:
                    nodes_to_exclude.append(n.name)
                    break
        log.info("  Excluding %d MatMul nodes from INT4 (will stay FP32, then "
                 "INT8-weight-only-patched if >= 1 MB):", len(nodes_to_exclude))
        for n in nodes_to_exclude:
            log.info("    %s", n)

    if algo == "rtn":
        cfg = DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=is_symmetric,
            accuracy_level=accuracy_level,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
        )
    elif algo == "hqq":
        # HQQ: Half-Quadratic Quantization. Iteratively solves for the
        # optimal (scale, zero_point) per block using a closed-form proximal
        # update on |W - dequant(quant(W))|. Always asymmetric. No
        # calibration data required (data-free, like RTN).
        cfg = HQQWeightOnlyQuantConfig(
            block_size=block_size,
            bits=4,
            axis=1,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
        )
    else:
        raise ValueError(f"Unknown algo: {algo!r} (use 'rtn' or 'hqq')")

    quantizer = MatMul4BitsQuantizer(
        proto,
        algo_config=cfg,
        nodes_to_exclude=nodes_to_exclude or None,
    )
    quantizer.process()
    dt = time.time() - t0
    log.info("  4-bit MatMul pass done in %.1fs", dt)

    # Save with external data — quantizer.model is an ONNXModel wrapper
    quantizer.model.save_model_to_file(str(dst), use_external_data_format=True)
    _normalize_sidecar(dst)
    log.info("  intermediate size: %s", human_size(file_size(dst)))

    # Second pass: weight-only INT8 for big FP32 initializers (embedding
    # tables — Gather operands — are still FP32 after the 4-bit pass).
    log.info("  [wo-int8 patch] embeddings & remaining big fp32 weights ...")
    model = onnx.load(str(dst), load_external_data=True)
    quantize_initializers_int8(model, size_threshold_mb=1.0)
    _save_external(model, dst)

    log.info("  dst: %s   (%s)", dst, human_size(file_size(dst)))


def _normalize_sidecar(model_path: Path) -> None:
    """Ensure the only sidecar in the model dir is named 'model.onnx_data'.

    ORT's ``quantize_dynamic`` writes the sidecar as ``<name>.data`` (or
    sometimes one file per tensor). We load WITH the original references,
    then re-save using our canonical naming.
    """
    proto = onnx.load(str(model_path), load_external_data=True)
    _save_external(proto, model_path)
    # `_save_external` already wiped the directory and wrote {model.onnx,
    # model.onnx_data}, so any stragglers are gone.


# ---------------------------------------------------------------------------
# FP16 conversion
# ---------------------------------------------------------------------------


def convert_fp16(src: Path, dst: Path, name: str) -> None:
    """Convert FP32 -> FP16 weights using ``onnxconverter_common``.

    Keeps Op-set 17 and skips ops that misbehave at fp16 on CPU (norms,
    LayerNorm, softmax inputs). ORT will up-cast activations to fp32 around
    those ops automatically.
    """
    from onnxconverter_common import float16

    log.info("[%s] FP16 weight conversion ...", name)
    log.info("  src: %s   (%s)", src, human_size(file_size(src)))

    t0 = time.time()
    proto = onnx.load(str(src), load_external_data=True)
    # NOTE on op_block_list:
    # `onnxconverter_common.float16` automatically inserts Cast nodes around
    # blocked ops *only when shape inference succeeds*.  Our LM is > 2 GB so
    # we must disable shape inference, which means a long block_list silently
    # creates fp32↔fp16 mixed-type edges that ORT then refuses to load.
    # Empirical sweet spot for Qwen3 + ORT-CPU:
    #   - keep RMSNorm-class ops in fp16 (numerically fine: variance is computed
    #     in fp32 internally by SimplifiedLayerNormalization regardless of I/O)
    #   - keep nothing in fp32 ⇒ no cast-stitching needed
    fp16_proto = float16.convert_float_to_float16(
        proto,
        keep_io_types=True,                       # graph inputs/outputs stay fp32 → drop-in replacement
        disable_shape_infer=True,                 # required: shape infer fails on >2GB models
        op_block_list=[],                         # empty — avoid mixed-precision edges with no Casts
    )
    _save_external(fp16_proto, dst)
    dt = time.time() - t0
    log.info("  dst: %s   (%s)   [%.1fs]", dst, human_size(file_size(dst)), dt)


# ---------------------------------------------------------------------------
# Smoke test (load + 1 inference) — confirms the file is well-formed
# ---------------------------------------------------------------------------


def smoke_test(model_path: Path, name: str) -> None:
    import onnxruntime as ort

    log.info("[%s] Smoke test (load + 1 inference) ...", name)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        # special-case audio length to be a multiple of 960 (encoder hop)
        if "audio" == inp.name and len(shape) == 3:
            shape[-1] = 960 * 25  # 1 second
        if "audio_codes" == inp.name and len(shape) == 3:
            shape[1] = 8
            shape[-1] = 25
        if inp.name == "input_ids":
            shape = [1, 8, 16]
        if inp.name == "audio_mask":
            shape = [1, 16]
        if inp.name == "attention_mask":
            shape = [1, 1, 16, 16]
        if inp.name == "position_ids":
            shape = [1, 16]
        dtype_map = {"tensor(int64)": np.int64, "tensor(float)": np.float32,
                     "tensor(bool)": bool, "tensor(float16)": np.float16}
        dt = dtype_map.get(inp.type, np.float32)
        if inp.name in {"input_ids", "audio_codes"}:
            data = np.zeros(shape, dtype=np.int64)
        elif inp.name in {"audio_mask", "attention_mask"}:
            data = np.ones(shape, dtype=bool)
        elif inp.name == "position_ids":
            data = np.arange(shape[-1], dtype=np.int64).reshape(1, -1)
        else:
            data = np.zeros(shape, dtype=dt)
        feeds[inp.name] = data

    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, feeds)
    for n, o in zip(out_names, outs):
        log.info("  out '%s' shape=%s dtype=%s", n, list(o.shape), o.dtype)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


JOBS_INT8 = [
    (LM_ONNX, LM_INT8_ONNX, "LM"),
    (AT_ENC_ONNX, AT_ENC_INT8_ONNX, "AT-encoder"),
    (AT_DEC_ONNX, AT_DEC_INT8_ONNX, "AT-decoder"),
]
# weight_only_hq is LM-only by design: AT models reuse the existing INT8
# build (excluding layers there gives no measurable quality benefit).
JOBS_INT8HQ = [
    (LM_ONNX, LM_INT8HQ_ONNX, "LM"),
]
JOBS_FP16 = [
    (LM_ONNX, LM_FP16_ONNX, "LM"),
]
JOBS_INT4 = [
    (LM_ONNX, LM_INT4_ONNX, "LM"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", default="lm,at_encoder,at_decoder",
                   help="Comma list: lm, at_encoder, at_decoder")
    p.add_argument("--methods", default="int8,fp16",
                   help="Comma list: int8, fp16, int4")
    p.add_argument("--int8-mode", default="weight_only",
                   choices=["weight_only", "dynamic", "weight_only_hq"],
                   help="weight_only (default) | dynamic (faster, lossy) | "
                        "weight_only_hq (same as weight_only, but keeps "
                        "quality-critical layers in FP32 — see --int8hq-exclude. "
                        "Output goes to a separate folder so the standard "
                        "INT8 build is preserved.)")
    p.add_argument("--int8hq-exclude", default="audio_heads",
                   help="Comma-separated substrings of MatMul/Gemm node names to "
                        "KEEP at FP32 in --int8-mode=weight_only_hq. Default "
                        "'audio_heads' protects the LM output head whose logits "
                        "feed the sampler. Pass '' to disable exclusions.")
    p.add_argument("--int4-block-size", type=int, default=16,
                   help="Group size for INT4 quantization (smaller=better quality, default 16)")
    p.add_argument("--int4-algo", default="hqq", choices=["rtn", "hqq"],
                   help="rtn = round-to-nearest (fastest); hqq = Half-Quadratic Quantization "
                        "(default, ~5-10x slower to build but noticeably better quality, no "
                        "calibration data needed).")
    p.add_argument("--int4-asymmetric", action="store_true",
                   help="Use asymmetric (zero_point per block) INT4. Only affects --int4-algo=rtn (HQQ is always asymmetric).")
    p.add_argument("--int4-quantize-output-head", action="store_true",
                   help="By default we KEEP the LM output head (audio_heads MatMul) at FP32+INT8 (much higher logits quality). Pass this flag to also INT4-quantize it.")
    p.add_argument("--no-smoke", action="store_true",
                   help="Skip the load + 1-inference smoke test")
    args = p.parse_args()

    targets = {t.strip() for t in args.targets.split(",") if t.strip()}
    methods = {m.strip() for m in args.methods.split(",") if m.strip()}

    target_to_name = {"lm": "LM", "at_encoder": "AT-encoder", "at_decoder": "AT-decoder"}

    if "int8" in methods:
        log.info("=" * 60)
        log.info("INT8 quantization (mode=%s)", args.int8_mode)
        log.info("=" * 60)
        if args.int8_mode == "weight_only_hq":
            jobs = JOBS_INT8HQ
            patterns = [s.strip() for s in args.int8hq_exclude.split(",") if s.strip()]
            log.info("  Output  -> %s  (separate folder, standard INT8 build untouched)",
                     LM_INT8HQ_OUT_DIR)
            log.info("  FP32-keep patterns: %s", patterns or "<none>")

            def quantize_fn(src, dst, name):
                quantize_int8_weight_only(
                    src, dst, name,
                    exclude_node_patterns=patterns or None,
                )
            label = "int8hq"
        else:
            jobs = JOBS_INT8
            quantize_fn = (quantize_int8_weight_only if args.int8_mode == "weight_only"
                           else quantize_int8_dynamic)
            label = "int8"
        for src, dst, name in jobs:
            short = {"LM": "lm", "AT-encoder": "at_encoder", "AT-decoder": "at_decoder"}[name]
            if short not in targets:
                continue
            if not src.exists():
                log.warning("Skip %s: source %s missing.", name, src)
                continue
            quantize_fn(src, dst, name)
            if not args.no_smoke:
                smoke_test(dst, f"{name}-{label}")

    if "fp16" in methods:
        log.info("=" * 60)
        log.info("FP16 weight conversion")
        log.info("=" * 60)
        for src, dst, name in JOBS_FP16:
            short = {"LM": "lm"}[name]
            if short not in targets:
                continue
            if not src.exists():
                log.warning("Skip %s: source %s missing.", name, src)
                continue
            convert_fp16(src, dst, name)
            if not args.no_smoke:
                smoke_test(dst, f"{name}-fp16")

    if "int4" in methods:
        log.info("=" * 60)
        log.info("INT4 group-wise weight-only quantization")
        log.info("=" * 60)
        for src, dst, name in JOBS_INT4:
            short = {"LM": "lm"}[name]
            if short not in targets:
                continue
            if not src.exists():
                log.warning("Skip %s: source %s missing.", name, src)
                continue
            quantize_int4(
                src, dst, name,
                block_size=args.int4_block_size,
                is_symmetric=not args.int4_asymmetric,
                algo=args.int4_algo,
                exclude_node_patterns=(
                    None if args.int4_quantize_output_head else ["audio_heads"]
                ),
            )
            if not args.no_smoke:
                smoke_test(dst, f"{name}-int4")

    log.info("=" * 60)
    log.info("Size summary")
    log.info("=" * 60)
    for label, p in [
        ("LM       fp32", LM_ONNX),
        ("LM       int8", LM_INT8_ONNX),
        ("LM       int8hq", LM_INT8HQ_ONNX),
        ("LM       int4", LM_INT4_ONNX),
        ("LM       fp16", LM_FP16_ONNX),
        ("AT-enc   fp32", AT_ENC_ONNX),
        ("AT-enc   int8", AT_ENC_INT8_ONNX),
        ("AT-dec   fp32", AT_DEC_ONNX),
        ("AT-dec   int8", AT_DEC_INT8_ONNX),
    ]:
        if p.exists():
            log.info("  %-15s  %s", label, human_size(file_size(p)))
        else:
            log.info("  %-15s  -- not built --", label)


if __name__ == "__main__":
    main()

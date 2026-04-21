"""Shared utilities for OmniVoice ONNX export / verification scripts.

All export scripts and verification scripts in this folder import from here so
that paths, device selection and logging are consistent.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent  # parent of this folder; expected to be the
                                # OmniVoice repo root (containing OmniVoice/
                                # weights, omnivoice/ package, assert/)

# Source PyTorch model (HF format) — has model.safetensors + audio_tokenizer/
PT_MODEL_DIR = PROJECT_ROOT / "OmniVoice"
AUDIO_TOKENIZER_DIR = PT_MODEL_DIR / "audio_tokenizer"

# Reference asset for end-to-end test
REF_AUDIO = PROJECT_ROOT / "assert" / "andelie.wav"

# Output layout — one folder per ONNX artefact, all under output/
OUTPUT_DIR = THIS_DIR / "output"
LM_OUT_DIR = OUTPUT_DIR / "omnivoice_lm"
AT_ENC_OUT_DIR = OUTPUT_DIR / "audio_tokenizer_encoder"
AT_DEC_OUT_DIR = OUTPUT_DIR / "audio_tokenizer_decoder"

LM_ONNX = LM_OUT_DIR / "model.onnx"
AT_ENC_ONNX = AT_ENC_OUT_DIR / "model.onnx"
AT_DEC_ONNX = AT_DEC_OUT_DIR / "model.onnx"

# Quantized variants — same layout as the FP32 counterparts but in different
# folders so the original is preserved.  Created by ``quantize.py``.
LM_INT8_OUT_DIR    = OUTPUT_DIR / "omnivoice_lm_int8"
LM_FP16_OUT_DIR    = OUTPUT_DIR / "omnivoice_lm_fp16"
AT_ENC_INT8_OUT_DIR = OUTPUT_DIR / "audio_tokenizer_encoder_int8"
AT_DEC_INT8_OUT_DIR = OUTPUT_DIR / "audio_tokenizer_decoder_int8"
LM_INT4_OUT_DIR     = OUTPUT_DIR / "omnivoice_lm_int4"
LM_INT8HQ_OUT_DIR   = OUTPUT_DIR / "omnivoice_lm_int8_hq"

LM_INT8_ONNX     = LM_INT8_OUT_DIR / "model.onnx"
LM_FP16_ONNX     = LM_FP16_OUT_DIR / "model.onnx"
AT_ENC_INT8_ONNX = AT_ENC_INT8_OUT_DIR / "model.onnx"
AT_DEC_INT8_ONNX = AT_DEC_INT8_OUT_DIR / "model.onnx"
LM_INT4_ONNX     = LM_INT4_OUT_DIR / "model.onnx"
LM_INT8HQ_ONNX   = LM_INT8HQ_OUT_DIR / "model.onnx"


def variant_paths(variant: str) -> tuple["Path", "Path", "Path"]:
    """Return (LM, AT-encoder, AT-decoder) ONNX paths for a given variant.

    AT models are kept in FP32 for ``fp16`` variant because the audio tokenizer
    is small (740 MB total) and FP16 conversion of conv-heavy decoders is
    fragile in ORT-CPU.
    """
    if variant == "fp32":
        return LM_ONNX, AT_ENC_ONNX, AT_DEC_ONNX
    if variant == "int8":
        return LM_INT8_ONNX, AT_ENC_INT8_ONNX, AT_DEC_INT8_ONNX
    if variant == "fp16":
        return LM_FP16_ONNX, AT_ENC_ONNX, AT_DEC_ONNX
    if variant == "int4":
        # Only LM is INT4 (group-wise weight-only via MatMulNBits). The audio
        # tokenizer is small and conv-heavy so we reuse its INT8 build.
        return LM_INT4_ONNX, AT_ENC_INT8_ONNX, AT_DEC_INT8_ONNX
    if variant == "int8hq":
        # "High-quality" INT8: same per-row weight-only scheme as ``int8`` but
        # keeps quality-critical MatMul layers (audio output head by default,
        # optionally also ``o_proj`` / ``down_proj``) in FP32 to reduce
        # quantization noise. AT models reuse the standard INT8 build.
        return LM_INT8HQ_ONNX, AT_ENC_INT8_ONNX, AT_DEC_INT8_ONNX
    raise ValueError(
        f"Unknown variant: {variant!r} (use fp32 / int8 / fp16 / int4 / int8hq)"
    )

# ONNX opset — 17 supports LayerNorm, GroupNormalization (18+), most modern ops
ONNX_OPSET = 17


def setup_logging(name: str = "onnx_export") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy transformers logging
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    return logging.getLogger(name)


def pick_device(prefer: str | None = None) -> torch.device:
    """Choose device for export. CPU is safest for ONNX tracing; CUDA is faster."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def add_project_root_to_path() -> None:
    """Make ``import omnivoice`` work when running these scripts directly."""
    p = str(PROJECT_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def patch_transformers_sdpa_mask_for_tracing() -> None:
    """Work around a transformers >=5.5 bug that breaks ``torch.onnx.export``.

    Inside ``transformers.masking_utils.sdpa_mask`` there is a backward-compat
    branch:

        if isinstance(q_length, torch.Tensor):
            q_length, q_offset = q_length.shape[0], q_length[0].to(device)

    During ``torch.onnx.export`` tracing, scalar python ints get wrapped into
    0-D tensors, so ``q_length.shape[0]`` raises ``IndexError`` even though
    we are not actually using ``cache_position`` semantics. We patch the
    function so that 0-D tensors fall through to the normal ``int(q_length)``
    path. Has zero effect on regular eager/inference behaviour.
    """
    try:
        from transformers import masking_utils
    except ImportError:
        return
    if getattr(masking_utils, "_omnivoice_onnx_patched", False):
        return

    _orig = masking_utils.sdpa_mask

    def _patched(*args, **kwargs):
        # ``q_length`` is the 2nd positional arg of sdpa_mask(batch_size, q_length, ...)
        if "q_length" in kwargs:
            q = kwargs["q_length"]
            if isinstance(q, torch.Tensor) and q.ndim == 0:
                kwargs["q_length"] = int(q.item())
        elif len(args) > 1:
            q = args[1]
            if isinstance(q, torch.Tensor) and q.ndim == 0:
                args = (args[0], int(q.item())) + args[2:]
        return _orig(*args, **kwargs)

    masking_utils.sdpa_mask = _patched
    # ``ALL_MASK_ATTENTION_FUNCTIONS._global_mapping`` caches the original
    # function objects at import time; update entries that delegate to sdpa.
    if hasattr(masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS"):
        gm = getattr(masking_utils.ALL_MASK_ATTENTION_FUNCTIONS, "_global_mapping", None)
        if isinstance(gm, dict) and "sdpa" in gm:
            gm["sdpa"] = _patched

    # Also short-circuit ``create_bidirectional_mask`` when there is no
    # padding mask. HuBERT (used inside the audio tokenizer) always calls this
    # helper unconditionally; the resulting attention bias gets baked into the
    # trace as a constant 4-D tensor, which breaks dynamic-length inference.
    # Returning ``None`` here is semantically correct — every attention
    # implementation we use treats ``None`` as "full attention, no bias".
    if hasattr(masking_utils, "create_bidirectional_mask"):
        _orig_bid = masking_utils.create_bidirectional_mask

        def _patched_bid(config=None, inputs_embeds=None, attention_mask=None,
                         **kwargs):
            if attention_mask is None:
                return None
            return _orig_bid(
                config=config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )

        masking_utils.create_bidirectional_mask = _patched_bid
        # The function is also re-exported into model modules; patch live refs.
        try:
            from transformers.models.hubert import modeling_hubert
            modeling_hubert.create_bidirectional_mask = _patched_bid
        except Exception:
            pass

    masking_utils._omnivoice_onnx_patched = True


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LM_OUT_DIR.mkdir(parents=True, exist_ok=True)
    AT_ENC_OUT_DIR.mkdir(parents=True, exist_ok=True)
    AT_DEC_OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_omnivoice(device: torch.device, dtype: torch.dtype = torch.float32,
                   attn_implementation: str = "eager"):
    """Load OmniVoice in ``train`` mode (no audio tokenizer / no ASR loaded).

    ``attn_implementation`` defaults to ``"eager"`` for ONNX export friendliness
    (additive float mask path traces cleanly). Set to ``"sdpa"`` for fastest
    PyTorch inference if you don't intend to export.
    """
    add_project_root_to_path()
    patch_transformers_sdpa_mask_for_tracing()
    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained(
        str(PT_MODEL_DIR),
        train=True,           # skip auto-loading audio_tokenizer / ASR
        dtype=dtype,
        device_map={"": device.type if device.type != "cuda" else f"cuda:{device.index or 0}"},
        attn_implementation=attn_implementation,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_audio_tokenizer(device: torch.device, dtype: torch.dtype = torch.float32):
    """Load HiggsAudioV2 tokenizer (eager attention for ONNX-friendly tracing)."""
    from transformers import HiggsAudioV2TokenizerModel

    patch_transformers_sdpa_mask_for_tracing()

    tok = HiggsAudioV2TokenizerModel.from_pretrained(
        str(AUDIO_TOKENIZER_DIR),
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


def human_size(n_bytes: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.2f} {u}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} TB"


def file_size(path: Path) -> int:
    """Total size including external-data sidecars (``model.onnx_data``)."""
    if not path.exists():
        return 0
    total = path.stat().st_size
    for sib in path.parent.iterdir():
        if sib.name.startswith(path.name) and sib != path:
            total += sib.stat().st_size
    return total


def report_node_stats(onnx_path: Path, top_k: int = 15) -> None:
    """Print node count and op-type breakdown for an ONNX model file."""
    import onnx
    from collections import Counter

    m = onnx.load(str(onnx_path), load_external_data=False)
    g = m.graph
    ops = Counter(n.op_type for n in g.node)
    print(f"  -- {onnx_path.relative_to(THIS_DIR)} --  total nodes={len(g.node)}  "
          f"initializers={len(g.initializer)}")
    for op, cnt in ops.most_common(top_k):
        print(f"     {op:30s} {cnt}")


def report_diff(name: str, ref: torch.Tensor, hyp: torch.Tensor) -> dict:
    """Print absolute / relative / cosine diff between two tensors."""
    ref = ref.detach().to(torch.float32).cpu()
    hyp = hyp.detach().to(torch.float32).cpu()
    if ref.shape != hyp.shape:
        raise ValueError(f"{name} shape mismatch: ref {tuple(ref.shape)} vs hyp {tuple(hyp.shape)}")
    diff = (ref - hyp).abs()
    abs_max = float(diff.max())
    abs_mean = float(diff.mean())
    denom = ref.abs().clamp_min(1e-8)
    rel_mean = float((diff / denom).mean())
    cos = float(
        torch.nn.functional.cosine_similarity(ref.flatten().unsqueeze(0), hyp.flatten().unsqueeze(0))
        .item()
    )
    print(
        f"  [{name}] shape={tuple(ref.shape)}  "
        f"max|Δ|={abs_max:.3e}  mean|Δ|={abs_mean:.3e}  "
        f"meanRel={rel_mean:.3e}  cos={cos:.6f}"
    )
    return {"abs_max": abs_max, "abs_mean": abs_mean, "rel_mean": rel_mean, "cos": cos}

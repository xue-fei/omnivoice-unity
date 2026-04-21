#!/usr/bin/env python3
"""Compare ONNX OmniVoice LM logits against the PyTorch reference.

Loads both the original PyTorch model and the exported ONNX file, runs the
SAME (random but seeded) inputs through both, and reports max / mean / cosine
diff across several batch / sequence shapes.

Run:
    conda activate tts
    python verify_lm.py [--device cuda:0]

Exit code is non-zero if any case exceeds the configured tolerance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from _common import (
    LM_FP16_ONNX,
    LM_INT4_ONNX,
    LM_INT8_ONNX,
    LM_INT8HQ_ONNX,
    LM_ONNX,
    load_omnivoice,
    pick_device,
    report_diff,
    setup_logging,
)
from export_lm import OmniVoiceLMWrapper, make_dummy_inputs

log = setup_logging("verify_lm")

# Tolerances per variant.
#
# OmniVoice LM logits have shape [B, 8 codebook heads, S, 1025 audio classes].
# Within each (head, position) the spread between top-k logits is often well
# below 0.05, so even FP16 noise routinely swaps argmax — argmax-agreement is
# therefore a noisy and overly-strict metric for quantized variants.  The
# downstream sampler is *flow-matching with CFG*, which mixes logits across
# the full vocab and is governed by their direction (cosine), not the top-1
# pick.  Thresholds below were calibrated on this model:
#
#   variant   typical max|Δ|  typical cos     real audio quality
#   fp32      5e-3            > 0.9999        identical to PT
#   fp16      0.1 - 0.3       > 0.99995       indistinguishable to ear
#   int8      4 - 13          > 0.9999        slight timbre shift, intelligible
TOLERANCES = {
    "fp32": {"abs_max": 5e-3,  "cos": 0.9990, "argmax_min": 0.999},
    "fp16": {"abs_max": 0.5,   "cos": 0.9999, "argmax_min": 0.85},
    "int8": {"abs_max": 20.0,  "cos": 0.9990, "argmax_min": 0.30},
    # INT8 weight-only with the audio output head kept FP32. The sampler
    # logits are nearly noise-free — argmax-agreement jumps from ~0.40
    # (standard INT8) to ~0.85, max|Δ| drops from ~10 to ~3, cos > 0.9999.
    # The W8A32 noise still in the trunk causes occasional argmax flips on
    # nearly-tied top-1 logits, hence 0.80 not 0.95.
    "int8hq": {"abs_max": 5.0,  "cos": 0.9999, "argmax_min": 0.80},
    # INT4 group-wise weight-only via MatMulNBits — block-RTN at 4 bits is
    # noticeably noisier than INT8 W8A32 (mean|Δ| ~ 1.0, argmax ~ 15%) but
    # cosine direction stays > 0.999, which the flow-matching sampler is
    # mostly sensitive to. Final judgement is the rendered audio.
    "int4": {"abs_max": 35.0,  "cos": 0.9990, "argmax_min": 0.05},
}
VARIANT_PATH = {
    "fp32": LM_ONNX,
    "fp16": LM_FP16_ONNX,
    "int8": LM_INT8_ONNX,
    "int8hq": LM_INT8HQ_ONNX,
    "int4": LM_INT4_ONNX,
}


def _block_mask(batch: int, seq: int, device) -> torch.Tensor:
    """Build a 4-D bool mask matching the structure used by
    ``OmniVoice._generate_iterative``: each item in the batch can either be
    full-attend (``True`` everywhere) or a "diagonal-pad" pattern with the
    real prefix attending fully to itself and the padded tail attending only
    to its own position.  We mix both styles so the verification covers the
    real generate-time topology, not just trivial all-true masks.
    """
    mask = torch.zeros(batch, 1, seq, seq, dtype=torch.bool, device=device)
    for i in range(batch):
        valid = max(1, seq - i * (seq // (2 * batch)))  # different valid len per item
        mask[i, 0, :valid, :valid] = True
        # diagonal padding so masked positions still have at least one key
        if valid < seq:
            diag = torch.arange(valid, seq, device=device)
            mask[i, 0, diag, diag] = True
    return mask


def run_case(wrapper, sess, device, batch: int, seq: int, seed: int, tol: dict) -> bool:
    log.info("Case  batch=%d  seq=%d  seed=%d", batch, seq, seed)
    torch.manual_seed(seed)

    input_ids, audio_mask, _, position_ids = make_dummy_inputs(
        wrapper.model, device, batch=batch, seq=seq
    )
    # Override the trivial all-true mask with a generate-style block mask
    attention_mask = _block_mask(batch, seq, device)

    with torch.inference_mode():
        ref = wrapper(input_ids, audio_mask, attention_mask, position_ids)

    feeds = {
        "input_ids":      input_ids.cpu().numpy().astype(np.int64),
        "audio_mask":     audio_mask.cpu().numpy().astype(bool),
        "attention_mask": attention_mask.cpu().numpy().astype(bool),
        "position_ids":   position_ids.cpu().numpy().astype(np.int64),
    }
    out = sess.run(["logits"], feeds)[0]

    stats = report_diff("logits", ref, torch.from_numpy(out))

    # Also compare argmax (token decisions) — most important for downstream quality
    ref_argmax = ref.argmax(dim=-1).cpu().numpy()
    out_argmax = out.argmax(axis=-1)
    match = float((ref_argmax == out_argmax).mean())
    print(f"  [argmax-agreement] {match * 100:.4f} %")

    ok = (stats["abs_max"] <= tol["abs_max"]
          and stats["cos"] >= tol["cos"]
          and match >= tol["argmax_min"])
    print(f"  -> {'PASS' if ok else 'FAIL'}  "
          f"(abs_max <= {tol['abs_max']:g}, cos >= {tol['cos']:g}, argmax >= {tol['argmax_min']:.3f})")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=None, help="PyTorch device for reference run")
    p.add_argument("--variant", default="fp32",
                   choices=["fp32", "fp16", "int8", "int8hq", "int4"],
                   help="Which exported ONNX variant to verify against PyTorch FP32")
    p.add_argument("--ort-provider", default="CPUExecutionProvider",
                   choices=["CPUExecutionProvider", "CUDAExecutionProvider"])
    args = p.parse_args()

    onnx_path = VARIANT_PATH[args.variant]
    if not onnx_path.exists():
        log.error("Missing %s — please run export_lm.py / quantize.py first.", onnx_path)
        sys.exit(2)
    tol = TOLERANCES[args.variant]

    device = pick_device(args.device)
    log.info("PyTorch device: %s   ORT provider: %s   variant: %s",
             device, args.ort_provider, args.variant)

    log.info("Loading PyTorch model ...")
    model = load_omnivoice(device, dtype=torch.float32)
    wrapper = OmniVoiceLMWrapper(model).eval()

    log.info("Loading ONNX session: %s", onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=[args.ort_provider])

    cases = [
        (1, 32, 0),
        (1, 64, 1),
        (1, 128, 2),
        (2, 96, 3),
    ]
    all_ok = True
    for b, s, seed in cases:
        all_ok &= run_case(wrapper, sess, device, b, s, seed, tol)

    print()
    if all_ok:
        log.info("ALL CASES PASSED ✓  (variant=%s)", args.variant)
        sys.exit(0)
    else:
        log.error("ONE OR MORE CASES FAILED ✗  (variant=%s)", args.variant)
        sys.exit(1)


if __name__ == "__main__":
    main()

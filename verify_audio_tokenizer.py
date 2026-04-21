#!/usr/bin/env python3
"""Compare audio_tokenizer ONNX (encoder + decoder) against PyTorch reference.

* Encoder: feeds the SAME random + the SAME real audio (assert/andelie.wav)
  through both backends and checks integer-exact codes match.
* Decoder: feeds the SAME random codes through both backends and checks the
  reconstructed waveform diff is small.

Run:
    conda activate tts
    python verify_audio_tokenizer.py [--device cuda:0]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

from _common import (
    AT_DEC_ONNX,
    AT_ENC_ONNX,
    REF_AUDIO,
    load_audio_tokenizer,
    pick_device,
    report_diff,
    setup_logging,
)
from export_audio_tokenizer import AudioDecoderWrapper, AudioEncoderWrapper

log = setup_logging("verify_audio_tokenizer")

ENC_TOL_MISMATCH_RATIO = 0.01   # ≤ 1 % code mismatch allowed (CPU/GPU rounding)
DEC_ABS_MAX_TOL = 5e-3
DEC_COS_MIN_TOL = 0.999


def _load_real_audio(target_sr: int, max_seconds: float = 5.0, device=None) -> torch.Tensor:
    if not REF_AUDIO.exists():
        return None
    wav, sr = torchaudio.load(str(REF_AUDIO))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    max_n = int(max_seconds * target_sr)
    wav = wav[:, :max_n]
    # Round to multiple of hop_length (960 samples) so encoder doesn't pad
    hop = 960
    n = (wav.size(-1) // hop) * hop
    wav = wav[:, :n]
    return wav.unsqueeze(0).to(device)  # [1, 1, N]


def verify_encoder(tok, device, ort_provider) -> bool:
    log.info("=" * 70)
    log.info("Verifying ENCODER")
    log.info("=" * 70)
    enc = AudioEncoderWrapper(tok).eval()

    sess = ort.InferenceSession(str(AT_ENC_ONNX), providers=[ort_provider])

    cases = []
    torch.manual_seed(0)
    cases.append(("random  2.0s", torch.randn(1, 1, 48000, dtype=torch.float32, device=device) * 0.05))
    torch.manual_seed(1)
    cases.append(("random  1.0s", torch.randn(1, 1, 24000, dtype=torch.float32, device=device) * 0.05))
    real = _load_real_audio(tok.config.sample_rate, max_seconds=4.0, device=device)
    if real is not None:
        cases.append((f"andelie {real.size(-1)/tok.config.sample_rate:.2f}s", real))

    all_ok = True
    for name, audio in cases:
        log.info("Case  %s  shape=%s", name, tuple(audio.shape))
        with torch.inference_mode():
            ref = enc(audio)  # int64 [1, 8, T]
        out = sess.run(["audio_codes"], {"audio": audio.cpu().numpy().astype(np.float32)})[0]

        if out.shape != tuple(ref.shape):
            log.error("Shape mismatch  ref=%s  ort=%s", tuple(ref.shape), out.shape)
            all_ok = False
            continue

        ref_np = ref.cpu().numpy().astype(np.int64)
        mismatch_ratio = float((out.astype(np.int64) != ref_np).mean())
        match_pct = (1.0 - mismatch_ratio) * 100.0
        ok = mismatch_ratio <= ENC_TOL_MISMATCH_RATIO
        all_ok &= ok
        print(f"  shape={out.shape}  exact-match={match_pct:.4f}%  mismatch={mismatch_ratio:.4e}  -> {'PASS' if ok else 'FAIL'}")

    return all_ok


def verify_decoder(tok, device, ort_provider) -> bool:
    log.info("=" * 70)
    log.info("Verifying DECODER")
    log.info("=" * 70)
    dec = AudioDecoderWrapper(tok).eval()
    sess = ort.InferenceSession(str(AT_DEC_ONNX), providers=[ort_provider])

    all_ok = True
    for name, T, seed in [("50f", 50, 10), ("100f", 100, 11), ("25f", 25, 12)]:
        log.info("Case  %s  num_frames=%d", name, T)
        torch.manual_seed(seed)
        codes = torch.randint(
            0, tok.config.codebook_size, (1, 8, T), dtype=torch.long, device=device,
        )
        with torch.inference_mode():
            ref = dec(codes)  # float32 [1, 1, N]
        out = sess.run(["audio"], {"audio_codes": codes.cpu().numpy().astype(np.int64)})[0]

        stats = report_diff("audio", ref, torch.from_numpy(out))
        ok = stats["abs_max"] <= DEC_ABS_MAX_TOL and stats["cos"] >= DEC_COS_MIN_TOL
        all_ok &= ok
        print(f"  -> {'PASS' if ok else 'FAIL'}")

    return all_ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=None)
    p.add_argument("--ort-provider", default="CPUExecutionProvider",
                   choices=["CPUExecutionProvider", "CUDAExecutionProvider"])
    p.add_argument("--only", choices=("encoder", "decoder"), default=None)
    args = p.parse_args()

    if not AT_ENC_ONNX.exists() or not AT_DEC_ONNX.exists():
        log.error("Missing tokenizer ONNX files. Run export_audio_tokenizer.py first.")
        sys.exit(2)

    device = pick_device(args.device)
    log.info("Loading PyTorch tokenizer on %s ...", device)
    tok = load_audio_tokenizer(device, dtype=torch.float32)

    all_ok = True
    if args.only in (None, "encoder"):
        all_ok &= verify_encoder(tok, device, args.ort_provider)
    if args.only in (None, "decoder"):
        all_ok &= verify_decoder(tok, device, args.ort_provider)

    if all_ok:
        log.info("ALL CASES PASSED ✓")
        sys.exit(0)
    else:
        log.error("ONE OR MORE CASES FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()

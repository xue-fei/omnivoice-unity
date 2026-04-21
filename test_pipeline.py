#!/usr/bin/env python3
"""End-to-end pipeline test:  PyTorch  vs  ONNX  on a real reference clip.

Pipeline:
    waveform (assert/andelie.wav)
        --[encoder]-->  audio_codes  [1, 8, T]
        --[decoder]-->  reconstructed waveform
    reconstructed_pt  vs  reconstructed_onnx     (encoder + decoder agreement)

We deliberately do NOT include the LM in this round-trip, because the LM is
non-deterministic (Gumbel sampling) at inference time. The LM is verified
separately in verify_lm.py against fixed dummy inputs.

Run:
    conda activate tts
    python test_pipeline.py [--device cuda:0]

Outputs:
    output/test_outputs/
        ref_input.wav        # the trimmed input
        recon_pytorch.wav    # PyTorch encoder + decoder
        recon_onnx.wav       # ONNX     encoder + decoder
        diff.txt             # numerical comparison report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio

from _common import (
    AT_DEC_ONNX,
    AT_ENC_ONNX,
    OUTPUT_DIR,
    REF_AUDIO,
    load_audio_tokenizer,
    pick_device,
    report_diff,
    setup_logging,
)

log = setup_logging("test_pipeline")


def _load_wav(target_sr: int, max_seconds: float, hop: int) -> torch.Tensor:
    if not REF_AUDIO.exists():
        raise FileNotFoundError(f"Missing reference audio: {REF_AUDIO}")
    wav, sr = torchaudio.load(str(REF_AUDIO))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    n = min(int(max_seconds * target_sr), wav.size(-1))
    n = (n // hop) * hop  # round to whole frames
    wav = wav[:, :n]
    return wav.unsqueeze(0)  # [1, 1, N]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=None)
    p.add_argument("--ort-provider", default="CPUExecutionProvider",
                   choices=["CPUExecutionProvider", "CUDAExecutionProvider"])
    p.add_argument("--seconds", type=float, default=5.0,
                   help="Clip the reference to this many seconds (default 5)")
    args = p.parse_args()

    if not (AT_ENC_ONNX.exists() and AT_DEC_ONNX.exists()):
        log.error("Missing audio_tokenizer ONNX files. Run export_audio_tokenizer.py first.")
        sys.exit(2)

    out_dir = OUTPUT_DIR / "test_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    log.info("Device: %s   ORT provider: %s", device, args.ort_provider)

    log.info("Loading PyTorch audio tokenizer ...")
    tok = load_audio_tokenizer(device, dtype=torch.float32)
    sr = tok.config.sample_rate
    hop = tok.config.hop_length
    log.info("sr=%d  frame_rate=%d  hop=%d", sr, tok.config.frame_rate, hop)

    log.info("Loading reference clip %s (max %.1fs) ...", REF_AUDIO, args.seconds)
    wav = _load_wav(sr, args.seconds, hop).to(device)
    log.info("Input audio shape=%s  duration=%.2fs", tuple(wav.shape), wav.size(-1) / sr)
    sf.write(str(out_dir / "ref_input.wav"),
             wav.squeeze().cpu().numpy(), sr)

    # ---- PyTorch reference ----
    log.info("[PyTorch] encode → decode ...")
    with torch.inference_mode():
        codes_pt = tok.encode(wav, return_dict=True).audio_codes
        recon_pt = tok.decode(codes_pt, return_dict=True).audio_values
    log.info("  codes_pt=%s  recon_pt=%s", tuple(codes_pt.shape), tuple(recon_pt.shape))
    sf.write(str(out_dir / "recon_pytorch.wav"),
             recon_pt.squeeze().cpu().numpy(), sr)

    # ---- ONNX runtime ----
    log.info("[ONNX] encode → decode ...")
    enc_sess = ort.InferenceSession(str(AT_ENC_ONNX), providers=[args.ort_provider])
    dec_sess = ort.InferenceSession(str(AT_DEC_ONNX), providers=[args.ort_provider])
    codes_ox = enc_sess.run(["audio_codes"],
                            {"audio": wav.cpu().numpy().astype(np.float32)})[0]
    recon_ox = dec_sess.run(["audio"],
                            {"audio_codes": codes_ox.astype(np.int64)})[0]
    log.info("  codes_ox=%s  recon_ox=%s", codes_ox.shape, recon_ox.shape)
    sf.write(str(out_dir / "recon_onnx.wav"),
             recon_ox.squeeze(), sr)

    # ---- Compare ----
    print()
    log.info("Numerical comparison (PyTorch  vs  ONNX):")
    code_match = float((codes_ox.astype(np.int64) == codes_pt.cpu().numpy().astype(np.int64)).mean())
    print(f"  [audio_codes] exact-match = {code_match * 100:.4f} %  shape={codes_ox.shape}")

    audio_stats = report_diff("recon_audio", recon_pt, torch.from_numpy(recon_ox))

    # Also compare reconstructed audio against the original input as a sanity baseline
    recon_pt_trim = recon_pt[..., : wav.size(-1)]
    if recon_pt_trim.shape == wav.shape:
        rms_in = float(wav.pow(2).mean().sqrt())
        rms_err_pt = float((wav - recon_pt_trim).pow(2).mean().sqrt())
        print(f"  [reconstruction quality]  RMS(input)={rms_in:.4f}  "
              f"RMS(input-recon_pt)={rms_err_pt:.4f}")

    # Persist a small textual report
    with open(out_dir / "diff.txt", "w") as f:
        f.write(f"audio_codes exact-match: {code_match*100:.4f}%\n")
        f.write(f"recon audio max|Δ|: {audio_stats['abs_max']:.6e}\n")
        f.write(f"recon audio mean|Δ|: {audio_stats['abs_mean']:.6e}\n")
        f.write(f"recon audio cosine: {audio_stats['cos']:.8f}\n")

    print()
    ok = code_match >= 0.99 and audio_stats["abs_max"] <= 5e-3 and audio_stats["cos"] >= 0.999
    if ok:
        log.info("PIPELINE TEST PASSED ✓   outputs in %s", out_dir)
        sys.exit(0)
    else:
        log.error("PIPELINE TEST FAILED ✗   see %s/diff.txt", out_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()

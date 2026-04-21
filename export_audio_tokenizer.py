#!/usr/bin/env python3
"""Export the HiggsAudioV2 audio tokenizer (encode + decode) to ONNX (FP32).

Produces TWO separate ONNX files:

  output/audio_tokenizer_encoder/model.onnx
      input  : audio          float32 [batch, 1, num_samples]   (24 kHz mono)
      output : audio_codes    int64   [batch, 8, num_frames]    (frame_rate = 25 Hz)

  output/audio_tokenizer_decoder/model.onnx
      input  : audio_codes    int64   [batch, 8, num_frames]
      output : audio          float32 [batch, 1, num_samples]

Run:
    conda activate tts
    python export_audio_tokenizer.py [--device cuda:0] [--no-optimize]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from _common import (
    AT_DEC_ONNX,
    AT_DEC_OUT_DIR,
    AT_ENC_ONNX,
    AT_ENC_OUT_DIR,
    ONNX_OPSET,
    ensure_dirs,
    file_size,
    human_size,
    load_audio_tokenizer,
    pick_device,
    setup_logging,
)

log = setup_logging("export_audio_tokenizer")


# ---------------------------------------------------------------------------
# Trace-friendly wrappers
# ---------------------------------------------------------------------------


class AudioEncoderWrapper(nn.Module):
    """Wraps ``HiggsAudioV2TokenizerModel.encode`` so that:
      * the function returns a single tensor (no dataclass)
      * default bandwidth is baked in
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        out = self.tokenizer.encode(audio, return_dict=True)
        return out.audio_codes  # [B, 8, T]


class AudioDecoderWrapper(nn.Module):
    """Wraps ``HiggsAudioV2TokenizerModel.decode`` to return raw audio tensor."""

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        out = self.tokenizer.decode(audio_codes, return_dict=True)
        return out.audio_values  # [B, 1, N]


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _save_with_external_data(out_path: Path) -> None:
    proto = onnx.load(str(out_path), load_external_data=True)
    for f in out_path.parent.glob(f"{out_path.name}*"):
        f.unlink()
    onnx.save_model(
        proto,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{out_path.name}_data",
        size_threshold=1024,
    )


def _optimize(out_path: Path) -> None:
    log.info("[%s] Running onnxsim ...", out_path.name)
    try:
        import onnxsim

        proto = onnx.load(str(out_path), load_external_data=True)
        simplified, ok = onnxsim.simplify(
            proto, check_n=0, perform_optimization=True, skip_shape_inference=False
        )
        if ok:
            for f in out_path.parent.glob(f"{out_path.name}*"):
                f.unlink()
            onnx.save_model(
                simplified,
                str(out_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{out_path.name}_data",
                size_threshold=1024,
            )
            log.info("[%s] onnxsim OK  size=%s", out_path.name, human_size(file_size(out_path)))
        else:
            log.warning("[%s] onnxsim returned failure; keeping original.", out_path.name)
    except Exception as e:
        log.warning("[%s] onnxsim skipped: %s", out_path.name, e)


def _smoke_check(out_path: Path, dummy_inputs: dict, ref: torch.Tensor, output_name: str) -> None:
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    out = sess.run([output_name], dummy_inputs)[0]
    if out.dtype.kind in ("i", "u"):
        diff = float(np.abs(out.astype(np.int64) - ref.cpu().numpy().astype(np.int64)).max())
        log.info("[%s] ORT %s  max|Δ| (int) = %.3e", out_path.name, out.shape, diff)
    else:
        diff = float(np.abs(out - ref.cpu().numpy()).max())
        log.info("[%s] ORT %s  max|Δ| = %.3e", out_path.name, out.shape, diff)


# ---------------------------------------------------------------------------
# Main exports
# ---------------------------------------------------------------------------


def export_encoder(tokenizer, device: torch.device, optimize: bool) -> None:
    log.info("=" * 70)
    log.info("Exporting audio_tokenizer ENCODER")
    log.info("=" * 70)

    enc = AudioEncoderWrapper(tokenizer).eval()
    # 2 s of audio @ 24 kHz = 48000 samples — matches our exploratory probe
    dummy_audio = torch.randn(1, 1, 48000, dtype=torch.float32, device=device) * 0.05

    with torch.inference_mode():
        ref = enc(dummy_audio)
    log.info("PyTorch encode OK  codes=%s  dtype=%s", tuple(ref.shape), ref.dtype)

    for f in AT_ENC_OUT_DIR.glob(f"{AT_ENC_ONNX.name}*"):
        f.unlink()

    log.info("Exporting → %s", AT_ENC_ONNX)
    torch.onnx.export(
        enc,
        (dummy_audio,),
        str(AT_ENC_ONNX),
        input_names=["audio"],
        output_names=["audio_codes"],
        dynamic_axes={
            "audio":       {0: "batch", 2: "num_samples"},
            "audio_codes": {0: "batch", 2: "num_frames"},
        },
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    _save_with_external_data(AT_ENC_ONNX)
    log.info("Encoder size=%s", human_size(file_size(AT_ENC_ONNX)))

    if optimize:
        _optimize(AT_ENC_ONNX)

    _smoke_check(
        AT_ENC_ONNX,
        {"audio": dummy_audio.cpu().numpy()},
        ref,
        "audio_codes",
    )


def export_decoder(tokenizer, device: torch.device, optimize: bool) -> None:
    log.info("=" * 70)
    log.info("Exporting audio_tokenizer DECODER")
    log.info("=" * 70)

    dec = AudioDecoderWrapper(tokenizer).eval()
    # 50 frames @ 25 Hz = 2 s
    dummy_codes = torch.randint(
        0, tokenizer.config.codebook_size, (1, 8, 50),
        dtype=torch.long, device=device,
    )

    with torch.inference_mode():
        ref = dec(dummy_codes)
    log.info("PyTorch decode OK  audio=%s  dtype=%s", tuple(ref.shape), ref.dtype)

    for f in AT_DEC_OUT_DIR.glob(f"{AT_DEC_ONNX.name}*"):
        f.unlink()

    log.info("Exporting → %s", AT_DEC_ONNX)
    torch.onnx.export(
        dec,
        (dummy_codes,),
        str(AT_DEC_ONNX),
        input_names=["audio_codes"],
        output_names=["audio"],
        dynamic_axes={
            "audio_codes": {0: "batch", 2: "num_frames"},
            "audio":       {0: "batch", 2: "num_samples"},
        },
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    _save_with_external_data(AT_DEC_ONNX)
    log.info("Decoder size=%s", human_size(file_size(AT_DEC_ONNX)))

    if optimize:
        _optimize(AT_DEC_ONNX)

    _smoke_check(
        AT_DEC_ONNX,
        {"audio_codes": dummy_codes.cpu().numpy()},
        ref,
        "audio",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    p.add_argument("--no-optimize", dest="optimize", action="store_false",
                   help="Skip onnxsim graph optimization")
    p.add_argument("--only", choices=("encoder", "decoder"), default=None,
                   help="Export only one side")
    p.set_defaults(optimize=True)
    args = p.parse_args()

    ensure_dirs()
    device = pick_device(args.device)
    log.info("Using device: %s", device)

    log.info("Loading HiggsAudioV2TokenizerModel ...")
    tok = load_audio_tokenizer(device, dtype=torch.float32)
    log.info("Loaded.  sample_rate=%d  frame_rate=%d  codebook_size=%d",
             tok.config.sample_rate, tok.config.frame_rate, tok.config.codebook_size)

    if args.only in (None, "encoder"):
        export_encoder(tok, device, args.optimize)
    if args.only in (None, "decoder"):
        export_decoder(tok, device, args.optimize)

    log.info("Done.")


if __name__ == "__main__":
    main()

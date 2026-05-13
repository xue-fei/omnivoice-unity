#!/usr/bin/env python3
"""End-to-end ONNX inference for OmniVoice.

Drives ``OmniVoice.generate(...)`` with **all neural-network modules replaced by
ONNX Runtime sessions** (FP32 LM + FP32 audio-tokenizer encoder + FP32
audio-tokenizer decoder).  PyTorch is still loaded but only used for:

  * tokenizer / text preprocessing
  * the iterative diffusion loop (sampling, CFG, top-k selection)
  * audio post-processing (denoising, fade in/out, RMS normalisation)

Three demos are produced in ``output/inference_demo/``:

  1. ``demo_auto.wav``           — no reference, no instruction (auto voice)
  2. ``demo_voice_clone.wav``    — clone the voice from ``zero_short_prompt``
  3. ``demo_voice_design.wav``   — voice design via natural-language style prompt

Run:
    conda activate tts
    python infer_onnx.py [--device cpu] [--num-step 32]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

from _common import (
    AT_DEC_ONNX,
    AT_ENC_ONNX,
    LM_ONNX,
    OUTPUT_DIR,
    PT_MODEL_DIR,
    REF_AUDIO,
    add_project_root_to_path,
    pick_device,
    setup_logging,
    variant_paths,
)

log = setup_logging("infer_onnx")

DEMO_ROOT = OUTPUT_DIR / "inference_demo"

# ----------------------- ONNX Runtime adapters ------------------------------


def _make_session(path: Path, providers: list[str]) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 0  # let ORT decide (=physical cores)
    log.info("Loading ONNX session: %s", path.relative_to(OUTPUT_DIR.parent))
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


class ONNXAdapter:
    """Bundles the three ORT sessions used to back the PyTorch model."""

    def __init__(self, providers: list[str], variant: str = "fp32"):
        lm_path, enc_path, dec_path = variant_paths(variant)
        for p in (lm_path, enc_path, dec_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing ONNX artefact for variant={variant!r}: {p}\n"
                    f"Run: python export_lm.py / export_audio_tokenizer.py / quantize.py"
                )
        log.info("Variant: %s", variant)
        self.variant = variant
        self.lm = _make_session(lm_path, providers)
        self.enc = _make_session(enc_path, providers)
        self.dec = _make_session(dec_path, providers)

    # ---- LM forward (replaces ``OmniVoice.forward``) ---------------------

    def lm_forward(
        self,
        input_ids: torch.Tensor,        # [B, C, S] int64
        audio_mask: torch.Tensor,       # [B, S] bool
        attention_mask: torch.Tensor,   # [B, 1, S, S] bool
        position_ids: torch.Tensor | None = None,  # [B, S] int64
    ) -> torch.Tensor:
        B, _, S = input_ids.shape
        if position_ids is None:
            position_ids = (
                torch.arange(S, device=input_ids.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(B, -1)
                .contiguous()
            )

        feeds = {
            "input_ids":      input_ids.detach().cpu().numpy().astype(np.int64),
            "audio_mask":     audio_mask.detach().cpu().numpy().astype(bool),
            "attention_mask": attention_mask.detach().cpu().numpy().astype(bool),
            "position_ids":   position_ids.detach().cpu().numpy().astype(np.int64),
        }
        out = self.lm.run(["logits"], feeds)[0]
        return torch.from_numpy(out).to(input_ids.device)

    # ---- Audio tokenizer encode / decode --------------------------------

    def at_encode(self, audio: torch.Tensor) -> torch.Tensor:
        """audio : float32 [B, 1, N] @ 24kHz   →   audio_codes : int64 [B, 8, T]"""
        feeds = {"audio": audio.detach().cpu().numpy().astype(np.float32)}
        codes = self.enc.run(["audio_codes"], feeds)[0]
        return torch.from_numpy(codes).to(audio.device).long()

    def at_decode(self, codes: torch.Tensor) -> torch.Tensor:
        """audio_codes : int64 [B, 8, T]   →   audio : float32 [B, 1, N]"""
        feeds = {"audio_codes": codes.detach().cpu().numpy().astype(np.int64)}
        wav = self.dec.run(["audio"], feeds)[0]
        return torch.from_numpy(wav).to(codes.device).float()


# ----------------------- monkey patches -------------------------------------


def patch_omnivoice_with_onnx(model, adapter: ONNXAdapter) -> None:
    """Replace the heavy neural-network parts of ``model`` with ONNX calls.

    Keeps PyTorch tokenizer / sampling / postprocess code path intact.
    """
    # 1) Replace OmniVoice.forward — generate calls self(..., attention_mask=...)
    #    and reads ``.logits`` off the result.
    def _onnx_forward(
        self,
        input_ids,
        audio_mask,
        labels=None,
        attention_mask=None,
        document_ids=None,
        position_ids=None,
    ):
        if attention_mask is None:
            B, _, S = input_ids.shape
            attention_mask = torch.ones(
                B, 1, S, S, dtype=torch.bool, device=input_ids.device
            )
        elif attention_mask.dim() == 2:
            # Promote 2-D padding mask to a 4-D self-attention mask
            keep = attention_mask.to(torch.bool)
            attention_mask = (keep[:, None, None, :] & keep[:, None, :, None])
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        logits = adapter.lm_forward(
            input_ids=input_ids,
            audio_mask=audio_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return SimpleNamespace(logits=logits, loss=None)

    import types
    model.forward = types.MethodType(_onnx_forward, model)
    # nn.Module.__call__ uses self.forward via the bound method, so the patch
    # above is enough; ``self(input_ids=...)`` will hit our function.

    # 2) Replace audio tokenizer encode / decode
    if model.audio_tokenizer is None:
        raise RuntimeError("audio_tokenizer not loaded on model")
    tok = model.audio_tokenizer

    def _onnx_encode(self, audio, *args, **kwargs):
        codes = adapter.at_encode(audio)
        return SimpleNamespace(audio_codes=codes)

    def _onnx_decode(self, audio_codes, *args, **kwargs):
        wav = adapter.at_decode(audio_codes)
        return SimpleNamespace(audio_values=wav)

    tok.encode = types.MethodType(_onnx_encode, tok)
    tok.decode = types.MethodType(_onnx_decode, tok)
    log.info("Patched OmniVoice.forward + audio_tokenizer.{encode,decode} → ONNX Runtime")


# ----------------------- model loading --------------------------------------


def load_pytorch_model(device: torch.device, dtype: torch.dtype = torch.float32):
    """Load the FULL OmniVoice (with audio_tokenizer + text_tokenizer)."""
    add_project_root_to_path()
    from omnivoice.models.omnivoice import OmniVoice

    log.info("Loading PyTorch OmniVoice from %s ...", PT_MODEL_DIR)
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        str(PT_MODEL_DIR),
        device_map={"": device.type if device.type != "cuda" else f"cuda:{device.index or 0}"},
        dtype=dtype,
        attn_implementation="eager",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("Loaded in %.1fs  device=%s  dtype=%s", time.time() - t0, device, dtype)
    return model


# ----------------------- demos ----------------------------------------------

REF_TEXT = (
    "希望你以后过的比我还好哟！"
)

DEMOS = [
    {
        "name": "demo_auto",
        "mode": "auto",
        "text": "Hello, this is OmniVoice running entirely on ONNX Runtime, "
                "with the language model and the audio tokenizer both exported "
                "to FP32 ONNX format.",
        "language": "English",
    },
    {
        "name": "demo_voice_clone",
        "mode": "voice_clone",
        "text": "你好，欢迎来到 OmniVoice。这是一段使用语音克隆模式生成的中文语音。",
        "language": "Chinese",
        "ref_audio": str(REF_AUDIO),
        "ref_text": REF_TEXT,
    },
    {
        "name": "demo_voice_design",
        "mode": "voice_design",
        "text": "Welcome to OmniVoice. This sample is generated with a voice "
                "design instruction, no reference audio required.",
        "language": "English",
        # NOTE: ``instruct`` must use the closed vocabulary defined by the
        # model.  See ``omnivoice/models/omnivoice.py::_resolve_instruct`` for
        # the full list (American/British/Indian/Chinese accents, male/female,
        # young adult/middle-aged/elderly, low/high pitch, whisper, ...).
        "instruct": "male, british accent, low pitch, middle-aged",
    },
]


def run_demo(model, demo: dict, gen_kwargs: dict, demo_dir: Path) -> Path:
    out_path = demo_dir / f"{demo['name']}.wav"
    log.info("== %s (%s) ==", demo["name"], demo["mode"])
    log.info("  text     : %s", demo["text"][:90] + ("..." if len(demo["text"]) > 90 else ""))
    if "ref_audio" in demo:
        log.info("  ref_audio: %s", demo["ref_audio"])
    if "instruct" in demo:
        log.info("  instruct : %s", demo["instruct"])

    kwargs = dict(
        text=demo["text"],
        language=demo.get("language"),
        ref_audio=demo.get("ref_audio"),
        ref_text=demo.get("ref_text"),
        instruct=demo.get("instruct"),
        **gen_kwargs,
    )
    t0 = time.time()
    audios = model.generate(**kwargs)
    elapsed = time.time() - t0

    audio = audios[0]  # [1, T]
    dur = audio.size(-1) / model.sampling_rate
    log.info(
        "  → wrote %s   audio=%.2fs  wallclock=%.1fs  RTF=%.2f",
        out_path.name,
        dur,
        elapsed,
        elapsed / max(dur, 1e-6),
    )
    torchaudio.save(str(out_path), audio.cpu(), model.sampling_rate)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", help="cpu or cuda:N for the PyTorch side")
    p.add_argument("--num-step", type=int, default=32)
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--t-shift", type=float, default=0.1)
    p.add_argument("--ort-provider", default="CPUExecutionProvider",
                   choices=["CPUExecutionProvider", "CUDAExecutionProvider"])
    p.add_argument("--only", default=None,
                   help="Run only one demo by name (e.g. demo_voice_design)")
    p.add_argument("--variant", default="fp32",
                   choices=["fp32", "int8", "int8hq", "fp16", "int4"],
                   help="Which ONNX precision to use for the LM (and the AT-encoder/decoder if INT8 versions exist)")
    args = p.parse_args()

    device = pick_device(args.device)
    demo_dir = DEMO_ROOT / args.variant
    demo_dir.mkdir(parents=True, exist_ok=True)

    adapter = ONNXAdapter([args.ort_provider], variant=args.variant)
    model = load_pytorch_model(device)
    patch_omnivoice_with_onnx(model, adapter)

    gen_kwargs = dict(
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        t_shift=args.t_shift,
        denoise=True,
        postprocess_output=True,
    )

    demos = DEMOS
    if args.only:
        demos = [d for d in DEMOS if d["name"] == args.only]
        if not demos:
            raise SystemExit(f"--only={args.only} not in {[d['name'] for d in DEMOS]}")

    log.info("=" * 60)
    log.info("Running %d demos (variant=%s) → %s", len(demos), args.variant, demo_dir)
    log.info("=" * 60)

    results = []
    for demo in demos:
        try:
            results.append(run_demo(model, demo, gen_kwargs, demo_dir))
        except Exception as e:
            log.exception("Demo %s FAILED: %s", demo["name"], e)
            results.append(None)

    log.info("=" * 60)
    log.info("Summary (variant=%s):", args.variant)
    for demo, res in zip(demos, results):
        status = res.name if res else "FAILED"
        log.info("  %-22s  %s", demo["name"], status)
    log.info("All outputs in: %s", demo_dir)


if __name__ == "__main__":
    main()

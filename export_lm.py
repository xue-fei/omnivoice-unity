#!/usr/bin/env python3
"""Export the OmniVoice language model (the diffusion LM stage) to ONNX (FP32).

Inputs / outputs are designed to be a drop-in replacement for the PyTorch
``OmniVoice.forward(...)`` call inside ``_generate_iterative``:

  Inputs:
    input_ids      : int64 [batch, num_codebooks=8, sequence]
    audio_mask     : bool  [batch, sequence]
    attention_mask : bool  [batch, 1, sequence, sequence]
                     (4-D causal/block mask, as built by `_generate_iterative`,
                      `True`  = attend, `False` = mask out)
    position_ids   : int64 [batch, sequence]

  Output:
    logits : float32 [batch, num_codebooks=8, sequence, audio_vocab_size=1025]

Run:
    conda activate tts
    python export_lm.py [--device cuda:0] [--no-optimize]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from _common import (
    LM_ONNX,
    LM_OUT_DIR,
    ONNX_OPSET,
    ensure_dirs,
    file_size,
    human_size,
    load_omnivoice,
    pick_device,
    report_node_stats,
    setup_logging,
)

log = setup_logging("export_lm")


# ---------------------------------------------------------------------------
# A trace-friendly wrapper around OmniVoice that:
#  * accepts a 2D HF-style attention_mask (matches the existing qint8 ONNX)
#  * accepts explicit position_ids
#  * returns raw logits tensor (no dataclass)
# ---------------------------------------------------------------------------


class OmniVoiceLMWrapper(nn.Module):
    """Trace-friendly wrapper around the OmniVoice LM.

    We bypass HuggingFace's ``create_causal_mask`` by passing the
    ``attention_mask`` argument as a ``dict`` (Qwen3 has a fast-path for
    pre-built mask mappings). The dict carries a 4-D **additive** float bias
    built from the 4-D **boolean** mask the caller provides. This matches what
    ``OmniVoice._generate_iterative`` already constructs (a per-batch block
    mask of shape ``[2B, 1, S, S]``) and keeps ``batch`` and ``sequence``
    fully dynamic in the exported graph.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @staticmethod
    def _bool_mask_to_bias(attention_mask_4d: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Convert a 4-D bool mask ``[B, 1, Sq, Sk]`` (True = attend) to a
        4-D additive bias of the same shape (0 / -inf) ready for SDPA.
        """
        keep = attention_mask_4d.to(dtype=torch.bool)
        bias = torch.zeros(keep.shape, dtype=dtype, device=keep.device)
        neg_inf = torch.finfo(dtype).min
        bias = bias.masked_fill(~keep, neg_inf)
        return bias

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, C, S] int64
        audio_mask: torch.Tensor,       # [B, S] bool
        attention_mask: torch.Tensor,   # [B, 1, S, S] bool
        position_ids: torch.Tensor,     # [B, S] int64
    ) -> torch.Tensor:
        inputs_embeds = self.model._prepare_embed_inputs(input_ids, audio_mask)

        attn_bias_4d = self._bool_mask_to_bias(attention_mask, inputs_embeds.dtype)
        # Qwen3 detects a pre-built mapping when ``attention_mask`` is a dict.
        causal_mask_mapping = {"full_attention": attn_bias_4d}

        llm_outputs = self.model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            return_dict=True,
        )
        hidden_states = llm_outputs[0]

        B, S, _ = hidden_states.shape
        logits_flat = self.model.audio_heads(hidden_states)
        # [B, S, C*V] -> [B, S, C, V] -> [B, C, S, V]
        audio_logits = logits_flat.view(
            B, S,
            self.model.config.num_audio_codebook,
            self.model.config.audio_vocab_size,
        ).permute(0, 2, 1, 3).contiguous()
        return audio_logits


def make_dummy_inputs(model, device: torch.device, batch: int = 1, seq: int = 64):
    """Build a tiny but realistic dummy batch (mix of text + audio mask positions)."""
    cfg = model.config
    C = cfg.num_audio_codebook
    V = cfg.audio_vocab_size
    txt_vocab = cfg.llm_config.vocab_size
    text_len = seq // 2
    audio_len = seq - text_len

    # Layer 0 holds either text tokens or shifted audio tokens depending on mask.
    txt_ids = torch.randint(0, txt_vocab, (batch, 1, text_len), device=device, dtype=torch.long)
    aud_ids = torch.randint(0, V - 1, (batch, 1, audio_len), device=device, dtype=torch.long)
    layer0 = torch.cat([txt_ids, aud_ids], dim=-1)

    # Layers 1..C-1 only matter on audio positions (gathered through audio_embeddings).
    other = torch.randint(0, V - 1, (batch, C - 1, seq), device=device, dtype=torch.long)
    input_ids = torch.cat([layer0, other], dim=1).contiguous()  # [B, C, S]

    audio_mask = torch.zeros(batch, seq, dtype=torch.bool, device=device)
    audio_mask[:, text_len:] = True

    # 4-D bool mask, full-attend (True everywhere).  Real generate-time masks
    # are block-diagonal but tracing only sees one shape — we build the most
    # generic "attend to everything" pattern so the resulting graph supports
    # any pattern the caller injects at runtime.
    attention_mask = torch.ones(batch, 1, seq, seq, dtype=torch.bool, device=device)
    position_ids = torch.arange(seq, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1).contiguous()

    return input_ids, audio_mask, attention_mask, position_ids


def _purge_output_dir(keep: set[str]) -> None:
    """Delete every file in ``LM_OUT_DIR`` whose name is not in ``keep``."""
    for f in LM_OUT_DIR.iterdir():
        if f.is_file() and f.name not in keep:
            try:
                f.unlink()
            except OSError as exc:
                log.warning("Could not delete %s: %s", f, exc)


def export(args):
    ensure_dirs()
    device = pick_device(args.device)
    log.info("Using device: %s", device)

    log.info("Loading OmniVoice (PyTorch FP32) from local checkpoint ...")
    model = load_omnivoice(device, dtype=torch.float32)
    wrapper = OmniVoiceLMWrapper(model).eval()

    dummy = make_dummy_inputs(model, device, batch=1, seq=64)
    input_ids, audio_mask, attention_mask, position_ids = dummy

    # Sanity check — make sure the wrapped forward runs in PyTorch first
    with torch.inference_mode():
        ref = wrapper(input_ids, audio_mask, attention_mask, position_ids)
    log.info("PyTorch forward OK  logits=%s  dtype=%s", tuple(ref.shape), ref.dtype)

    if LM_ONNX.exists():
        log.info("Removing existing %s", LM_ONNX)
        for f in LM_OUT_DIR.glob("model.onnx*"):
            f.unlink()

    log.info("Exporting to ONNX  opset=%d  →  %s", ONNX_OPSET, LM_ONNX)
    torch.onnx.export(
        wrapper,
        (input_ids, audio_mask, attention_mask, position_ids),
        str(LM_ONNX),
        input_names=["input_ids", "audio_mask", "attention_mask", "position_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch", 2: "sequence"},
            "audio_mask":     {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 2: "sequence", 3: "sequence"},
            "position_ids":   {0: "batch", 1: "sequence"},
            "logits":         {0: "batch", 2: "sequence"},
        },
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,  # The legacy TorchScript path is more reliable for HF models
    )

    raw_size = file_size(LM_ONNX)
    log.info("Exported raw ONNX  size=%s", human_size(raw_size))

    # Re-save with external data so the model is well-formed for ORT (>2 GB).
    # ``torch.onnx.export`` writes one sidecar file *per* tensor (e.g.
    # ``onnx__MatMul_8211``, ``model.llm.embed_tokens.weight`` ...), which
    # leaves hundreds of stray files in the output directory.  We re-load
    # everything into memory, then re-save with ``all_tensors_to_one_file``
    # and wipe everything that is not the canonical pair.
    log.info("Re-saving with external data layout (single sidecar 'model.onnx_data') ...")
    proto = onnx.load(str(LM_ONNX), load_external_data=True)
    _purge_output_dir(keep={"model.onnx"})  # drop main file + all stray sidecars
    onnx.save_model(
        proto,
        str(LM_ONNX),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
        size_threshold=1024,
    )
    _purge_output_dir(keep={"model.onnx", "model.onnx_data"})
    log.info("Saved.  total=%s", human_size(file_size(LM_ONNX)))

    log.info("Node stats BEFORE optimization:")
    report_node_stats(LM_ONNX)

    if args.optimize:
        optimize_lm()
        log.info("Node stats AFTER optimization:")
        report_node_stats(LM_ONNX)
    else:
        log.info("Skipping optimization (--no-optimize).")

    # Quick round-trip check via ORT to make sure the exported file actually loads
    _smoke_check_with_ort(dummy, ref)
    log.info("Done.")


def optimize_lm() -> None:
    """Apply onnxsim simplification + onnxruntime transformer optimizer."""
    log.info("Optimizing graph (onnxsim) ...")
    try:
        import onnxsim

        proto = onnx.load(str(LM_ONNX), load_external_data=True)
        simplified, ok = onnxsim.simplify(
            proto,
            check_n=0,             # skip ORT inference comparison (too slow / heavy here)
            perform_optimization=True,
            skip_shape_inference=False,
        )
        if ok:
            _purge_output_dir(keep=set())
            onnx.save_model(
                simplified,
                str(LM_ONNX),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="model.onnx_data",
                size_threshold=1024,
            )
            _purge_output_dir(keep={"model.onnx", "model.onnx_data"})
            log.info("onnxsim OK  size=%s", human_size(file_size(LM_ONNX)))
        else:
            log.warning("onnxsim reported failure; keeping original.")
    except Exception as e:
        log.warning("onnxsim skipped: %s", e)

    log.info("Optimizing graph (onnxruntime.transformers) ...")
    try:
        from onnxruntime.transformers import optimizer as ort_opt
        from onnxruntime.transformers.fusion_options import FusionOptions

        # Qwen3 uses RMSNorm + RotaryEmbedding + GQA. ORT's "gpt2" / "bert"
        # presets cover LayerNorm/Attention fusions; "gpt2" is the closest
        # to a decoder-only transformer.
        fopt = FusionOptions("gpt2")
        fopt.enable_layer_norm = True            # also matches RMSNorm patterns
        fopt.enable_skip_layer_norm = True
        fopt.enable_attention = True
        fopt.enable_bias_gelu = True
        fopt.enable_gelu = True
        fopt.enable_gemm_fast_gelu = True
        fopt.enable_rotary_embeddings = True

        opt_model = ort_opt.optimize_model(
            str(LM_ONNX),
            model_type="gpt2",
            num_heads=16,           # from config: num_attention_heads
            hidden_size=1024,       # from config: hidden_size
            opt_level=99,
            optimization_options=fopt,
            use_gpu=False,
            only_onnxruntime=False,
        )
        # Save back, again with single external-data sidecar.  ORT names the
        # sidecar ``<model>.data`` by default, but our convention (and the
        # convention used by the existing qint8 ONNX) is ``model.onnx_data``.
        # Round-trip via onnx.save_model to get the naming we want.
        _purge_output_dir(keep=set())
        proto = opt_model.model
        onnx.save_model(
            proto,
            str(LM_ONNX),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx_data",
            size_threshold=1024,
        )
        _purge_output_dir(keep={"model.onnx", "model.onnx_data"})
        log.info("ORT optimizer OK  size=%s", human_size(file_size(LM_ONNX)))
    except Exception as e:
        log.warning("ORT optimizer skipped: %s", e)


def _smoke_check_with_ort(dummy, ref) -> None:
    log.info("Smoke-checking the exported file with onnxruntime ...")
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(LM_ONNX),
        providers=["CPUExecutionProvider"],
    )
    feeds = {
        "input_ids":      dummy[0].cpu().numpy().astype(np.int64),
        "audio_mask":     dummy[1].cpu().numpy().astype(bool),
        "attention_mask": dummy[2].cpu().numpy().astype(bool),
        "position_ids":   dummy[3].cpu().numpy().astype(np.int64),
    }
    out = sess.run(["logits"], feeds)[0]
    diff = np.abs(out - ref.cpu().numpy()).max()
    log.info("ORT logits %s  max|Δ| vs PyTorch = %.3e", out.shape, float(diff))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    p.add_argument("--no-optimize", dest="optimize", action="store_false",
                   help="Skip onnxsim / onnxruntime graph optimization")
    p.set_defaults(optimize=True)
    args = p.parse_args()
    export(args)


if __name__ == "__main__":
    main()

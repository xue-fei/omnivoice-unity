#!/usr/bin/env python3
"""Micro-benchmark for the OmniVoice LM ONNX session.

Sweeps ``intra_op_num_threads`` and the SEQUENTIAL/PARALLEL execution mode
to find the best CPU configuration for a single LM forward at typical
generation shapes.

Why a micro-benchmark and not the full ``infer_onnx.py``?
  - The full pipeline mixes one-shot costs (graph optimisation on first
    run, audio_tokenizer encode, postprocess) with per-step latency, so
    the reported RTF is not a clean signal for tuning ORT threading.
  - The LM accounts for ~95% of generation time at ``num_step=32`` (each
    decode step calls LM once on the full prefix), so optimising LM
    latency directly maps to overall RTF.

Run:
    conda activate tts
    python benchmark_lm.py [--variant int8hq] [--threads 1,4,8,16,24]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from _common import setup_logging, variant_paths, file_size, human_size

log = setup_logging("bench_lm")


def make_inputs(batch: int, seq: int) -> dict:
    """Build a dummy input feed mimicking generate-time shapes."""
    rng = np.random.default_rng(0)
    return {
        "input_ids":      rng.integers(0, 1000, size=(batch, 8, seq), dtype=np.int64),
        "audio_mask":     np.ones((batch, seq), dtype=bool),
        "attention_mask": np.ones((batch, 1, seq, seq), dtype=bool),
        "position_ids":   np.tile(np.arange(seq, dtype=np.int64), (batch, 1)),
    }


def make_session(path: Path, intra: int, inter: int, parallel: bool
                 ) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = intra
    so.inter_op_num_threads = inter
    so.execution_mode = (
        ort.ExecutionMode.ORT_PARALLEL if parallel
        else ort.ExecutionMode.ORT_SEQUENTIAL
    )
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def time_run(sess: ort.InferenceSession, feeds: dict,
             warmup: int = 2, repeat: int = 5) -> tuple[float, float]:
    """Return (median_ms, min_ms) over ``repeat`` runs after warmup."""
    for _ in range(warmup):
        sess.run(["logits"], feeds)
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        sess.run(["logits"], feeds)
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    median = samples[len(samples) // 2]
    return median, samples[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="int8hq",
                   choices=["fp32", "int8", "int8hq", "fp16", "int4"])
    p.add_argument("--threads", default="1,4,8,12,16,24",
                   help="Comma list of intra_op_num_threads to try")
    p.add_argument("--parallel", action="store_true",
                   help="Also test ORT_PARALLEL execution_mode (inter_op=4)")
    p.add_argument("--shapes", default="1x256,1x512,1x1024",
                   help="Comma list of BxS shapes for the LM forward")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()

    lm_path, _, _ = variant_paths(args.variant)
    if not lm_path.exists():
        raise SystemExit(f"Missing {lm_path} — run quantize.py first.")

    log.info("CPU affinity hint: taskset -c 0-15 binds to P-cores only")
    log.info("Variant: %s   file: %s (%s)",
             args.variant, lm_path, human_size(file_size(lm_path)))

    threads_list = [int(t) for t in args.threads.split(",") if t]
    shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes.split(",")]

    log.info("Sweeping shapes=%s  threads=%s  parallel=%s",
             shapes, threads_list, args.parallel)

    # Print header
    print()
    print(f"{'shape':>10} | {'mode':<10} | {'intra':>6} | "
          f"{'median ms':>10} | {'min ms':>10} | {'tokens/s':>10}")
    print("-" * 78)

    for batch, seq in shapes:
        feeds = make_inputs(batch, seq)
        for intra in threads_list:
            modes = [("seq", False)]
            if args.parallel:
                modes.append(("par4", True))
            for mode_name, parallel in modes:
                inter = 4 if parallel else 1
                sess = make_session(lm_path, intra=intra, inter=inter,
                                    parallel=parallel)
                med, mn = time_run(sess, feeds,
                                   warmup=args.warmup, repeat=args.repeat)
                tokens_per_s = (batch * seq) / (med / 1000)
                print(f"{batch}x{seq:<7} | {mode_name:<10} | "
                      f"{intra:>6} | {med:>9.1f}  | {mn:>9.1f}  | "
                      f"{tokens_per_s:>10.0f}")
                del sess  # free OrtEnv worker pool before next config
        print("-" * 78)


if __name__ == "__main__":
    main()

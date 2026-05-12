# OmniVoice ONNX Export / Quantization / Inference

Turn [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) (PyTorch) into ONNX—covering the full pipeline **export → quantization (INT8 / INT8-HQ / INT4 / FP16) → numerical verification → end-to-end inference → performance benchmarking**.

> **Current production recommendation**: `int8hq` (LM 587→611 MB, audio output head kept FP32), numerical cos > 0.9999, argmax agreement > 80 %, perceptually almost indistinguishable from FP32. See [§7.2](#72--int8-hq-recommended-for-production).

[中文版](README.md)

## Prerequisites

This repository contains **only export / quantization / inference scripts**, not model weights or test audio. Before running:

1. **OmniVoice PyTorch model** — Clone the main repo from [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice), download pretrained weights, and obtain an `OmniVoice/` layout (with `model.safetensors` + `audio_tokenizer/`).
2. **`tts` conda environment** — Install dependencies per the main OmniVoice README (`torch`, `transformers>=5.5`, `onnx`, `onnxruntime`, `onnxsim`, `onnxconverter_common`, `torchaudio`).
3. **Reference audio** (optional, only needed for `voice_clone` demos) — Any 24 kHz mono WAV; this document defaults to `assert/andelie.wav`.

Recommended layout:

```
your-workspace/
├── OmniVoice/                # Main k2-fsa/OmniVoice repo (with weights)
│   ├── OmniVoice/             # Weight directory (PT_MODEL_DIR)
│   ├── omnivoice/             # Python package
│   ├── assert/andelie.wav     # Voice-clone reference
│   └── onnx_export/           # ← Clone this repo here
│       ├── _common.py
│       ├── export_lm.py
│       └── ...
```

`_common.py` pins paths at the top: `PROJECT_ROOT = Path(__file__).parent.parent`, `PT_MODEL_DIR = PROJECT_ROOT / "OmniVoice"`. If your layout differs, change those two constants.

## Contents

1. [Overview](#1-overview)
2. [Directory layout](#2-directory-layout)
3. [Quick start](#3-quick-start)
4. [ONNX I/O conventions](#4-onnx-io-conventions)
5. [Numerical verification](#5-numerical-verification)
6. [Graph optimization](#6-graph-optimization)
7. [Quantization options](#7-quantization-options)
8. [End-to-end inference](#8-end-to-end-inference)
9. [Performance / RTF / multithreading](#9-performance--rtf--multithreading)
10. [Troubleshooting and rollback](#10-troubleshooting-and-rollback)

---

## 1. Overview

OmniVoice is a text→speech TTS model with two parts:

- **LM**: A ~600 M-parameter Qwen3-style LM, outputs audio token logits `[B, 8 codebooks, S, 1025]`
- **Audio tokenizer**: HiggsAudioV2—encoder (audio→8 codebook codes) and decoder (codes→audio)

This directory exports those three subgraphs into separate ONNX files and provides five precision variants:

| variant | LM size | Use case |
|---------|---------|----------|
| `fp32` | 2.28 GB | Reference / sanity checks |
| `int8` | 587 MB | Size-first, acceptable minor quality loss |
| **`int8hq`** | **611 MB** | **★ Recommended for production** (near-lossless perceptually, only +24 MB vs int8) |
| `int4` | 586 MB | Experimental: tighter size/speed but clearly worse perceptual quality |
| `fp16` | 1.14 GB | GPU deployments (often slowest on CPU) |

---

## 2. Directory layout

Source tree (artifacts land under `output/` at runtime; `output/` is `.gitignore`d):

```
.
├── README.md                  # Human docs (Chinese)
├── README_EN.md               # This file (English)
├── .gitignore
├── _common.py                 # Shared paths / device / load / diff printing / variant_paths
├── export_lm.py               # Export LM
├── export_audio_tokenizer.py  # Export audio tokenizer encoder + decoder
├── quantize.py                # ★ One-stop INT8 / INT8-HQ / INT4 / FP16
├── verify_lm.py               # ONNX vs PyTorch (LM), per variant
├── verify_audio_tokenizer.py # ONNX vs PyTorch (encoder / decoder)
├── test_pipeline.py           # End-to-end encode→decode using assert/andelie.wav
├── infer_onnx.py              # ★ End-to-end TTS: LM + tokenizer all ONNX
└── benchmark_lm.py            # ★ LM single-step latency micro-benchmark (thread sweep)
```

After runs (not committed):

```
output/
├── omnivoice_lm/                  ← FP32 LM (baseline)
├── omnivoice_lm_int8/             ← INT8 weight-only (W8A32)
├── omnivoice_lm_int8_hq/          ← ★ INT8-HQ: W8A32 + audio_heads FP32
├── omnivoice_lm_int4/             ← INT4 group-wise (MatMulNBits, RTN/HQQ)
├── omnivoice_lm_fp16/             ← FP16 LM (GPU path)
├── audio_tokenizer_encoder/        ← FP32 encoder
├── audio_tokenizer_encoder_int8/   ← INT8 encoder (shared by int8 / int8hq / int4)
├── audio_tokenizer_decoder/        ← FP32 decoder
├── audio_tokenizer_decoder_int8/   ← INT8 decoder (shared by int8 / int8hq / int4)
├── test_outputs/                   ← test_pipeline outputs
└── inference_demo/
    ├── fp32/   demo_auto.wav, demo_voice_clone.wav, demo_voice_design.wav
    ├── int8/   ...
    ├── int8hq/ ...                 ← ★ Perceptual A/B baseline
    ├── int4/   ...
    └── fp16/   ...
```

Each quantized LM lives in its own subdirectory—nothing overwrites anything else. Deleting one folder rolls back that variant only.

---

## 3. Quick start

```bash
conda activate tts
cd path/to/OmniVoice/onnx_export   # See Prerequisites for layout

# 1. FP32 export (one-time; ~5–10 minutes)
python export_audio_tokenizer.py
python export_lm.py

# 2. Numerical checks (FP32)
python verify_audio_tokenizer.py
python verify_lm.py --variant fp32

# 3. End-to-end pipeline (real audio via assert/andelie.wav)
python test_pipeline.py

# 4. Quantize to recommended int8hq (LM only; AT can stay standard INT8)
python quantize.py --targets at_encoder,at_decoder --methods int8 --no-smoke
python quantize.py --targets lm --methods int8 --int8-mode weight_only_hq --no-smoke

# 5. int8hq numerical check
python verify_lm.py --variant int8hq

# 6. ★ Full TTS inference (three perceptual demos)
python infer_onnx.py --variant int8hq
```

---

## 4. ONNX I/O conventions

### 4.1 `omnivoice_lm/model.onnx` (shared across variants)

| name | dtype | shape |
|------|-------|-------|
| **in** `input_ids` | int64 | `[batch, 8, sequence]` |
| **in** `audio_mask` | bool | `[batch, sequence]` |
| **in** `attention_mask` | bool | `[batch, 1, sequence, sequence]` |
| **in** `position_ids` | int64 | `[batch, sequence]` |
| **out** `logits` | float32 | `[batch, 8, sequence, 1025]` |

This matches how `OmniVoice._generate_iterative` invokes `self.forward()`—the generation loop builds a `[2B, 1, S, S]` 4-D bool block-diagonal mask (first `B` rows = cond, second `B` = uncond for CFG), so this ONNX can drop straight into generation.

### 4.2 `audio_tokenizer_encoder/model.onnx`

| name | dtype | shape |
|------|-------|-------|
| **in** `audio` | float32 | `[batch, 1, num_samples]`, 24 kHz mono, length multiple of `hop_length=960` |
| **out** `audio_codes` | int64 | `[batch, 8, num_frames]`, `num_frames = num_samples / 960` |

### 4.3 `audio_tokenizer_decoder/model.onnx`

| name | dtype | shape |
|------|-------|-------|
| **in** `audio_codes` | int64 | `[batch, 8, num_frames]` |
| **out** `audio` | float32 | `[batch, 1, num_samples]`, `num_samples = num_frames * 960` |

The eight encoder codebooks align 1:1 with LM `num_audio_codebook=8`—feed-through compatible.

---

## 5. Numerical verification

| Target | FP32 thresholds | int8hq thresholds | Notes |
|--------|-----------------|-------------------|-------|
| LM logits | `max|Δ| ≤ 5e-3`, `cos ≥ 0.999`, argmax ≥ 99.9 % | `max|Δ| ≤ 5`, `cos ≥ 0.9999`, argmax ≥ 80 % | All thresholds in `verify_lm.py::TOLERANCES` |
| Audio codes (encoder) | exact-match ≥ 99 % | — | Integer outputs are quantization-sensitive |
| Decoded waveform | `max|Δ| ≤ 5e-3`, `cos ≥ 0.999` | — | — |

Scripts exit non-zero on failure.

---

## 6. Graph optimization

Default order in each exporter:

1. `torch.onnx.export(..., do_constant_folding=True)`.
2. **onnxsim**: constant folding, identity removal, shape inference, dead-node pruning.
3. **onnxruntime.transformers.optimizer** (LM only): `gpt2` preset merges RMSNorm / SkipLayerNorm / RotaryEmbedding.

LM node counts:

| Stage | Nodes | Highlights |
|------|-------|------------|
| Raw export | 7092 | `Constant` 2380, `Pow/ReduceMean/Sqrt` ×113 each (RMSNorm fully expanded) |
| onnxsim + ORT optimizer | **3176** (−55 %) | `Constant` → 0, RMSNorm fused to 57 × `SimplifiedLayerNormalization` |

Pass `--no-optimize` to skip steps 2 and 3.

---

## 7. Quantization options

### 7.0 Overview

| variant | LM size | LM cos | argmax | Listening | Command |
|---------|---------|--------|--------|-----------|---------|
| FP32 | 2.28 GB | 1.0 | 100 % | — | (baseline) |
| INT8 dynamic | 587 MB | 0.99987 | 36–42 % | Audible artifacts | `--int8-mode dynamic` |
| INT8 weight-only | 587 MB | 0.99999 | 71–75 % | Close to FP32, occasional metallic timbre | `--int8-mode weight_only` (default) |
| **★ INT8-HQ** | **611 MB** | **0.99999+** | **80–91 %** | **Indistinguishable from FP32** | `--int8-mode weight_only_hq` |
| INT4 (RTN) | 586 MB | 0.9990 | ~5 % | Clearly degraded | `--methods int4 --int4-algo rtn` |
| INT4 (HQQ) | 586 MB | 0.9991 | ~6 % | Still degraded | `--methods int4 --int4-algo hqq` (default) |
| FP16 | 1.14 GB | 0.99995 | 85 % | Matches FP32 | `--methods fp16` |

> AT encoder/decoder INT8 builds (`audio_tokenizer_*_int8/`) are reused by LM variants `int8`, `int8hq`, `int4`—tokenizer quantization matters far less than LM, so HQ splits aren’t worth it.

### 7.1 Weight-only INT8 (W8A32)

Quantize every 2-D FP32 weight ≥ 1 MB with **per-row symmetric INT8**, insert `DequantizeLinear` before each weight, activations stay FP32 (W8A32).

```bash
python quantize.py --targets lm,at_encoder,at_decoder --methods int8
```

Numerics:

- 198 LM weights quantized (includes `embed_tokens`, `audio_embeddings`, all attention/MLP MatMuls)
- max|Δ| 0.87–3.17, cos 0.99999+, argmax agreement 71–75 %
- AT encoder/decoder treated similarly (some conv tensors stay FP32 due to rank ≥ 3—still ~60 % size savings)

### 7.2 ★ INT8-HQ (recommended for production)

#### Motivation

Standard INT8 weight-only already preserves >99.9 % directional agreement (cos > 0.99999), yet **argmax agreement is only 71–75 %**—why?

Diagnosis narrows noise to the **LM audio output head**:

- A `[1024, 8200] = 8×1025` MatMul that directly feeds logits to the sampler
- Top logits often differ by < 0.05 across 1025 audio tokens—per-row INT8 noise easily flips argmax
- The other ~197 quantized layers get “averaged out” through later nonlinearities—they don’t map 1:1 to logits

**INT8-HQ = baseline INT8 + selected MatMul nodes kept FP32**. Default protection is `audio_heads` only (+24 MB: 611 vs 587), lifting argmax from ~75 % to ~80–91 % with perceptual parity to FP32.

#### How to export

```bash
# Default: protect audio_heads
python quantize.py --targets lm --methods int8 --int8-mode weight_only_hq

# Protect more ops (comma-separated name substrings)
python quantize.py --targets lm --methods int8 --int8-mode weight_only_hq \
                   --int8hq-exclude "audio_heads,o_proj,down_proj"

# Disable exclusions entirely (writes into `_hq` dir but behaves like plain weight_only)
python quantize.py --targets lm --methods int8 --int8-mode weight_only_hq \
                   --int8hq-exclude ""
```

Implementation pointers (`quantize.py`):

| Function | Role |
|----------|------|
| `_init_names_for_node_patterns(model, patterns, op_types=("MatMul","Gemm"))` | Scan graph for MatMul/Gemm nodes whose names match any substring; return initializer names they consume |
| `quantize_initializers_int8(model, exclude_init_names=...)` | Extend W8A32 path with exclusions—skipped weights remain FP32 |
| `quantize_int8_weight_only(..., exclude_node_patterns=...)` | High-level helper resolving patterns→initializers then calling above |
| `--int8-mode weight_only_hq` | Runs dedicated `JOBS_INT8HQ` (LM-only, emits `omnivoice_lm_int8_hq/`) |

> AT checkpoints are **not** re-quantified in hq mode—they reuse existing `audio_tokenizer_*_int8/` artifacts (`_common.py::variant_paths("int8hq")`).

#### Test plan

Three layers of validation (increasing importance):

**A. Offline numerics (~5 s)**

```bash
python verify_lm.py --variant int8hq
```

Runs four `(batch, seq)` shapes vs PyTorch FP32:

```
  [logits] shape=(1, 8,  32, 1025)  max|Δ|=6.6e-01  cos=0.999999   argmax 80.9 %
  [logits] shape=(1, 8,  64, 1025)  max|Δ|=1.1e+00  cos=0.999999   argmax 91.0 %
  [logits] shape=(1, 8, 128, 1025)  max|Δ|=7.7e-01  cos=1.000000   argmax 87.3 %
  [logits] shape=(2, 8,  96, 1025)  max|Δ|=3.1e+00  cos=1.000003   argmax 87.2 %
ALL CASES PASSED ✓
```

Thresholds (`verify_lm.py::TOLERANCES["int8hq"]`): max|Δ| ≤ 5, cos ≥ 0.9999, argmax ≥ 0.80.

**B. Graph self-check**

```bash
python -c "
import onnx
m = onnx.load('output/omnivoice_lm_int8_hq/model.onnx', load_external_data=False)
n_dq = sum(1 for n in m.graph.node if n.op_type == 'DequantizeLinear')
n_int8 = sum(1 for i in m.graph.initializer if i.data_type == 3)
fp32_big = [(i.name, list(i.dims)) for i in m.graph.initializer
            if i.data_type == 1 and len(i.dims) == 2 and i.dims[0]*i.dims[1]*4 >= 1024*1024]
print(f'DequantizeLinear nodes: {n_dq}')
print(f'INT8 initializers: {n_int8}')
print(f'FP32 2-D weights >= 1MB still in graph ({len(fp32_big)}):')
for nm, dims in fp32_big:
    print(f'  {nm}: {dims} ({dims[0]*dims[1]*4/1024/1024:.1f} MB)')
"
```

Expected output (default `audio_heads` exclusion):

```
DequantizeLinear nodes: 198
INT8 initializers: 198
FP32 2-D weights >= 1MB still in graph (1):
  onnx::MatMul_9011: [1024, 8200] (32.0 MB)   ← audio output head
```

**C. End-to-end listening (~100 s)**

```bash
python infer_onnx.py --variant int8hq
```

Writes three WAVs under `output/inference_demo/int8hq/`:

| File | Mode | Text | Highlights |
|------|------|------|--------------|
| `demo_auto.wav` | auto | Long English sentence | No reference tone |
| `demo_voice_clone.wav` | voice clone | Short Mandarin | Uses `assert/andelie.wav` (Russian female reference) |
| `demo_voice_design.wav` | voice design | Short English | `instruct = "male, british accent, low pitch, middle-aged"` |

**How to compare**: Listen to the same demo across `int8/`, `int8hq/`, `fp32/`, focusing on vowel hiss/metallic sheen (classic INT8 issue), HF crispness, phrase endings, and Mandarin tone/rhythm during clone runs.

Expectation: `int8hq` ≈ indistinguishable from `fp32`; plain `int8` occasionally shows metallic coloration and faint clicks.

#### Rollback

```bash
rm -rf output/omnivoice_lm_int8_hq output/inference_demo/int8hq
```

Leaves other variants untouched.

### 7.3 INT8 dynamic (not recommended for LM)

```bash
python quantize.py --int8-mode dynamic
```

- ORT `quantize_dynamic` quantizes weights **and** activations
- Pros: roughly **2×** faster CPU path (INT8 GEMM fusion)
- Cons: activation outliers in Qwen-style stacks blow up quantization error → clearly worse listening (argmax 36–42 % only)
- Fine for AT encoder/decoder (more uniform CNN activations), **avoid for LM**.

### 7.4 INT4 group-wise (experimental)

```bash
# HQQ (default—better quality)
python quantize.py --methods int4 --int4-algo hqq --int4-block-size 16

# RTN (faster, weaker)
python quantize.py --methods int4 --int4-algo rtn --int4-block-size 16
```

- Uses `onnxruntime.quantization.matmul_4bits_quantizer.MatMul4BitsQuantizer` → emits `MatMulNBits`
- Defaults: `block_size=16`, HQQ asymmetric, `audio_heads` excluded (same head protection as int8hq)
- ~586 MB but argmax only ~5–6 % listening quality tanks—**not production-ready**, kept for research on API / block sizes / algorithms.

### 7.5 FP16

```bash
python quantize.py --methods fp16
```

- `onnxconverter_common.float16.convert_float_to_float16` with `keep_io_types=True`, `op_block_list=[]`
- LM ~1.14 GB (2× shrink vs FP32)
- Matches PyTorch FP32 numerically almost everywhere
- **Slowest option on CPUs**: desktop x86 lacks native FP16 SIMD—each FP16 kernel converts to FP32 internally, dominates runtime
- Use only with GPU `--ort-provider CUDAExecutionProvider`.

---

## 8. End-to-end inference

`infer_onnx.py` preserves the PyTorch `OmniVoice.generate(...)` scaffolding (tokenization, text prep, diffusion sampling loop, CFG, audio post)—and monkey-patches three NN entry points:

| PyTorch hook | ONNX replacement |
|--------------|------------------|
| `model.forward(...) → logits` | `omnivoice_lm[_int8/_int8_hq/_int4/_fp16]/model.onnx` |
| `model.audio_tokenizer.encode(audio) → codes` | `audio_tokenizer_encoder[_int8]/model.onnx` |
| `model.audio_tokenizer.decode(audio_codes) → wave` | `audio_tokenizer_decoder[_int8]/model.onnx` |

`_common.py::variant_paths(variant)` resolves filenames.

```bash
# All three demos → output/inference_demo/<variant>/
python infer_onnx.py --variant int8hq

# Single demo only
python infer_onnx.py --variant int8hq --only demo_voice_clone

# Diffusion steps (default 32; fewer = faster but slightly weaker)
python infer_onnx.py --variant int8hq --num-step 16
```

**`instruct` contract**: Voice-design prompts must stick to enumerated vocabulary supported by the model (`omnivoice/models/omnivoice.py::_resolve_instruct`). Examples:

- English: `american accent / british accent / indian accent / male / female / low pitch / high pitch / whisper / young adult / middle-aged / elderly`
- Chinese: tokens like `男 / 女 / 老年 / 中年 / 高音调 / 低音调 / 河南话`, etc.

---

## 9. Performance / RTF / multithreading

### 9.1 RTF definition

`RTF = wall_time / audio_duration` (lower better); `< 1.0` means faster than realtime.

### 9.2 LM-only benchmark (`benchmark_lm.py`)

LM dominates ~95 % of sampling time—one ONNX forward each step—so LM micro-benches predict overall RTF.

```bash
python benchmark_lm.py --variant int8hq --threads 1,4,8,12,16,24 --shapes 1x256,1x512,1x1024
```

**i9-14900KF** (8 P-cores + HT + 16 E-cores)—LM forward at 1024 tokens:

| intra_op_num_threads | Median latency | vs 1 thread | Diminishing returns |
|---|---:|---:|---:|
| 1 | 7926 ms | 1.0× | — |
| 4 | 2457 ms | 3.2× | Sweet spot ramp |
| **8** | **1633 ms** | **4.9×** | **★ Best bang-for-buck** |
| 12 | 1563 ms | 5.1× | +4 % |
| 16 | 1512 ms | 5.2× | +3 % |
| 24 | 1506 ms | 5.3× | ~0 % |

Takeaways:

- **8 threads** lines up well with eight physical P-cores.
- Scaling 8→16 gains ~7 % from hyper-threading—marginal thereafter.
- 16→24 saturates—extra threads spill onto slower E-cores (4.4 GHz vs P-core 6.0 GHz & ~30 % lower IPC), hurting BLAS cadence.

### 9.3 Recommended deployment knobs

`infer_onnx.py::_make_session` defaults `intra_op_num_threads = 0` (ORT auto, often all physical cores = 24 on this SKU). Pin manually if desired:

```python
so.intra_op_num_threads = 8     # i9 / i7 K-class hybrids
# Or affinity to P cores: taskset -c 0-15 python infer_onnx.py ...
```

CLI affinity example:

```bash
taskset -c 0-15 python infer_onnx.py --variant int8hq
```

> Tune per SKU:
> - Intel 12/13/14 K desktop: aim for physical P-core count (6–8 typical)
> - AMD Ryzen / Threadripper: physical core counts (no heterogeneous clusters)
> - Xeon / EPYC: watch NUMA pinning

### 9.4 End-to-end RTF (i9-14900KF defaults, `--num-step 32`)

| variant | demo_auto (9.2 s speech) | demo_voice_clone (7.8 s) | demo_voice_design (6.4 s) |
|---------|---------------------------:|--------------------------:|--------------------------:|
| FP32 | 1.34 | 3.84 | 1.45 |
| INT8 weight-only | 2.92 | 6.74 | 3.63 |
| **INT8-HQ** | **2.79** | **6.69** | **3.42** |
| INT8 dynamic | 1.19 | 3.36 | 1.32 |
| FP16 (CPU) | 5.43 | 14.82 | 6.68 |

> Notes:
> - `voice_clone` RTF is higher mostly due to one-time reference-audio encode (~5–10 s) plus doubled Mandarin token counts vs English demos.
> - INT8 weight-only / int8hq are ~**2×** slower than INT8 dynamic—ORT lacks `DequantizeLinear+MatMul` fusion, forcing per-layer dequantize back to FP32. That’s deliberate quality-vs-speed trade.
> - First runs add ~5–8 s warmup (graph optimization + PT load).

### 9.5 GPU roadmap (estimated)

Current INT8/INT8-HQ ONNX targets CPUs (`DequantizeLinear` rarely fuses into INT8 GEMM on GPUs). Faster GPU paths:

| Path | Work | Estimated RTF (RTX 4090) |
|------|------|-------------------------:|
| ORT-CUDA + FP16 | `pip install onnxruntime-gpu`, `--variant fp16 --ort-provider CUDAExecutionProvider --device cuda:0` | **0.05–0.15** |
| TensorRT EP | Convert FP16 ONNX → TRT + INT8 calib | 0.03–0.08 |
| vLLM / TensorRT-LLM | Rebuild LM loop inside specialized stacks | 0.02–0.05 |

> RTX 4090 ORT CUDA FP16 is roughly **20–50×** faster than desktop CPU INT8-HQ here—often 7–20× faster than realtime.

---

## 10. Troubleshooting and rollback

### 10.1 Remove a variant

```bash
rm -rf output/omnivoice_lm_int8_hq output/inference_demo/int8hq
rm -rf output/omnivoice_lm_int4    output/inference_demo/int4
rm -rf output/omnivoice_lm_int8    output/inference_demo/int8
rm -rf output/omnivoice_lm_fp16    output/inference_demo/fp16
```

### 10.2 Full reset

```bash
rm -rf output/
```

Deletes every ONNX artifact + regression WAV dumps.

### 10.3 Common errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| `IndexError: tuple index out of range` in `transformers/masking_utils.py` | transformers wraps Python ints as 0-D tensors during tracing | Auto-patched via `_common.py::patch_transformers_sdpa_mask_for_tracing` |
| `Could not find an implementation for ConvInteger` | INT8 dynamic touched Conv kernels ORT CPU lacks | `quantize_int8_dynamic` limits `op_types_to_quantize=["MatMul","Gemm","Attention"]` |
| `Unable to find data type for weight_name='...'` | `quantize_dynamic` dtype inference flake | Added `extra_options["DefaultTensorType"] = onnx.TensorProto.FLOAT` |
| `should be stored in .../model.onnx.data, but it doesn't exist` | External data sibling missing | Load-order fix `_normalize_sidecar` |
| Obvious perceptual regressions during inference | Default `--variant int8` | Switch to `--variant int8hq` |
| `--variant fp16` crawling on CPU | x86 lacks FP16 SIMD—FP16 is GPU-first | Prefer `int8hq` locally or CUDA EP |

# OmniVoice ONNX 导出 / 量化 / 推理

[English README](README_EN.md)

把 [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)（PyTorch）转成 ONNX，覆盖 **导出 → 量化（INT8 / INT8-HQ / INT4 / FP16）→ 数值校验 → 端到端推理 → 性能基准** 全流程。

> **当前生产推荐**：`int8hq`（LM 587→611 MB，audio output head 保留 FP32），数值 cos > 0.9999、argmax 一致率 > 80 %，听感几乎与 FP32 无异。详见 [§7.2](#72--int8-hq推荐生产)。

## 前置依赖

本仓库只包含**导出 / 量化 / 推理脚本**，不含模型权重或测试音频。运行前需要：

1. **OmniVoice PyTorch 模型** —— 从 [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) 克隆主仓库并下载预训练权重，得到一个 `OmniVoice/` 目录（含 `model.safetensors` + `audio_tokenizer/`）。
2. **`tts` conda 环境** —— 按 OmniVoice 主仓库的 README 安装依赖（`torch`, `transformers>=5.5`, `onnx`, `onnxruntime`, `onnxsim`, `onnxconverter_common`, `torchaudio`）。
3. **参考音频**（可选，仅 `voice_clone` demo 需要）—— 任意一段 24 kHz mono wav，本文档默认引用 `assert/andelie.wav`。

推荐布局：

```
your-workspace/
├── OmniVoice/                # k2-fsa/OmniVoice 主仓库（含模型权重）
│   ├── OmniVoice/             # 模型权重目录（PT_MODEL_DIR）
│   ├── omnivoice/             # python 包
│   ├── assert/andelie.wav     # voice_clone 参考音频
│   └── onnx_export/           # ← 把本仓库克隆到这里
│       ├── _common.py
│       ├── export_lm.py
│       └── ...
```

`_common.py` 顶部对路径有约定：`PROJECT_ROOT = Path(__file__).parent.parent`、`PT_MODEL_DIR = PROJECT_ROOT / "OmniVoice"`。如果你的布局不同，改 `_common.py` 里的两行常量即可。

## 目录

1. [概览](#1-概览)
2. [目录布局](#2-目录布局)
3. [一键流程](#3-一键流程)
4. [ONNX IO 约定](#4-onnx-io-约定)
5. [数值校验](#5-数值校验)
6. [图优化](#6-图优化)
7. [量化方案](#7-量化方案)
8. [端到端推理](#8-端到端推理)
9. [性能 / RTF / 多线程](#9-性能--rtf--多线程)
10. [故障排查与回滚](#10-故障排查与回滚)

---

## 1. 概览

OmniVoice 是一个文本→音频的 TTS 模型，由两部分组成：

- **LM**：Qwen3 类 600 M 参数语言模型，输出 `[B, 8 codebooks, S, 1025]` 的 audio token logits
- **Audio Tokenizer**：HiggsAudioV2，分 encoder（音频→8 codebook codes）和 decoder（codes→音频）

本目录把这三部分各自导出为一个独立 ONNX 文件，并提供 5 种精度变体：

| variant | LM 大小 | 推荐场景 |
|---------|---------|----------|
| `fp32` | 2.28 GB | 基准 / 对照 |
| `int8` | 587 MB | 体积优先，可接受轻微音质损失 |
| **`int8hq`** | **611 MB** | **★ 生产推荐**（音质几乎无损，体积只多 24 MB） |
| `int4` | 586 MB | 实验：体积/速度更紧，但听感明显下降 |
| `fp16` | 1.14 GB | GPU 部署专用（CPU 上反而最慢） |

---

## 2. 目录布局

仓库源码（这里只是源码，运行时会在 `output/` 下生成所有产物，`output/` 已被 `.gitignore` 排除）：

```
.
├── README.md                  # 你正在读的这份文档
├── .gitignore
├── _common.py                 # 共享路径 / 设备 / 加载 / 误差打印 / variant_paths
├── export_lm.py               # 导出 LM
├── export_audio_tokenizer.py  # 导出 audio tokenizer 的 encoder + decoder
├── quantize.py                # ★ INT8 / INT8-HQ / INT4 / FP16 一站式量化
├── verify_lm.py               # ONNX vs PyTorch 数值校验（LM，按 variant）
├── verify_audio_tokenizer.py  # ONNX vs PyTorch 数值校验（encoder / decoder）
├── test_pipeline.py           # 用 assert/andelie.wav 跑端到端 encode→decode
├── infer_onnx.py              # ★ 端到端 TTS 推理：LM + tokenizer 全 ONNX
└── benchmark_lm.py            # ★ LM 单步延迟微基准（线程扫描）
```

运行后会自动创建（**不会被提交**）：

```
output/
├── omnivoice_lm/                  ← FP32 LM（基准）
├── omnivoice_lm_int8/             ← INT8 weight-only（W8A32）
├── omnivoice_lm_int8_hq/          ← ★ INT8-HQ：W8A32 + audio_heads 保 FP32
├── omnivoice_lm_int4/             ← INT4 group-wise（MatMulNBits, RTN/HQQ）
├── omnivoice_lm_fp16/             ← FP16 LM（GPU 路径）
├── audio_tokenizer_encoder/        ← FP32 encoder
├── audio_tokenizer_encoder_int8/   ← INT8 encoder（int8 / int8hq / int4 共用）
├── audio_tokenizer_decoder/        ← FP32 decoder
├── audio_tokenizer_decoder_int8/   ← INT8 decoder（int8 / int8hq / int4 共用）
├── test_outputs/                   ← test_pipeline 产物
└── inference_demo/
    ├── fp32/   demo_auto.wav, demo_voice_clone.wav, demo_voice_design.wav
    ├── int8/   ...
    ├── int8hq/ ...     ← ★ 听感对比基准
    ├── int4/   ...
    └── fp16/   ...
```

> 所有量化结果都在独立子目录，互相不会覆盖。删除某个子目录就完全回滚那个 variant。

---

## 3. 一键流程

```bash
conda activate tts
cd path/to/OmniVoice/onnx_export   # 见 §前置依赖 的推荐布局

# 1. 导出 FP32（一次性，约 5-10 分钟）
python export_audio_tokenizer.py
python export_lm.py

# 2. 数值校验（FP32）
python verify_audio_tokenizer.py
python verify_lm.py --variant fp32

# 3. 端到端 pipeline 测试（用 assert/andelie.wav 真实音频）
python test_pipeline.py

# 4. 量化到推荐的 int8hq（LM only；AT 模型用标准 int8 即可）
python quantize.py --targets at_encoder,at_decoder --methods int8 --no-smoke
python quantize.py --targets lm --methods int8 --int8-mode weight_only_hq --no-smoke

# 5. int8hq 数值校验
python verify_lm.py --variant int8hq

# 6. ★ 端到端 TTS 推理（生成 3 个示例听感测试）
python infer_onnx.py --variant int8hq
```

---

## 4. ONNX IO 约定

### 4.1 `omnivoice_lm/model.onnx`（所有 variant 共享接口）

| name | dtype | shape |
|------|-------|-------|
| **in** `input_ids` | int64 | `[batch, 8, sequence]` |
| **in** `audio_mask` | bool | `[batch, sequence]` |
| **in** `attention_mask` | bool | `[batch, 1, sequence, sequence]` |
| **in** `position_ids` | int64 | `[batch, sequence]` |
| **out** `logits` | float32 | `[batch, 8, sequence, 1025]` |

接口与 `OmniVoice._generate_iterative` 内部对 `self.forward()` 的调用方式一一对应——generate 循环里构造的就是一个 `[2B, 1, S, S]` 的 4-D bool 块对角 mask（前 B 个是 cond，后 B 个是 uncond，做 CFG），所以这个 ONNX 是一个真正可被 generate 直接驱动的 drop-in 替换。

### 4.2 `audio_tokenizer_encoder/model.onnx`

| name | dtype | shape |
|------|-------|-------|
| **in** `audio` | float32 | `[batch, 1, num_samples]`，24 kHz mono，长度需为 `hop_length=960` 的整数倍 |
| **out** `audio_codes` | int64 | `[batch, 8, num_frames]`，`num_frames = num_samples / 960` |

### 4.3 `audio_tokenizer_decoder/model.onnx`

| name | dtype | shape |
|------|-------|-------|
| **in** `audio_codes` | int64 | `[batch, 8, num_frames]` |
| **out** `audio` | float32 | `[batch, 1, num_samples]`，`num_samples = num_frames * 960` |

> Encoder 输出的 8 个 codebook 与 LM 的 `num_audio_codebook=8` 一一对应，可直接互通。

---

## 5. 数值校验

| 校验对象 | FP32 阈值 | int8hq 阈值 | 说明 |
|---------|----------|-------------|------|
| LM logits | `max\|Δ\| ≤ 5e-3`，`cos ≥ 0.999`，argmax ≥ 99.9 % | `max\|Δ\| ≤ 5`，`cos ≥ 0.9999`，argmax ≥ 80 % | 全部 variant 的阈值在 `verify_lm.py::TOLERANCES` |
| 音频 codes（encoder） | exact-match ≥ 99 % | — | int 量化敏感 |
| 解码后音频（decoder） | `max\|Δ\| ≤ 5e-3`，`cos ≥ 0.999` | — | — |

校验脚本退出码非 0 即失败。

---

## 6. 图优化

每个导出脚本默认顺序：

1. `torch.onnx.export(..., do_constant_folding=True)`：torch 自带的常量折叠。
2. **onnxsim**：常量折叠、Identity 消除、shape 推导、dead-node 剪枝。
3. **onnxruntime.transformers.optimizer**（仅 LM）：`gpt2` preset 做 RMSNorm / SkipLayerNorm / RotaryEmbedding 融合。

LM 优化前后节点数：

| 阶段 | 节点数 | 关键变化 |
|------|--------|----------|
| 原始导出 | 7092 | `Constant` 2380, `Pow/ReduceMean/Sqrt` 各 113（RMSNorm 全展开） |
| onnxsim + ORT optimizer | **3176** (-55 %) | `Constant` 0，RMSNorm 融合成 57 个 `SimplifiedLayerNormalization` |

加 `--no-optimize` 跳过 2、3。

---

## 7. 量化方案

### 7.0 总览

| variant | LM 大小 | LM cos | argmax | 听感 | 命令 |
|---------|---------|--------|--------|------|------|
| FP32 | 2.28 GB | 1.0 | 100 % | — | （基准） |
| INT8 dynamic | 587 MB | 0.99987 | 36–42 % | 可听到失真 | `--int8-mode dynamic` |
| INT8 weight-only | 587 MB | 0.99999 | 71–75 % | 接近 FP32，偶有微小金属感 | `--int8-mode weight_only`（默认） |
| **★ INT8-HQ** | **611 MB** | **0.99999+** | **80–91 %** | **听感与 FP32 无异** | `--int8-mode weight_only_hq` |
| INT4 (RTN) | 586 MB | 0.9990 | ~5 % | 明显失真 | `--methods int4 --int4-algo rtn` |
| INT4 (HQQ) | 586 MB | 0.9991 | ~6 % | 仍有失真 | `--methods int4 --int4-algo hqq`（默认） |
| FP16 | 1.14 GB | 0.99995 | 85 % | 与 FP32 无异 | `--methods fp16` |

> AT-encoder/decoder 的 INT8 build（`audio_tokenizer_*_int8/`）被 `int8`、`int8hq`、`int4` 三种 LM variant 共用——audio tokenizer 量化对听感影响远小于 LM，没必要再分 hq。

### 7.1 weight-only INT8（W8A32）

把所有 ≥ 1 MB 的二维 FP32 权重做 **per-row 对称 INT8** 量化，并在每个权重前插入一个 `DequantizeLinear`，激活始终保持 FP32（W8A32 = 8-bit weight + 32-bit activation）。

```bash
python quantize.py --targets lm,at_encoder,at_decoder --methods int8
```

数值表现：
- 198 个 LM 权重被量化（含 `embed_tokens`、`audio_embeddings`、所有 attention/MLP MatMul）
- max\|Δ\| 0.87–3.17，cos 0.99999+，argmax 一致率 71–75 %
- AT encoder/decoder 同样处理（部分 conv 因为是 ≥ 3 维所以保留 FP32，体积仍降到约 60 %）

### 7.2 ★ INT8-HQ（推荐生产）

#### 动机

标准 INT8 weight-only 已经把 95 % 的 logit 方向保留了（cos > 0.99999），但 **argmax 一致率仅 71–75 %**——为什么？

经诊断，问题集中在 **LM 的 audio output head**：

- 它是一个 `[1024, 8200] = 8 codebooks × 1025 vocab` 的 MatMul，直接产生 sampler 用的 logits
- 1025 个 audio token 的 top-k logits 间距常 < 0.05，per-row INT8 量化噪声在量级上就足以翻转 argmax
- 而其他 197 个层的量化噪声会被后续层的非线性"平均掉"，不会一对一传到最终 logits

**INT8-HQ = 标准 INT8 + 选定的若干 MatMul 节点保留 FP32**。默认只保护 `audio_heads`，体积只增加 24 MB（611 vs 587），换来 argmax 一致率从 75 % 直接跳到 80–91 %、听感与 FP32 无差。

#### 导出方法

```bash
# 默认：保护 audio_heads
python quantize.py --targets lm --methods int8 --int8-mode weight_only_hq

# 自定义保护更多层（用逗号分隔节点名子串）
python quantize.py --targets lm --methods int8 --int8-mode weight_only_hq \
                   --int8hq-exclude "audio_heads,o_proj,down_proj"

# 完全禁用排除（等价于 --int8-mode weight_only，但写到 _hq 目录）
python quantize.py --targets lm --methods int8 --int8-mode weight_only_hq \
                   --int8hq-exclude ""
```

实现要点（`quantize.py`）：

| 函数 | 作用 |
|------|------|
| `_init_names_for_node_patterns(model, patterns, op_types=("MatMul","Gemm"))` | 扫描计算图，找到节点名包含 `patterns` 任一子串的 MatMul/Gemm，返回它们使用的 initializer 名字集合 |
| `quantize_initializers_int8(model, exclude_init_names=...)` | 在原 W8A32 量化函数上加 `exclude_init_names` 参数，被排除的权重保持 FP32 |
| `quantize_int8_weight_only(..., exclude_node_patterns=...)` | 高层入口：传入节点名模式，内部解析为 initializer 名字后调用上面两个函数 |
| `--int8-mode weight_only_hq` | 在 main 中走专属 jobs 列表 `JOBS_INT8HQ`（仅 LM，输出到 `omnivoice_lm_int8_hq/`），不会污染标准 `int8` 目录 |

> AT 模型在 hq 模式下不会重新量化——它们直接复用 `audio_tokenizer_*_int8/` 目录。这是 `_common.py::variant_paths("int8hq")` 决定的。

#### 测试方案

完整测试分三层，按重要性递增：

**A. 离线数值校验**（5 秒）

```bash
python verify_lm.py --variant int8hq
```

会跑 4 个 (batch, seq) 配置，分别对比 PyTorch FP32：

```
  [logits] shape=(1, 8,  32, 1025)  max|Δ|=6.6e-01  cos=0.999999   argmax 80.9 %
  [logits] shape=(1, 8,  64, 1025)  max|Δ|=1.1e+00  cos=0.999999   argmax 91.0 %
  [logits] shape=(1, 8, 128, 1025)  max|Δ|=7.7e-01  cos=1.000000   argmax 87.3 %
  [logits] shape=(2, 8,  96, 1025)  max|Δ|=3.1e+00  cos=1.000003   argmax 87.2 %
ALL CASES PASSED ✓
```

阈值在 `verify_lm.py::TOLERANCES["int8hq"]`：`max|Δ| ≤ 5`，`cos ≥ 0.9999`，`argmax ≥ 0.80`。

**B. 计算图自检**（确认排除规则真的生效）

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

期望输出（默认排除 `audio_heads`）：

```
DequantizeLinear nodes: 198
INT8 initializers: 198
FP32 2-D weights >= 1MB still in graph (1):
  onnx::MatMul_9011: [1024, 8200] (32.0 MB)   ← audio output head
```

**C. 端到端听感**（最终判据，每次约 100 秒）

```bash
python infer_onnx.py --variant int8hq
```

生成三个 wav 到 `output/inference_demo/int8hq/`：

| 文件 | 模式 | 文本 | 关键 |
|------|------|------|------|
| `demo_auto.wav` | auto | 英文长句 | 无参考，模型自挑音色 |
| `demo_voice_clone.wav` | voice clone | 中文短句 | 用 `assert/andelie.wav`（俄语女声）做参考 |
| `demo_voice_design.wav` | voice design | 英文短句 | `instruct = "male, british accent, low pitch, middle-aged"` |

**对比方法**：把同一个 demo 的 `int8/`、`int8hq/`、`fp32/` 三个版本依次对照听，重点听：
- 元音的金属/嘶嘶感（INT8 主要劣化点）
- 高频清脆度（INT8 经常变钝）
- 句尾收音的自然度
- 中文克隆下的声调/韵律

预期：`int8hq` 与 `fp32` 听不出区别；`int8`（标准）会偶有金属感和轻微 click。

#### 回滚

```bash
rm -rf output/omnivoice_lm_int8_hq output/inference_demo/int8hq
```

不会影响其他 variant。

### 7.3 INT8 dynamic（不推荐 LM 用）

```bash
python quantize.py --int8-mode dynamic
```

- 用 ORT 的 `quantize_dynamic`，weight + activation 都 INT8
- 优点：CPU 上比 weight_only 快 ~2×（因为有 INT8 GEMM 融合算子）
- 缺点：Qwen3 类模型 activation 有大量 outlier，per-tensor absmax 求 scale 后量化误差被放大，听感明显劣化（argmax 一致率仅 36–42 %）
- 仅推荐 AT-encoder/decoder（卷积 activation 分布更均匀），LM 千万别用

### 7.4 INT4 group-wise（实验）

```bash
# HQQ (默认，质量较好)
python quantize.py --methods int4 --int4-algo hqq --int4-block-size 16

# RTN (较快，质量稍差)
python quantize.py --methods int4 --int4-algo rtn --int4-block-size 16
```

- 用 `onnxruntime.quantization.matmul_4bits_quantizer.MatMul4BitsQuantizer`，输出 `MatMulNBits` 自定义算子
- 默认 `block_size=16`、HQQ asymmetric、`audio_heads` 排除（与 int8hq 同样保护输出头）
- 体积 586 MB，但 argmax 仅 ~5–6 %，听感劣化明显——**不建议生产用**
- 留作研究：`MatMul4BitsQuantizer` API 与不同 block_size / algo 的探索

### 7.5 FP16

```bash
python quantize.py --methods fp16
```

- `onnxconverter_common.float16.convert_float_to_float16`，`keep_io_types=True`、`op_block_list=[]`
- LM 体积 1.14 GB（2× 缩减）
- 数值精度高（与 PyTorch FP32 几乎一致）
- **CPU 上反而最慢**：x86 没有原生 FP16 SIMD，每个 fp16 算子运行时 cast 到 fp32 再算，cast 本身是大头
- 只在 GPU (`--ort-provider CUDAExecutionProvider`) 下值得用

---

## 8. 端到端推理

`infer_onnx.py` 的设计：保留 PyTorch 的 `OmniVoice.generate(...)` 框架（它负责 tokenize / 文本预处理 / 扩散采样循环 / CFG / 音频后处理），monkey-patch 三个神经网络入口：

| PyTorch 调用 | 替换为 |
|---|---|
| `model.forward(input_ids, audio_mask, attention_mask, ...)` → `.logits` | `omnivoice_lm[_int8 / _int8_hq / _int4 / _fp16]/model.onnx` |
| `model.audio_tokenizer.encode(audio)` → `.audio_codes` | `audio_tokenizer_encoder[_int8]/model.onnx` |
| `model.audio_tokenizer.decode(audio_codes)` → `.audio_values` | `audio_tokenizer_decoder[_int8]/model.onnx` |

variant 由 `_common.py::variant_paths(variant)` 解析为具体的三元组路径。

```bash
# 完整 3 demo，输出到 output/inference_demo/<variant>/
python infer_onnx.py --variant int8hq

# 只跑某一个 demo
python infer_onnx.py --variant int8hq --only demo_voice_clone

# 调采样步数（默认 32；越小越快、质量略降）
python infer_onnx.py --variant int8hq --num-step 16
```

**`instruct` 字段约定**：voice_design 模式下的 `instruct` 必须使用模型支持的固定词库（见 `omnivoice/models/omnivoice.py::_resolve_instruct`），不接受自由文本。常见枚举：
- 英文：`american accent / british accent / indian accent / male / female / low pitch / high pitch / whisper / young adult / middle-aged / elderly`
- 中文：`男 / 女 / 老年 / 中年 / 高音调 / 低音调 / 河南话` 等

---

## 9. 性能 / RTF / 多线程

### 9.1 RTF 定义

`RTF = wallclock / audio_duration`，越小越好；`< 1.0` 表示比实时还快。

### 9.2 LM 单步基准（`benchmark_lm.py`）

LM 占整个 generate 流程的 ~95 % 时间（每个 sampling step 调用一次 LM forward），所以单独基准 LM 就能直接预测整体 RTF。

```bash
python benchmark_lm.py --variant int8hq --threads 1,4,8,12,16,24 --shapes 1x256,1x512,1x1024
```

**i9-14900KF（8 P-core 含超线程 + 16 E-core）实测 LM 1024-token forward**：

| intra_op_num_threads | median latency | 相对 1 线程 | 边际收益 |
|---:|---:|---:|---:|
| 1 | 7926 ms | 1.0× | — |
| 4 | 2457 ms | 3.2× | 主收益 |
| **8** | **1633 ms** | **4.9×** | **★ 性价比最优** |
| 12 | 1563 ms | 5.1× | +4 % |
| 16 | 1512 ms | 5.2× | +3 % |
| 24 | 1506 ms | 5.3× | ~0 % |

**结论**：

- **8 线程是性价比最佳点**：恰好填满 8 个物理 P 核
- 8→16 的 ~7 % 提升来自 P 核超线程，再多收益打折
- 16→24 几乎饱和，因为继续加线程会调度到 E 核（4.4 GHz vs P 核 6.0 GHz，IPC 还低 30 %），反而拖累 BLAS 节奏

### 9.3 推荐部署配置

在 `infer_onnx.py::_make_session` 里默认是 `intra_op_num_threads = 0`（让 ORT 自己决定，通常等于物理核数 = 24）。如果想稳定吃到最佳：

```python
so.intra_op_num_threads = 8     # 14900K / 13900K 类机型
# 或者绑核到 P 核：启动时用 taskset -c 0-15 python infer_onnx.py ...
```

或者命令行：

```bash
taskset -c 0-15 python infer_onnx.py --variant int8hq
```

> 不同 CPU 的最佳点：
> - Intel 12/13/14 代消费 K：物理 P-core 数（8、6、8）
> - AMD Ryzen / Threadripper：物理核数（无 P/E 异构）
> - Xeon / EPYC 服务器：物理核数，但 NUMA 跨节点要避免

### 9.4 端到端 RTF（i9-14900KF 默认配置，`--num-step 32`）

| variant | demo_auto (9.2 s) | demo_voice_clone (7.8 s) | demo_voice_design (6.4 s) |
|---------|------------------:|-------------------------:|---------------------------:|
| FP32 | 1.34 | 3.84 | 1.45 |
| INT8 weight-only | 2.92 | 6.74 | 3.63 |
| **INT8-HQ** | **2.79** | **6.69** | **3.42** |
| INT8 dynamic | 1.19 | 3.36 | 1.32 |
| FP16 (CPU) | 5.43 | 14.82 | 6.68 |

> 说明：
> - `voice_clone` RTF 显著高，是因为它额外做了 ref-audio encode（约 5–10 秒一次性开销），以及中文 token 数比英文翻倍
> - INT8 weight-only / int8hq 比 INT8 dynamic 慢 ~2×：ORT 没有 `DequantizeLinear+MatMul` 的融合算子，每个 MatMul 前都要把权重 dequant 回 fp32。但音频质量明显更好——质量与速度的取舍
> - 第一次跑会比稳态多 5–8 秒（ORT 图优化 + PyTorch 加载）

### 9.5 GPU 路径（预估）

当前 INT8/INT8-HQ ONNX 是为 CPU 调优的（`DequantizeLinear` 在 GPU 上反而拖累，因为不会被融合成 INT8 GEMM）。要吃 GPU 性能：

| 路径 | 改动 | 预期 RTF（RTX 4090） |
|------|------|--------:|
| ORT-CUDA + FP16 | `pip install onnxruntime-gpu` + `--variant fp16 --ort-provider CUDAExecutionProvider --device cuda:0` | **0.05 – 0.15** |
| TensorRT EP | 把 fp16 onnx 转 TRT engine + INT8 calibration | 0.03 – 0.08 |
| vLLM / TensorRT-LLM | 换 LLM 专用栈（要重写 generate 循环） | 0.02 – 0.05 |

> 即 4090 + ORT-CUDA + FP16 ≈ 比当前 CPU INT8-HQ 快 **20–50×**，比实时快 7–20×。

---

## 10. 故障排查与回滚

### 10.1 单 variant 回滚

```bash
# 删除某个 LM variant（不影响其他）
rm -rf output/omnivoice_lm_int8_hq output/inference_demo/int8hq
rm -rf output/omnivoice_lm_int4    output/inference_demo/int4
rm -rf output/omnivoice_lm_int8    output/inference_demo/int8
rm -rf output/omnivoice_lm_fp16    output/inference_demo/fp16
```

### 10.2 完全重置

```bash
rm -rf output/                          # 删除所有 ONNX 产物 + 测试 wav
```

### 10.3 常见错误

| 现象 | 原因 | 解决 |
|------|------|------|
| `IndexError: tuple index out of range` 在 `transformers/masking_utils.py` | tracing 阶段 transformers 把 python int 包成 0-D tensor | 已用 `_common.py::patch_transformers_sdpa_mask_for_tracing` 自动 monkey-patch |
| `Could not find an implementation for ConvInteger` | INT8 dynamic 把 Conv 量化了，但 ORT-CPU 没有这个算子 | `quantize_int8_dynamic` 已限定 `op_types_to_quantize=["MatMul","Gemm","Attention"]` |
| `Unable to find data type for weight_name='...'` | quantize_dynamic 推中间张量 dtype 失败 | `extra_options["DefaultTensorType"] = onnx.TensorProto.FLOAT` 已加 |
| `should be stored in .../model.onnx.data, but it doesn't exist` | external data sidecar 找不到 | 加载顺序问题；`_normalize_sidecar` 已修 |
| 推理时音质明显劣化 | 用了 `--variant int8`（标准 dynamic / weight_only） | 切到 `--variant int8hq` |
| `--variant fp16` 在 CPU 极慢 | x86 无 FP16 SIMD，FP16 是 GPU 路径 | 切回 `int8hq` 或上 `--ort-provider CUDAExecutionProvider` |

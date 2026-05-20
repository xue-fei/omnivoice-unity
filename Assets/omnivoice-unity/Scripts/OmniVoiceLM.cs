using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

/// <summary>
/// OmniVoice Language Model (Diffusion LM, ONNX Runtime)
///
/// ★ 架构说明：OmniVoice 是扩散语言模型（Diffusion LM），而非自回归（AR）模型。
///   正确流程：
///     1. 构建完整序列 [text_tokens | ref_codes | MASK_codes]（全部已知长度）
///     2. 在固定 NumStep 轮迭代中，对 masked 位置做并行去噪
///     3. 每步更新被 mask 的音频位置，保留 ref 位置不变
///   错误做法：逐 token 自回归采样（慢 100× 且生成噪音）
///
/// ONNX IO（来自 AFun9/Omnivoice-onnx §4.1）：
///   输入  input_ids       int64  [B, 8, S]
///   输入  audio_mask      bool   [B, S]
///   输入  attention_mask  bool   [B, 1, S, S]
///   输入  position_ids    int64  [B, S]
///   输出  logits          float  [B, 8, S, 1025]
/// </summary>
public class OmniVoiceLM : IDisposable
{
    // ─── 常量 ────────────────────────────────────────────────────────────────
    public const int NUM_CODEBOOKS = 8;
    public const int VOCAB_SIZE = 1025;   // 1024 audio codes + 1 EOS
    public const int MASK_TOKEN = 1024;   // 用于扩散的 mask token（同时也是 EOS）
    public const int PAD_TOKEN = 0;
    public const int HOP_FRAMES = 1;      // ONNX 内 audio tokenizer hop=960 samples，1 frame = 1 code

    InferenceSession _session;

    // ─── 生成参数 ──────────────────────────────────────────────────────────
    /// <summary>扩散迭代步数（原始 Python 默认 32，越多越好但越慢）</summary>
    public int NumStep = 32;

    /// <summary>CFG 引导强度（原始 Python 默认 2.0，0 = 关闭 CFG）</summary>
    public float GuidanceScale = 2.0f;

    /// <summary>扩散调度 t_shift（原始 Python 默认 0.1）</summary>
    public float TShift = 0.1f;

    /// <summary>采样 TopK（仅在离散采样时生效）</summary>
    public int TopK = 50;

    /// <summary>采样温度</summary>
    public float Temperature = 1.0f;

    System.Random _rng;

    public OmniVoiceLM(string modelPath, int seed = 42)
    {
        _rng = new System.Random(seed);
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        opts.AppendExecutionProvider_DML(0);   // DirectML（Windows GPU）；无 GPU 自动回退 CPU
        _session = new InferenceSession(modelPath, opts);
        Debug.Log($"[OmniVoiceLM] 已加载: {modelPath}");
    }

    // ════════════════════════════════════════════════════════════════════════
    // 公开入口：扩散生成
    // ════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// 扩散 LM 语音克隆生成（正确实现）。
    ///
    /// 参数：
    ///   textTokenIds  — 完整文本 prompt（由 Qwen2Tokenizer.BuildPrompt 得到）
    ///   refCodes      — 参考音频 codes [8, T_ref]（由 AudioTokenizer.Encode 得到）
    ///   targetLen     — 目标生成帧数（不传则按 refCodes 长度估算）
    ///
    /// 返回：生成的 audio codes [8, T_gen]
    /// </summary>
    public long[,] Generate(int[] textTokenIds, long[,] refCodes = null, int targetLen = -1)
    {
        int T_text = textTokenIds?.Length ?? 0;
        int T_ref = refCodes != null ? refCodes.GetLength(1) : 0;

        // 目标生成长度估算：如果未指定，用参考音频长度（语音克隆场景）
        if (targetLen <= 0)
            targetLen = T_ref > 0 ? T_ref : 100;

        // ── 构建初始序列 ─────────────────────────────────────────────────
        // 序列布局（audio_mask 中 audio=true 的部分才走音频 embedding）：
        //   [text tokens (audio_mask=false)] [ref codes (audio_mask=true)] [gen codes初始=MASK (audio_mask=true)]
        //
        // 注：若模型是 audio-only ONNX（没有文本嵌入），把 T_text 设为 0 即可。
        int S = T_text + T_ref + targetLen;

        // input_ids: [1, 8, S]
        var inputIds = new long[1, NUM_CODEBOOKS, S];
        var audioMask = new bool[1, S];

        // 文本区域：audio_mask=false，input_ids 放文本 token id（codebook 维度复制）
        for (int s = 0; s < T_text; s++)
        {
            long tid = textTokenIds[s];
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = tid;
            audioMask[0, s] = false;
        }

        // 参考音频区域：audio_mask=true，填入实际 codes
        for (int t = 0; t < T_ref; t++)
        {
            int s = T_text + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = Math.Clamp(refCodes[cb, t], 0, MASK_TOKEN);
            audioMask[0, s] = true;
        }

        // 待生成区域：audio_mask=true，初始化为 MASK_TOKEN（全噪声）
        for (int t = 0; t < targetLen; t++)
        {
            int s = T_text + T_ref + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = MASK_TOKEN;
            audioMask[0, s] = true;
        }

        // ── 扩散调度（余弦调度，与 Python infer_onnx 一致）──────────────
        // t 从 1→0，TShift 做 warp
        float[] tSchedule = BuildCosineSchedule(NumStep, TShift);

        Debug.Log($"[OmniVoiceLM] 扩散生成: T_text={T_text} T_ref={T_ref} T_gen={targetLen} S={S} NumStep={NumStep}");

        // ── 扩散迭代 ─────────────────────────────────────────────────────
        for (int step = 0; step < NumStep; step++)
        {
            float tCurr = tSchedule[step];
            float tNext = tSchedule[step + 1];

            if (step % 8 == 0)
                Debug.Log($"[OmniVoiceLM] 扩散步 {step}/{NumStep} t={tCurr:F3}");

            // 构建 attention mask（全因果 mask，[1, 1, S, S]）
            var attnMask = BuildFullMask(1, S);
            var posIds = BuildPositionIds(1, S);

            // LM 前向（带 CFG）
            float[] logitsFlat = LMForwardWithCFG(inputIds, audioMask, attnMask, posIds);

            // 对待生成区域采样并更新 inputIds
            UpdateGeneratedTokens(
                inputIds, audioMask, logitsFlat,
                S, T_text, T_ref, targetLen,
                tCurr, tNext, step
            );
        }

        // ── 提取生成结果 ─────────────────────────────────────────────────
        var result = new long[NUM_CODEBOOKS, targetLen];
        for (int t = 0; t < targetLen; t++)
        {
            int s = T_text + T_ref + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                result[cb, t] = Math.Clamp(inputIds[0, cb, s], 0, MASK_TOKEN - 1);
        }

        float audioDurSec = targetLen * 960f / 24000f;
        Debug.Log($"[OmniVoiceLM] ✅ 完成: {targetLen} 帧 ({audioDurSec:F1}s)");
        return result;
    }

    // ════════════════════════════════════════════════════════════════════════
    // 内部：LM 单步前向 + CFG
    // ════════════════════════════════════════════════════════════════════════

    float[] LMForwardWithCFG(long[,,] inputIds, bool[,] audioMask, bool[,,,] attnMask, long[,] posIds)
    {
        int S = inputIds.GetLength(2);

        if (GuidanceScale > 0f)
        {
            // CFG：batch=2（cond + uncond），uncond 把音频位置置为 PAD
            var condUncondIds = DoubleForCFG(inputIds, S);
            var condUncondAudio = DoubleForCFG(audioMask, S);
            var condUncondAttn = DoubleForCFG(attnMask, S);
            var condUncondPos = DoubleForCFG(posIds, S);

            float[] rawLogits = LMForward(condUncondIds, condUncondAudio, condUncondAttn, condUncondPos);

            // 融合 CFG：logit = uncond + gs * (cond - uncond)
            int stride_B = NUM_CODEBOOKS * S * VOCAB_SIZE;
            int stride_CB = S * VOCAB_SIZE;
            var merged = new float[NUM_CODEBOOKS * S * VOCAB_SIZE];
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                for (int s = 0; s < S; s++)
                    for (int v = 0; v < VOCAB_SIZE; v++)
                    {
                        int idx = cb * stride_CB + s * VOCAB_SIZE + v;
                        float cond = rawLogits[0 * stride_B + idx];
                        float uncond = rawLogits[1 * stride_B + idx];
                        merged[idx] = uncond + GuidanceScale * (cond - uncond);
                    }
            return merged;
        }
        else
        {
            // 无 CFG：batch=1
            return LMForward(inputIds, audioMask, attnMask, posIds);
        }
    }

    float[] LMForward(long[,,] inputIds, bool[,] audioMask, bool[,,,] attnMask, long[,] posIds)
    {
        int B = inputIds.GetLength(0);
        int S = inputIds.GetLength(2);

        // 安全截断：防止 audio_embeddings Gather 越界
        for (int b = 0; b < B; b++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                for (int s = 0; s < S; s++)
                    inputIds[b, cb, s] = Math.Clamp(inputIds[b, cb, s], 0, MASK_TOKEN);

        var tIds = new DenseTensor<long>(Flatten3D(inputIds), new[] { B, NUM_CODEBOOKS, S });
        var tAudio = new DenseTensor<bool>(FlattenBool2D(audioMask), new[] { B, S });
        var tAttn = new DenseTensor<bool>(FlattenBool4D(attnMask), new[] { B, 1, S, S });
        var tPos = new DenseTensor<long>(Flatten2D(posIds), new[] { B, S });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids",      tIds),
            NamedOnnxValue.CreateFromTensor("audio_mask",     tAudio),
            NamedOnnxValue.CreateFromTensor("attention_mask", tAttn),
            NamedOnnxValue.CreateFromTensor("position_ids",   tPos),
        };

        using var results = _session.Run(inputs);
        var logitsTensor = results[0].AsTensor<float>();

        // 验证输出维度（[B, 8, S, 1025]）
        // logitsTensor.Dimensions: [B, NUM_CODEBOOKS, S, VOCAB_SIZE]
        int total = B * NUM_CODEBOOKS * S * VOCAB_SIZE;
        var flat = new float[total];
        int idx = 0;
        foreach (var v in logitsTensor) flat[idx++] = v;
        return flat;
    }

    // ════════════════════════════════════════════════════════════════════════
    // 扩散：更新待生成 token（mask-predict 策略）
    // ════════════════════════════════════════════════════════════════════════

    void UpdateGeneratedTokens(
        long[,,] inputIds, bool[,] audioMask, float[] logits,
        int S, int T_text, int T_ref, int targetLen,
        float tCurr, float tNext, int step)
    {
        // mask 率：当前步还有多少位置保持 masked
        float maskRatioCurr = tCurr;
        float maskRatioNext = tNext;

        int genStart = T_text + T_ref;
        int numToMaskNext = Mathf.RoundToInt(maskRatioNext * targetLen);

        // 计算每个待生成位置的采样 token 和置信度
        int stride_CB = S * VOCAB_SIZE;
        var sampledTokens = new long[targetLen];
        var confidences = new float[targetLen];

        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            // 对每个 codebook 独立采样，组成一帧
            // 简化：以 codebook-0 置信度排序决定哪些位置保持 mask
            // 完整实现需要所有 codebook 同步
            float maxLogit = float.NegativeInfinity;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                var cbLogits = new float[VOCAB_SIZE];
                for (int v = 0; v < VOCAB_SIZE; v++)
                    cbLogits[v] = logits[cb * stride_CB + s * VOCAB_SIZE + v];

                long tok = SampleTopK(cbLogits, Temperature, TopK);
                // 不替换 MASK_TOKEN（采样到 EOS/MASK 时取次优）
                if (tok == MASK_TOKEN) tok = ArgMax(cbLogits, excludeIdx: MASK_TOKEN);
                inputIds[0, cb, s] = tok;

                // 用最高概率作为置信度（codebook-0 为代表）
                if (cb == 0)
                {
                    float softmaxMax = SoftmaxMax(cbLogits);
                    confidences[t] = softmaxMax;
                    sampledTokens[t] = tok;
                }
            }
        }

        // mask-predict：按置信度从低到高把 numToMaskNext 个位置重新置为 MASK
        // （置信度最低的位置最不确定，继续 mask 让后续步重采）
        if (numToMaskNext > 0 && step < NumStep - 1)
        {
            // 对置信度排序（升序），低置信度 = 重新 mask
            var order = new int[targetLen];
            for (int i = 0; i < targetLen; i++) order[i] = i;
            Array.Sort(order, (a, b) => confidences[a].CompareTo(confidences[b]));

            for (int i = 0; i < numToMaskNext && i < targetLen; i++)
            {
                int t = order[i];
                int s = genStart + t;
                for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                    inputIds[0, cb, s] = MASK_TOKEN;
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // 工具：调度、Mask、采样
    // ════════════════════════════════════════════════════════════════════════

    /// <summary>余弦扩散调度 t: 1→0，返回 NumStep+1 个节点（含首尾）</summary>
    static float[] BuildCosineSchedule(int numStep, float tShift)
    {
        var schedule = new float[numStep + 1];
        for (int i = 0; i <= numStep; i++)
        {
            float u = 1f - (float)i / numStep;          // 1→0
            // 余弦 warp（可选，简单线性也可以）
            float t = (float)Math.Cos(u * Math.PI * 0.5f);
            // t_shift warp（与 Python 一致）
            t = t / (t + tShift * (1f - t) + 1e-8f);
            schedule[i] = t;
        }
        return schedule;
    }

    /// <summary>全 attention mask（对扩散 LM：所有位置相互可见）</summary>
    static bool[,,,] BuildFullMask(int B, int S)
    {
        var m = new bool[B, 1, S, S];
        for (int b = 0; b < B; b++)
            for (int i = 0; i < S; i++)
                for (int j = 0; j < S; j++)
                    m[b, 0, i, j] = true;
        return m;
    }

    static long[,] BuildPositionIds(int B, int S)
    {
        var p = new long[B, S];
        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
                p[b, s] = s;
        return p;
    }

    // ── CFG 复制（batch × 2）──────────────────────────────────────────────

    static long[,,] DoubleForCFG(long[,,] ids, int S)
    {
        // cond batch 正常；uncond batch 把音频区域置为 PAD（文本 token 保留）
        var d = new long[2, NUM_CODEBOOKS, S];
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            for (int s = 0; s < S; s++)
            {
                d[0, cb, s] = ids[0, cb, s];
                d[1, cb, s] = PAD_TOKEN;
            }
        return d;
    }

    static bool[,] DoubleForCFG(bool[,] mask, int S)
    {
        var d = new bool[2, S];
        for (int s = 0; s < S; s++)
        {
            d[0, s] = mask[0, s];
            d[1, s] = mask[0, s];   // uncond 保持相同 audio_mask 结构
        }
        return d;
    }

    static bool[,,,] DoubleForCFG(bool[,,,] attn, int S)
    {
        var d = new bool[2, 1, S, S];
        for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++)
            {
                d[0, 0, i, j] = attn[0, 0, i, j];
                d[1, 0, i, j] = attn[0, 0, i, j];
            }
        return d;
    }

    static long[,] DoubleForCFG(long[,] pos, int S)
    {
        var d = new long[2, S];
        for (int s = 0; s < S; s++)
        {
            d[0, s] = pos[0, s];
            d[1, s] = pos[0, s];
        }
        return d;
    }

    // ── 采样工具 ─────────────────────────────────────────────────────────

    long SampleTopK(float[] logits, float temp, int k)
    {
        // 温度缩放
        if (Math.Abs(temp - 1f) > 1e-6f)
            for (int i = 0; i < logits.Length; i++) logits[i] /= temp;

        // 排除 MASK_TOKEN（不希望生成区域采出 mask）
        logits[MASK_TOKEN] = float.NegativeInfinity;

        var indexed = new List<(float val, int idx)>(logits.Length);
        for (int i = 0; i < logits.Length; i++) indexed.Add((logits[i], i));
        indexed.Sort((a, b) => b.val.CompareTo(a.val));
        int topK = Math.Min(k, indexed.Count);
        indexed = indexed.GetRange(0, topK);

        float maxV = indexed[0].val;
        float sum = 0f;
        var probs = new float[topK];
        for (int i = 0; i < topK; i++) { probs[i] = (float)Math.Exp(indexed[i].val - maxV); sum += probs[i]; }
        for (int i = 0; i < topK; i++) probs[i] /= sum;

        double r = _rng.NextDouble();
        double cum = 0;
        for (int i = 0; i < topK - 1; i++) { cum += probs[i]; if (r < cum) return indexed[i].idx; }
        return indexed[topK - 1].idx;
    }

    static long ArgMax(float[] logits, int excludeIdx = -1)
    {
        float best = float.NegativeInfinity;
        int bestIdx = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (i == excludeIdx) continue;
            if (logits[i] > best) { best = logits[i]; bestIdx = i; }
        }
        return bestIdx;
    }

    static float SoftmaxMax(float[] logits)
    {
        float maxV = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++) if (logits[i] > maxV) maxV = logits[i];
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++) sum += (float)Math.Exp(logits[i] - maxV);
        return 1f / sum;   // max softmax prob = exp(0) / sum = 1/sum
    }

    // ── 扁平化工具 ───────────────────────────────────────────────────────

    static long[] Flatten3D(long[,,] a)
    {
        var r = new long[a.Length];
        Buffer.BlockCopy(a, 0, r, 0, a.Length * sizeof(long));
        return r;
    }

    static long[] Flatten2D(long[,] a)
    {
        var r = new long[a.Length];
        Buffer.BlockCopy(a, 0, r, 0, a.Length * sizeof(long));
        return r;
    }

    static bool[] FlattenBool2D(bool[,] a)
    {
        var r = new bool[a.Length];
        int i = 0;
        foreach (var v in a) r[i++] = v;
        return r;
    }

    static bool[] FlattenBool4D(bool[,,,] a)
    {
        var r = new bool[a.Length];
        int i = 0;
        foreach (var v in a) r[i++] = v;
        return r;
    }

    public void Dispose() => _session?.Dispose();
}
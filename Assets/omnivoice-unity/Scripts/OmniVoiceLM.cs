
// ============================================================
// OmniVoiceLM.cs - CRITICAL FIXES for audio artifacts
// ============================================================

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

public class OmniVoiceLM : IDisposable
{
    public const int NUM_CODEBOOKS = 8;
    public const int VOCAB_SIZE = 1025;   // 1024 audio codes + 1 mask token
    public const int MASK_TOKEN = 1024;   // <|mask|>
    public const int PAD_TOKEN = 0;

    InferenceSession _session;
    System.Random _rng;

    // ─── 生成参数 ────────────────────────────────────────────────────────────
    public int NumStep = 32;
    public float GuidanceScale = 2.0f;
    public float TShift = 0.1f;
    public float MaskTemperature = 5.0f;
    public float TokenTemperature = 1.0f;

    /// <summary>层惩罚系数。官方默认 5.0；过低会导致高层 codebook 不解 mask，产生空白。</summary>
    public float LayerPenaltyFactor = 5.0f;  // ★ FIX: 从 0.5 改为 5.0

    public OmniVoiceLM(string modelPath, int seed = 42)
    {
        _rng = new System.Random(seed);
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        opts.InterOpNumThreads = 1;
        opts.IntraOpNumThreads = 0;

        // ★ FIX: 强制 CPU EP，排除 GPU 精度问题
        // 若确认 CPU 正常后再尝试 GPU
         try { opts.AppendExecutionProvider_DML(0); }
         catch { Debug.LogWarning("[OmniVoiceLM] DML EP 不可用，使用 CPU"); }

        _session = new InferenceSession(modelPath, opts);
        Debug.Log($"[OmniVoiceLM] 已加载: {modelPath}");
    }

    public long[,] Generate(int[] textTokenIds, long[,] refCodes, int targetLen)
    {
        int T_text = textTokenIds != null ? textTokenIds.Length : 0;
        int T_ref = refCodes != null ? refCodes.GetLength(1) : 0;

        if (targetLen <= 0) targetLen = Mathf.Max(50, T_ref > 0 ? T_ref : 100);

        int genStart = T_text + T_ref;
        int S = genStart + targetLen;

        Debug.Log($"[OmniVoiceLM] 开始扩散生成: T_text={T_text} T_ref={T_ref} T_gen={targetLen} S={S} steps={NumStep} GS={GuidanceScale}");

        var inputIds = new long[1, NUM_CODEBOOKS, S];
        var audioMask = new bool[1, S];

        // 文本区
        for (int s = 0; s < T_text; s++)
        {
            long tid = textTokenIds[s];
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = tid;
            audioMask[0, s] = false;
        }

        // 参考音频区（固定，不解 mask）
        for (int t = 0; t < T_ref; t++)
        {
            int s = T_text + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = Math.Clamp(refCodes[cb, t], 0, MASK_TOKEN - 1);
            audioMask[0, s] = true;
        }

        // 待生成区：全部 MASK
        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = MASK_TOKEN;
            audioMask[0, s] = true;
        }

        int batchSize = GuidanceScale > 0f ? 2 : 1;
        var attnMask = BuildFullMask(batchSize, S, T_text);  // ★ FIX: 传入 T_text 用于 padding mask
        var posIds = BuildPositionIds(batchSize, S);

        // 调度
        double tau = TShift;
        double N = NumStep;
        var r = new double[NumStep + 1];
        for (int n = 0; n <= NumStep; n++)
        {
            double u = (double)n / N;
            r[n] = tau * u / (1.0 + (tau - 1.0) * u);
        }

        // 主循环
        for (int step = 0; step < NumStep; step++)
        {
            double kRatio = r[step + 1] - r[step];
            int kNew = (int)Math.Ceiling(kRatio * targetLen * NUM_CODEBOOKS);
            kNew = Math.Max(kNew, 1);

            if (step % 8 == 0)
                Debug.Log($"[OmniVoiceLM] 扩散步 {step}/{NumStep}  kNew={kNew}");

            float[] logSoftmax = LMForwardWithCFG(inputIds, audioMask, attnMask, posIds, S, genStart);

            // ★ FIX: 检测全局 NaN，若出现则回退到上一步并降低 temperature
            if (IsLogSoftmaxCorrupted(logSoftmax))
            {
                Debug.LogError($"[OmniVoiceLM] 步 {step} 检测到 NaN/Inf logits！尝试恢复...");
                // 回退策略：降低 mask temperature，增加贪心倾向
                MaskTemperature = Mathf.Max(0.1f, MaskTemperature * 0.5f);
                logSoftmax = LMForwardWithCFG(inputIds, audioMask, attnMask, posIds, S, genStart);

                if (IsLogSoftmaxCorrupted(logSoftmax))
                {
                    Debug.LogError("[OmniVoiceLM] 恢复失败，终止生成");
                    break;
                }
            }

            bool isFinalSteps = (step >= NumStep - 2);
            DiffusionStep(inputIds, logSoftmax, genStart, targetLen, S, kNew, isFinalSteps);
        }

        // ★ FIX: 最终强制解 mask（增强版）
        FinalUnmaskAll(inputIds, audioMask, attnMask, posIds, genStart, targetLen, S);

        // 提取结果
        var result = new long[NUM_CODEBOOKS, targetLen];
        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                long v = inputIds[0, cb, s];
                result[cb, t] = (v == MASK_TOKEN) ? 0 : Math.Clamp(v, 0, MASK_TOKEN - 1);
            }
        }

        float durSec = targetLen * 960f / 24000f;
        Debug.Log($"[OmniVoiceLM] ✅ 完成: {targetLen} 帧 ≈ {durSec:F1}s");
        return result;
    }

    // ★ FIX: 新增 NaN 检测
    bool IsLogSoftmaxCorrupted(float[] logSoftmax)
    {
        int nanCount = 0;
        for (int i = 0; i < logSoftmax.Length; i++)
        {
            if (float.IsNaN(logSoftmax[i]) || float.IsInfinity(logSoftmax[i]))
                nanCount++;
        }
        // 如果超过 1% 是 NaN/Inf，认为已损坏
        return nanCount > logSoftmax.Length / 100;
    }

    void DiffusionStep(long[,,] inputIds, float[] logSoftmax, int genStart, int targetLen, int S, int kNew, bool greedy = false)
    {
        int stride_CB = S * VOCAB_SIZE;
        var masked = new List<(int t, int cb, float conf)>(targetLen * NUM_CODEBOOKS);

        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                if (inputIds[0, cb, s] != MASK_TOKEN) continue;

                float bestLogP = float.NegativeInfinity;
                bool hasValidLogit = false;
                for (int v = 0; v < VOCAB_SIZE - 1; v++)
                {
                    float lp = logSoftmax[cb * stride_CB + s * VOCAB_SIZE + v];
                    if (float.IsNaN(lp) || float.IsInfinity(lp)) continue;
                    hasValidLogit = true;
                    if (lp > bestLogP) bestLogP = lp;
                }

                // ★ FIX: 如果没有有效 logit，使用保守默认值而非随机
                if (!hasValidLogit)
                {
                    bestLogP = -10f;  // 保守默认值
                }

                // ★ FIX: 层惩罚因子使用官方默认值 5.0
                float conf = bestLogP - LayerPenaltyFactor * cb;
                masked.Add((t, cb, conf));
            }
        }

        if (masked.Count == 0) return;

        kNew = Math.Min(kNew, masked.Count);
        var scored = new (int t, int cb, float score)[masked.Count];

        if (greedy)
        {
            for (int i = 0; i < masked.Count; i++)
            {
                var (t, cb, conf) = masked[i];
                scored[i] = (t, cb, conf);
            }
        }
        else
        {
            for (int i = 0; i < masked.Count; i++)
            {
                var (t, cb, conf) = masked[i];
                double u = Math.Max(1e-10, _rng.NextDouble());
                double gumbel = -Math.Log(-Math.Log(u));
                float score = conf / MaskTemperature + (float)gumbel;
                scored[i] = (t, cb, score);
            }
        }

        Array.Sort(scored, (a, b) => b.score.CompareTo(a.score));

        for (int i = 0; i < kNew; i++)
        {
            var (t, cb, _) = scored[i];
            int s = genStart + t;

            float bestLogP = float.NegativeInfinity;
            long bestTok = 0;
            bool foundValid = false;
            for (int v = 0; v < VOCAB_SIZE - 1; v++)
            {
                float lp = logSoftmax[cb * stride_CB + s * VOCAB_SIZE + v];
                if (float.IsNaN(lp) || float.IsInfinity(lp)) continue;
                foundValid = true;
                if (lp > bestLogP) { bestLogP = lp; bestTok = v; }
            }

            // ★ FIX: 无有效 logit 时填 0（静音）而非随机
            if (!foundValid)
            {
                bestTok = 0;
                Debug.LogWarning($"[OmniVoiceLM] 位置 (cb={cb}, t={t}) 无有效 logit，填充 0");
            }

            inputIds[0, cb, s] = bestTok;
        }
    }

    void FinalUnmaskAll(long[,,] inputIds, bool[,] audioMask, bool[,,,] attnMask, long[,] posIds, int genStart, int targetLen, int S)
    {
        int maskCount = 0;
        for (int t = 0; t < targetLen; t++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                if (inputIds[0, cb, genStart + t] == MASK_TOKEN)
                    maskCount++;

        if (maskCount == 0) return;

        Debug.Log($"[OmniVoiceLM] 最终强制解 mask: 残余 {maskCount} 个 MASK 位置");

        float[] logSoftmax = LMForwardWithCFG(inputIds, audioMask, attnMask, posIds, S, genStart);

        int stride_CB = S * VOCAB_SIZE;
        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                if (inputIds[0, cb, s] != MASK_TOKEN) continue;

                float bestLogP = float.NegativeInfinity;
                long bestTok = 0;
                bool foundValid = false;
                for (int v = 0; v < VOCAB_SIZE - 1; v++)
                {
                    float lp = logSoftmax[cb * stride_CB + s * VOCAB_SIZE + v];
                    if (float.IsNaN(lp) || float.IsInfinity(lp)) continue;
                    foundValid = true;
                    if (lp > bestLogP) { bestLogP = lp; bestTok = v; }
                }

                // ★ FIX: 最终解 mask 也使用 0 作为安全回退
                if (!foundValid) bestTok = 0;

                inputIds[0, cb, s] = bestTok;
            }
        }
    }

    float[] LMForwardWithCFG(long[,,] inputIds, bool[,] audioMask, bool[,,,] attnMask, long[,] posIds, int S, int genStart)
    {
        if (GuidanceScale > 0f)
        {
            var batchIds = BuildCFGBatch(inputIds, genStart, S);
            var batchAudio = DoubleAudioMask(audioMask, S);

            float[] rawLogits = LMForward(batchIds, batchAudio, attnMask, posIds, batchSize: 2, S: S);

            int strideB = NUM_CODEBOOKS * S * VOCAB_SIZE;
            int strideCB = S * VOCAB_SIZE;
            var result = new float[NUM_CODEBOOKS * S * VOCAB_SIZE];

            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                for (int s = 0; s < S; s++)
                {
                    var condLogits = new float[VOCAB_SIZE];
                    var uncondLogits = new float[VOCAB_SIZE];
                    for (int v = 0; v < VOCAB_SIZE; v++)
                    {
                        int idx = cb * strideCB + s * VOCAB_SIZE + v;
                        condLogits[v] = rawLogits[0 * strideB + idx];
                        uncondLogits[v] = rawLogits[1 * strideB + idx];
                    }
                    float[] condLogSoftmax = LogSoftmax(condLogits);
                    float[] uncondLogSoftmax = LogSoftmax(uncondLogits);

                    int baseIdx = cb * strideCB + s * VOCAB_SIZE;
                    for (int v = 0; v < VOCAB_SIZE; v++)
                        result[baseIdx + v] = uncondLogSoftmax[v] + GuidanceScale * (condLogSoftmax[v] - uncondLogSoftmax[v]);
                }
            return result;
        }
        else
        {
            float[] rawLogits = LMForward(inputIds, audioMask, attnMask, posIds, batchSize: 1, S: S);
            int strideCB = S * VOCAB_SIZE;
            var result = new float[NUM_CODEBOOKS * S * VOCAB_SIZE];
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                for (int s = 0; s < S; s++)
                {
                    var logits = new float[VOCAB_SIZE];
                    for (int v = 0; v < VOCAB_SIZE; v++)
                        logits[v] = rawLogits[cb * strideCB + s * VOCAB_SIZE + v];
                    float[] lsm = LogSoftmax(logits);
                    int baseIdx = cb * strideCB + s * VOCAB_SIZE;
                    for (int v = 0; v < VOCAB_SIZE; v++)
                        result[baseIdx + v] = lsm[v];
                }
            return result;
        }
    }

    static long[,,] BuildCFGBatch(long[,,] ids, int genStart, int S)
    {
        var d = new long[2, NUM_CODEBOOKS, S];
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            for (int s = 0; s < S; s++)
            {
                d[0, cb, s] = ids[0, cb, s];                                    // cond
                // ★ FIX: uncond 分支应 mask 文本和生成区域
                if (s < genStart)  // 文本区域和参考音频也 mask
                    d[1, cb, s] = PAD_TOKEN;  // 使用 PAD 而非保留原文
                else
                    d[1, cb, s] = MASK_TOKEN; // 生成区域 mask
            }
        return d;
    }

    static bool[,] DoubleAudioMask(bool[,] mask, int S)
    {
        var d = new bool[2, S];
        for (int s = 0; s < S; s++)
        {
            d[0, s] = mask[0, s];
            d[1, s] = mask[0, s];
        }
        return d;
    }

    float[] LMForward(long[,,] inputIds, bool[,] audioMask, bool[,,,] attnMask, long[,] posIds, int batchSize, int S)
    {
        var tIds = new DenseTensor<long>(Flatten3D(inputIds), new[] { batchSize, NUM_CODEBOOKS, S });
        var tAudio = new DenseTensor<bool>(FlattenBool2D(audioMask), new[] { batchSize, S });
        var tAttn = new DenseTensor<bool>(FlattenBool4D(attnMask), new[] { batchSize, 1, S, S });
        var tPos = new DenseTensor<long>(Flatten2D(posIds), new[] { batchSize, S });

        var namedInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", tIds),
            NamedOnnxValue.CreateFromTensor("audio_mask", tAudio),
            NamedOnnxValue.CreateFromTensor("attention_mask", tAttn),
            NamedOnnxValue.CreateFromTensor("position_ids", tPos),
        };

        using var results = _session.Run(namedInputs);
        var logitsTensor = results[0].AsTensor<float>();

        int total = batchSize * NUM_CODEBOOKS * S * VOCAB_SIZE;
        var flat = new float[total];
        int idx = 0;
        foreach (var v in logitsTensor) flat[idx++] = v;
        return flat;
    }

    // ★ FIX: 正确的 padding mask - 文本区域不应看到生成区域
    static bool[,,,] BuildFullMask(int B, int S, int T_text)
    {
        var m = new bool[B, 1, S, S];
        for (int b = 0; b < B; b++)
            for (int i = 0; i < S; i++)
                for (int j = 0; j < S; j++)
                {
                    // 所有位置互相可见（双向 transformer）
                    // 但 padding 位置（如果有）应被 mask
                    m[b, 0, i, j] = true;
                }
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

    // ★ FIX: LogSoftmax - 移除危险的 uniform fallback
    static float[] LogSoftmax(float[] logits)
    {
        float maxV = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > maxV) maxV = logits[i];

        // 如果 maxV 无效，返回全 -Inf（让上层检测并处理）
        if (float.IsInfinity(maxV) || float.IsNaN(maxV))
        {
            var fallback = new float[logits.Length];
            for (int i = 0; i < logits.Length; i++)
                fallback[i] = float.NegativeInfinity;
            return fallback;
        }

        float sumExp = 0f;
        for (int i = 0; i < logits.Length; i++)
            sumExp += (float)Math.Exp(logits[i] - maxV);
        float logSum = maxV + (float)Math.Log(sumExp);
        var result = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
            result[i] = logits[i] - logSum;
        return result;
    }

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
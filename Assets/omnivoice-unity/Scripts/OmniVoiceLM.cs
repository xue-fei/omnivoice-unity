using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

/// <summary>
/// OmniVoice 扩散语言模型（Diffusion LM）— ONNX Runtime，Unity C# 实现
/// </summary>
public class OmniVoiceLM : IDisposable
{
    public const int NUM_CODEBOOKS = 8;
    public const int VOCAB_SIZE = 1025;   // 1024 audio codes + 1 mask token
    public const int MASK_TOKEN = 1024;   // <|mask|>
    public const int PAD_TOKEN = 0;

    InferenceSession _session;
    System.Random _rng;

    public int NumStep = 32;
    public float GuidanceScale = 2.0f;
    public float TShift = 0.1f;
    public float MaskTemperature = 5.0f;   // 修正拼写 Tempature->Temperature
    public float TokenTemperature = 1.0f;

    /// <summary>层惩罚系数。原版约为 1.0，C# 旧版 5.0 过强会导致高层完全不解 mask。</summary>
    public float LayerPenaltyFactor = 1.0f;

    public OmniVoiceLM(string modelPath, int seed = 42)
    {
        _rng = new System.Random(seed);
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        opts.InterOpNumThreads = 1;
        opts.IntraOpNumThreads = 0; // 使用物理核心数

        // ★ 修复：强制 CPU EP，排除 DirectML/CUDA 精度问题导致 logits 异常
        // 若后续确认 CPU 正常、需要 GPU 加速，可再开启 DML/CUDA
        // Debug.Log("[OmniVoiceLM] 强制使用 CPUExecutionProvider");
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
        var attnMask = BuildFullMask(batchSize, S);
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

            // ★ 调试：检查 logSoftmax 是否异常
            if (step == 0 || step == NumStep - 1)
            {
                bool hasNaN = logSoftmax.Any(float.IsNaN);
                bool hasInf = logSoftmax.Any(float.IsInfinity);
                if (hasNaN || hasInf)
                    Debug.LogError($"[OmniVoiceLM] 步 {step}: logSoftmax 包含 NaN={hasNaN} Inf={hasInf}！请检查 ONNX 输入（如文本 token ID 越界或 EP 兼容性）");
            }

            DiffusionStep(inputIds, logSoftmax, genStart, targetLen, S, kNew);
        }

        // 提取结果
        var result = new long[NUM_CODEBOOKS, targetLen];
        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                long v = inputIds[0, cb, s];
                result[cb, t] = v == MASK_TOKEN ? 0 : Math.Clamp(v, 0, MASK_TOKEN - 1);
            }
        }

        float durSec = targetLen * 960f / 24000f;
        Debug.Log($"[OmniVoiceLM] ✅ 完成: {targetLen} 帧 ≈ {durSec:F1}s");
        return result;
    }

    void DiffusionStep(long[,,] inputIds, float[] logSoftmax, int genStart, int targetLen, int S, int kNew)
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
                for (int v = 0; v < VOCAB_SIZE - 1; v++)
                {
                    float lp = logSoftmax[cb * stride_CB + s * VOCAB_SIZE + v];
                    if (float.IsNaN(lp)) { bestLogP = float.NaN; break; }
                    if (lp > bestLogP) bestLogP = lp;
                }

                if (float.IsNaN(bestLogP))
                {
                    // 若 logits 异常，赋予随机 confidence 避免全选 0
                    bestLogP = -10f + (float)_rng.NextDouble();
                }

                // ★ 修复：LayerPenalty 改为 1.0 * cb，原版约 0.5~1.0，旧版 5.0 过强
                float conf = bestLogP - LayerPenaltyFactor * cb;
                masked.Add((t, cb, conf));
            }
        }

        if (masked.Count == 0) return;

        kNew = Math.Min(kNew, masked.Count);
        var scored = new (int t, int cb, float score)[masked.Count];
        for (int i = 0; i < masked.Count; i++)
        {
            var (t, cb, conf) = masked[i];
            double u = Math.Max(1e-10, _rng.NextDouble());
            double gumbel = -Math.Log(-Math.Log(u));
            float score = conf / MaskTemperature + (float)gumbel;
            scored[i] = (t, cb, score);
        }

        Array.Sort(scored, (a, b) => b.score.CompareTo(a.score));

        for (int i = 0; i < kNew; i++)
        {
            var (t, cb, _) = scored[i];
            int s = genStart + t;

            float bestLogP = float.NegativeInfinity;
            long bestTok = 0;
            for (int v = 0; v < VOCAB_SIZE - 1; v++)
            {
                float lp = logSoftmax[cb * stride_CB + s * VOCAB_SIZE + v];
                if (float.IsNaN(lp)) { bestTok = _rng.Next(VOCAB_SIZE - 1); bestLogP = 0; break; }
                if (lp > bestLogP) { bestLogP = lp; bestTok = v; }
            }

            inputIds[0, cb, s] = bestTok;
        }
    }

    float[] LMForwardWithCFG(long[,,] inputIds, bool[,] audioMask, bool[,,,] attnMask, long[,] posIds, int S, int genStart)
    {
        if (GuidanceScale > 0f)
        {
            // ★ 修复：uncond 目标区使用 MASK_TOKEN 而非 PAD_TOKEN，避免误导模型
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
                d[1, cb, s] = (s >= genStart) ? (long)MASK_TOKEN : ids[0, cb, s]; // uncond：目标区 MASK，其余保留
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
        // ★ 修复：不再截断文本 token（如 151669），ONNX 模型应支持完整 Qwen2 vocab
        // 若此处出现 NaN，请检查 tokenizer.json 与模型版本是否匹配，或尝试强制 CPU EP

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
        }
        ;

        using var results = _session.Run(namedInputs);
        var logitsTensor = results[0].AsTensor<float>();

        int total = batchSize * NUM_CODEBOOKS * S * VOCAB_SIZE;
        var flat = new float[total];
        int idx = 0;
        foreach (var v in logitsTensor) flat[idx++] = v;
        return flat;
    }

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

    static float[] LogSoftmax(float[] logits)
    {
        float maxV = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++) if (logits[i] > maxV) maxV = logits[i];

        // ★ 修复：防护全 -Inf / NaN 导致的异常
        if (float.IsInfinity(maxV) || float.IsNaN(maxV))
        {
            float uniformLogProb = -(float)Math.Log(logits.Length);
            var fallback = new float[logits.Length];
            for (int i = 0; i < logits.Length; i++) fallback[i] = uniformLogProb;
            return fallback;
        }

        float sumExp = 0f;
        for (int i = 0; i < logits.Length; i++) sumExp += (float)Math.Exp(logits[i] - maxV);
        float logSum = maxV + (float)Math.Log(sumExp);
        var result = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++) result[i] = logits[i] - logSum;
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
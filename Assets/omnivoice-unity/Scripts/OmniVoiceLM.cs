using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

/// <summary>
/// OmniVoice 扩散语言模型（Diffusion LM）— ONNX Runtime，Unity C# 实现
///
/// ★ 架构要点（来自论文 arxiv:2604.00688 §3.4 + AFun9/Omnivoice-onnx README）：
///
///   模型是双向 Transformer（NAR），不是自回归。
///
///   序列布局（全部在一个序列里，双向 attention）：
///     [text_tokens | ref_audio_codes | target_masked_codes]
///     audio_mask: false 对 text 区域，true 对两段音频区域
///
///   推理循环（iterative unmasking，共 NumStep=32 步）：
///     每步：
///       1. LM 前向（batch=2 做 CFG，或 batch=1 不做 CFG）
///       2. logit → log_softmax，CFG 在 log_softmax 空间融合
///       3. 用 max(log_softmax) 作为 confidence score
///       4. 对 confidence 施加 layer_penalty（深层 codebook 惩罚）
///       5. 按调度确定本步要新 unmask 的数量 k_n
///       6. 从仍是 MASK 的位置中，按 confidence 最高的 k_n 个解 mask
///          (confidence 用温度 T_mask=5 采样，而非贪心选 —— 引入随机性)
///       7. 解 mask 位置的 token 用 argmax 决定（greedy）
///
///   调度公式（论文式 3）：
///     r_n = τ*(n/N) / (1 + (τ-1)*(n/N))，τ=0.1，N=NumStep
///     k_n = r_n - r_{n-1}（本步新 unmask 的比例）
///
///   CFG（在 log_softmax 空间）：
///     log_p_cfg = log_p_uncond + gs * (log_p_cond - log_p_uncond)
///
/// ONNX 接口（AFun9/Omnivoice-onnx §4.1）：
///   输入  input_ids       int64  [B, 8, S]
///   输入  audio_mask      bool   [B, S]
///   输入  attention_mask  bool   [B, 1, S, S]  （全 true，双向）
///   输入  position_ids    int64  [B, S]
///   输出  logits          float  [B, 8, S, 1025]
/// </summary>
public class OmniVoiceLM : IDisposable
{
    // ─── 常量 ────────────────────────────────────────────────────────────────
    public const int NUM_CODEBOOKS = 8;
    public const int VOCAB_SIZE = 1025;   // 1024 audio codes + 1 mask token
    public const int MASK_TOKEN = 1024;   // <|mask|>
    public const int PAD_TOKEN = 0;

    InferenceSession _session;
    System.Random _rng;

    // ─── 生成参数（与 Python 原版对齐）─────────────────────────────────────
    /// <summary>扩散步数（原版默认 32）</summary>
    public int NumStep = 32;

    /// <summary>CFG 引导强度（原版默认 2.0）</summary>
    public float GuidanceScale = 2.0f;

    /// <summary>调度时移 τ（原版默认 0.1）</summary>
    public float TShift = 0.1f;

    /// <summary>mask 位置选择的温度（原版默认 5.0，控制解码随机性）</summary>
    public float MaskTempature = 5.0f;

    /// <summary>token 采样温度（原版默认 1.0，argmax 时设为 0）</summary>
    public float TokenTemperature = 1.0f;

    /// <summary>层惩罚系数（原版默认 5.0，鼓励低层 codebook 先解 mask）</summary>
    public float LayerPenaltyFactor = 5.0f;

    public OmniVoiceLM(string modelPath, int seed = 42)
    {
        _rng = new System.Random(seed);
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        // DirectML = Windows GPU；无 GPU 时自动回退 CPU
        try { opts.AppendExecutionProvider_DML(0); }
        catch { Debug.LogWarning("[OmniVoiceLM] DML EP 不可用，使用 CPU"); }
        _session = new InferenceSession(modelPath, opts);
        Debug.Log($"[OmniVoiceLM] 已加载: {modelPath}");
    }

    // ════════════════════════════════════════════════════════════════════════
    // 公开入口
    // ════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// 扩散 LM 语音克隆/TTS 生成。
    ///
    /// 参数：
    ///   textTokenIds — 完整文本 prompt（Qwen2Tokenizer.BuildPrompt 输出）
    ///   refCodes     — 参考音频 codes [8, T_ref]（AudioTokenizer.Encode 输出），可为 null
    ///   targetLen    — 目标生成帧数（24000/960 帧/秒）
    ///
    /// 返回：生成的 audio codes [8, targetLen]
    /// </summary>
    public long[,] Generate(int[] textTokenIds, long[,] refCodes, int targetLen)
    {
        int T_text = textTokenIds != null ? textTokenIds.Length : 0;
        int T_ref = refCodes != null ? refCodes.GetLength(1) : 0;

        if (targetLen <= 0) targetLen = Mathf.Max(50, T_ref > 0 ? T_ref : 100);

        int genStart = T_text + T_ref;
        int S = genStart + targetLen;

        Debug.Log($"[OmniVoiceLM] 开始扩散生成: T_text={T_text} T_ref={T_ref} T_gen={targetLen} S={S} steps={NumStep}");

        // ── 构建初始序列 ─────────────────────────────────────────────────
        // input_ids [1, 8, S]：文本区用文本 token id（各 codebook 相同），
        //                       音频区用实际 code 或 MASK_TOKEN
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

        // 参考音频区（不会被 mask，保持固定）
        for (int t = 0; t < T_ref; t++)
        {
            int s = T_text + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = Math.Clamp(refCodes[cb, t], 0, MASK_TOKEN - 1);
            audioMask[0, s] = true;
        }

        // 待生成区：全部初始化为 MASK_TOKEN
        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = MASK_TOKEN;
            audioMask[0, s] = true;
        }

        // attention mask：全 true（双向，NAR 模型）
        var attnMask = BuildFullMask(GuidanceScale > 0f ? 2 : 1, S);
        var posIds = BuildPositionIds(GuidanceScale > 0f ? 2 : 1, S);

        // ── 调度：计算每步的累计 unmask 比例 ────────────────────────────
        // r_n = τ*(n/N) / (1 + (τ-1)*(n/N))，论文式(3)，τ=TShift
        double tau = TShift;
        double N = NumStep;
        var r = new double[NumStep + 1];
        for (int n = 0; n <= NumStep; n++)
        {
            double u = (double)n / N;
            r[n] = tau * u / (1.0 + (tau - 1.0) * u);
        }
        // r[0]=0, r[NumStep]≈1

        // ── 主循环 ───────────────────────────────────────────────────────
        for (int step = 0; step < NumStep; step++)
        {
            // 本步新 unmask 的 token 数（按 targetLen 缩放）
            double kRatio = r[step + 1] - r[step];
            int kNew = (int)Math.Ceiling(kRatio * targetLen * NUM_CODEBOOKS);
            kNew = Math.Max(kNew, 1);

            if (step % 8 == 0)
                Debug.Log($"[OmniVoiceLM] 扩散步 {step}/{NumStep}  kNew={kNew}");

            // LM 前向
            float[] logSoftmax = LMForwardWithCFG(inputIds, audioMask, attnMask, posIds, S);

            // 提取所有 (codebook, genPosition) 的 confidence（max log-softmax 值）
            // 并施加层惩罚，找出仍是 MASK 的位置中 confidence 最高的 kNew 个解 mask
            DiffusionStep(inputIds, logSoftmax, genStart, targetLen, S, kNew);
        }

        // ── 提取结果 ─────────────────────────────────────────────────────
        var result = new long[NUM_CODEBOOKS, targetLen];
        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                long v = inputIds[0, cb, s];
                // 如果某位置仍是 MASK（极少数边缘情况），用 0 填充
                result[cb, t] = v == MASK_TOKEN ? 0 : Math.Clamp(v, 0, MASK_TOKEN - 1);
            }
        }

        float durSec = targetLen * 960f / 24000f;
        Debug.Log($"[OmniVoiceLM] ✅ 完成: {targetLen} 帧 ≈ {durSec:F1}s");
        return result;
    }

    // ════════════════════════════════════════════════════════════════════════
    // 扩散单步：解 mask
    // ════════════════════════════════════════════════════════════════════════

    void DiffusionStep(
        long[,,] inputIds, float[] logSoftmax,
        int genStart, int targetLen, int S, int kNew)
    {
        // stride：logSoftmax 维度是 [B_eff, 8, S, 1025]
        // 我们只用 batch=0（cond 侧，CFG 已融合到 logSoftmax 里）
        int stride_CB = S * VOCAB_SIZE;

        // 找出仍是 MASK 的 (t, cb) 对，计算其 confidence
        // confidence = max(log_softmax) over vocab（不含 MASK_TOKEN 自身）
        // layer_penalty：confidence -= LayerPenaltyFactor * cb
        var masked = new List<(int t, int cb, float conf)>(targetLen * NUM_CODEBOOKS);

        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                if (inputIds[0, cb, s] != MASK_TOKEN) continue;

                // 找此位置 log-softmax 最大值（排除 MASK_TOKEN 索引）
                float bestLogP = float.NegativeInfinity;
                for (int v = 0; v < VOCAB_SIZE - 1; v++) // VOCAB_SIZE-1 = 1024 = MASK_TOKEN，跳过
                {
                    float lp = logSoftmax[cb * stride_CB + s * VOCAB_SIZE + v];
                    if (lp > bestLogP) bestLogP = lp;
                }

                // 层惩罚：层越深（cb 越大），惩罚越重
                float conf = bestLogP - LayerPenaltyFactor * cb;
                masked.Add((t, cb, conf));
            }
        }

        if (masked.Count == 0) return;

        // 用温度 MaskTempature 对 confidence 做 Gumbel 采样（等效于 Gumbel top-k）
        // 简化实现：对 conf/T 加 Gumbel 噪声后取 top-k
        kNew = Math.Min(kNew, masked.Count);
        var scored = new (int t, int cb, float score)[masked.Count];
        for (int i = 0; i < masked.Count; i++)
        {
            var (t, cb, conf) = masked[i];
            // Gumbel 噪声 = -log(-log(U))，U ~ Uniform(0,1)
            double u = Math.Max(1e-10, _rng.NextDouble());
            double gumbel = -Math.Log(-Math.Log(u));
            float score = conf / MaskTempature + (float)gumbel;
            scored[i] = (t, cb, score);
        }

        // 按 score 降序，取前 kNew 个解 mask
        Array.Sort(scored, (a, b) => b.score.CompareTo(a.score));

        for (int i = 0; i < kNew; i++)
        {
            var (t, cb, _) = scored[i];
            int s = genStart + t;

            // token 用 argmax（greedy，temperature=0 等效）
            float bestLogP = float.NegativeInfinity;
            long bestTok = 0;
            for (int v = 0; v < VOCAB_SIZE - 1; v++)
            {
                float lp = logSoftmax[cb * stride_CB + s * VOCAB_SIZE + v];
                if (lp > bestLogP) { bestLogP = lp; bestTok = v; }
            }

            // 如果 TokenTemperature > 0，可做温度采样（原版默认 greedy/argmax）
            if (TokenTemperature > 0f && TokenTemperature != 0f)
            {
                // 仍用 argmax（原版对 token 是 argmax，只对 mask 位置选择用温度）
            }

            inputIds[0, cb, s] = bestTok;
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // LM 前向 + CFG（在 log-softmax 空间融合）
    // ════════════════════════════════════════════════════════════════════════

    float[] LMForwardWithCFG(
        long[,,] inputIds, bool[,] audioMask,
        bool[,,,] attnMask, long[,] posIds, int S)
    {
        if (GuidanceScale > 0f)
        {
            // batch=2：cond + uncond
            // uncond：把生成区的 input_ids 全置为 PAD（文本和 ref 区域保留）
            var condIds = DoubleForCFG(inputIds, S, T_text: inputIds.GetLength(2) - S);
            // 注意：这里 T_text 信息没有直接传入，改为通过 audioMask 判断
            var batchIds = BuildCFGBatch(inputIds, audioMask, S);
            var batchAudio = DoubleAudioMask(audioMask, S);

            float[] rawLogits = LMForward(batchIds, batchAudio, attnMask, posIds, batchSize: 2, S: S);

            // 转换为 log_softmax 并在该空间做 CFG 融合
            int strideB = NUM_CODEBOOKS * S * VOCAB_SIZE;
            int strideCB = S * VOCAB_SIZE;
            var result = new float[NUM_CODEBOOKS * S * VOCAB_SIZE];

            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                for (int s = 0; s < S; s++)
                {
                    // cond logits
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

                    // CFG 融合
                    int baseIdx = cb * strideCB + s * VOCAB_SIZE;
                    for (int v = 0; v < VOCAB_SIZE; v++)
                        result[baseIdx + v] = uncondLogSoftmax[v] + GuidanceScale * (condLogSoftmax[v] - uncondLogSoftmax[v]);
                }
            return result;
        }
        else
        {
            // 无 CFG：batch=1，直接 log-softmax
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

    // ── CFG batch 构建：uncond 侧把音频区全置为 PAD ──────────────────────
    static long[,,] BuildCFGBatch(long[,,] ids, bool[,] audioMask, int S)
    {
        var d = new long[2, NUM_CODEBOOKS, S];
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            for (int s = 0; s < S; s++)
            {
                d[0, cb, s] = ids[0, cb, s];                              // cond：原样
                d[1, cb, s] = audioMask[0, s] ? (long)PAD_TOKEN : ids[0, cb, s]; // uncond：音频区置 PAD，文本区保留
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

    // 注：这个重载是旧版留的，已被 BuildCFGBatch 替代，保留以防外部引用
    static long[,,] DoubleForCFG(long[,,] ids, int S, int T_text)
    {
        return BuildCFGBatch(ids, new bool[1, S], S); // 占位，实际走 BuildCFGBatch
    }

    // ── ONNX 单次前向 ────────────────────────────────────────────────────
    float[] LMForward(
        long[,,] inputIds, bool[,] audioMask,
        bool[,,,] attnMask, long[,] posIds,
        int batchSize, int S)
    {
        // 安全截断
        for (int b = 0; b < batchSize; b++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                for (int s = 0; s < S; s++)
                    inputIds[b, cb, s] = Math.Clamp(inputIds[b, cb, s], 0, MASK_TOKEN);

        var tIds = new DenseTensor<long>(Flatten3D(inputIds), new[] { batchSize, NUM_CODEBOOKS, S });
        var tAudio = new DenseTensor<bool>(FlattenBool2D(audioMask), new[] { batchSize, S });
        var tAttn = new DenseTensor<bool>(FlattenBool4D(attnMask), new[] { batchSize, 1, S, S });
        var tPos = new DenseTensor<long>(Flatten2D(posIds), new[] { batchSize, S });

        var namedInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids",      tIds),
            NamedOnnxValue.CreateFromTensor("audio_mask",     tAudio),
            NamedOnnxValue.CreateFromTensor("attention_mask", tAttn),
            NamedOnnxValue.CreateFromTensor("position_ids",   tPos),
        };

        using var results = _session.Run(namedInputs);
        var logitsTensor = results[0].AsTensor<float>();

        int total = batchSize * NUM_CODEBOOKS * S * VOCAB_SIZE;
        var flat = new float[total];
        int idx = 0;
        foreach (var v in logitsTensor) flat[idx++] = v;
        return flat;
    }

    // ════════════════════════════════════════════════════════════════════════
    // 工具方法
    // ════════════════════════════════════════════════════════════════════════

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
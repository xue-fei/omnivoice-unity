using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

/// <summary>
/// OmniVoice 语言模型 ONNX 推理 + 扩散采样循环
///
/// LM IO:
///   input_ids       [B, 8, S]  int64
///   audio_mask      [B, S]     bool
///   attention_mask  [B, 1, S, S] bool
///   position_ids    [B, S]     int64
///   → logits        [B, 8, S, 1025]  float32
///
/// 简化的生成流程（voice clone 单步 teacher-forced）：
///   1. 把参考 codes 拼到 BOS，加文本 token
///   2. 自回归采样 top-k，直到 EOS 或 max_len
///   3. 用 CFG（条件 / 无条件双 batch）
/// </summary>
public class OmniVoiceLM : IDisposable
{
    // ─── 词表常量（与 OmniVoice 模型一致）────────────────────
    public const int VOCAB_SIZE = 1025;  // 1024 audio + 1 special
    public const int NUM_CODEBOOKS = 8;
    public const int BOS_TOKEN = 1024;  // audio BOS
    public const int EOS_TOKEN = 1024;  // 模型用同一 id 做 EOS 判断（第 0 codebook）
    public const int TEXT_BOS = 1;
    public const int TEXT_EOS = 2;
    public const int PAD_TOKEN = 0;

    InferenceSession _session;

    // 采样超参
    public int TopK = 50;
    public float Temperature = 1.0f;
    public float GuidanceScale = 2.0f;  // CFG 强度
    public int MaxNewTokens = 1500;  // 对应约 60s @ 24kHz

    System.Random _rng = new System.Random(42);

    public OmniVoiceLM(string modelPath)
    {
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        // fp16 在 CPU 上反而慢，若有 DirectML / CUDA 可在此处加 EP
        _session = new InferenceSession(modelPath, opts);
        Debug.Log($"[OmniVoiceLM] Loaded: {modelPath}");
    }

    // ── 单步 LM 前向 ──────────────────────────────────────────
    // 返回 logits [B, 8, S, 1025]（摊平为 float[]）
    float[] LMForward(long[,,] inputIds,   // [B, 8, S]
                       bool[,] audioMask,  // [B, S]
                       bool[,,,] attnMask,  // [B, 1, S, S]
                       long[,] posIds)     // [B, S]
    {
        int B = inputIds.GetLength(0);
        int S = inputIds.GetLength(2);

        // 展开各张量为一维（C# row-major）
        var flatIds = Flatten3D(inputIds);
        var flatAudio = FlattenBool2D(audioMask);
        var flatAttn = FlattenBool4D(attnMask);
        var flatPos = Flatten2D(posIds);

        var tIds = new DenseTensor<long>(flatIds, new[] { B, NUM_CODEBOOKS, S });
        var tAudio = new DenseTensor<bool>(flatAudio, new[] { B, S });
        var tAttn = new DenseTensor<bool>(flatAttn, new[] { B, 1, S, S });
        var tPos = new DenseTensor<long>(flatPos, new[] { B, S });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids",      tIds),
            NamedOnnxValue.CreateFromTensor("audio_mask",     tAudio),
            NamedOnnxValue.CreateFromTensor("attention_mask", tAttn),
            NamedOnnxValue.CreateFromTensor("position_ids",   tPos),
        };

        using var results = _session.Run(inputs);
        var logitsTensor = results[0].AsTensor<float>();
        // [B, 8, S, 1025] → flat float[]
        int total = B * NUM_CODEBOOKS * S * VOCAB_SIZE;
        var flat = new float[total];
        int idx = 0;
        foreach (var v in logitsTensor) flat[idx++] = v;
        return flat;
    }

    // ── 语音克隆生成入口 ──────────────────────────────────────
    /// <param name="textTokenIds">文本 token id 序列（BOS/EOS 已包含）</param>
    /// <param name="refCodes">参考音频 codes [8, T_ref]，null 则为 auto 模式</param>
    public long[,] Generate(int[] textTokenIds, long[,] refCodes = null)
    {
        // 构造初始 input_ids
        // 格式（简化版 OmniVoice context）：
        //   [PAD...PAD, text_tokens..., AUDIO_BOS, ref_codes..., AUDIO_BOS(生成起点)]
        int T_ref = refCodes != null ? refCodes.GetLength(1) : 0;
        int prefixLen = textTokenIds.Length + 1 + T_ref + 1; // text + BOS + ref + BOS

        // 构建 prefix input_ids [1, 8, prefixLen]
        var prefix = new long[1, NUM_CODEBOOKS, prefixLen];
        // 文本段（所有 codebook 用 PAD，第 0 codebook 填文本 token）
        for (int i = 0; i < textTokenIds.Length; i++)
            prefix[0, 0, i] = textTokenIds[i];
        // AUDIO_BOS
        int bosPos = textTokenIds.Length;
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            prefix[0, cb, bosPos] = BOS_TOKEN;
        // 参考音频 codes
        for (int t = 0; t < T_ref; t++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                prefix[0, cb, bosPos + 1 + t] = refCodes[cb, t];
        // 生成起点 BOS
        int genStart = bosPos + 1 + T_ref;
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            prefix[0, cb, genStart] = BOS_TOKEN;

        // audio_mask: true 表示该位置是音频 token
        var audioMask = new bool[1, prefixLen];
        for (int i = bosPos; i < prefixLen; i++)
            audioMask[0, i] = true;

        // 自回归生成
        var generatedCodes = new List<long[]>(); // 每步一组 8 个 code
        var currentIds = prefix;
        int curLen = prefixLen;

        for (int step = 0; step < MaxNewTokens; step++)
        {
            // attention_mask: 因果 mask [1, 1, S, S]
            var attnMask = BuildCausalMask(1, curLen);
            // position_ids
            var posIds = BuildPositionIds(1, curLen);

            // CFG: 拼接 cond + uncond（batch=2）
            var condUncondIds = DoubleForCFG(currentIds, curLen);
            var condUncondAudio = DoubleForCFG(audioMask, curLen);
            var condUncondAttn = DoubleForCFG(attnMask);
            var condUncondPos = DoubleForCFG(posIds, curLen);

            var logits = LMForward(condUncondIds, condUncondAudio,
                                   condUncondAttn, condUncondPos);

            // 取最后一步 logits，应用 CFG
            // logits shape: [2, 8, S, 1025]  → 取 S-1 位置
            float[] sampledCodes = SampleWithCFG(logits, curLen, GuidanceScale);

            // 判断是否 EOS（第 0 codebook 为 EOS_TOKEN）
            if ((long)sampledCodes[0] == EOS_TOKEN) break;

            // 追加到生成结果
            var stepCodes = new long[NUM_CODEBOOKS];
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                stepCodes[cb] = (long)sampledCodes[cb];
            generatedCodes.Add(stepCodes);

            // 拼接新 token 到 currentIds
            currentIds = AppendToken(currentIds, stepCodes, curLen);
            audioMask = AppendAudioMask(audioMask, curLen);
            curLen++;
        }

        if (generatedCodes.Count == 0)
        {
            Debug.LogWarning("[OmniVoiceLM] 未生成任何 token");
            return new long[NUM_CODEBOOKS, 0];
        }

        // 整理输出 [8, T_gen]
        int T_gen = generatedCodes.Count;
        var output = new long[NUM_CODEBOOKS, T_gen];
        for (int t = 0; t < T_gen; t++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                output[cb, t] = generatedCodes[t][cb];

        Debug.Log($"[OmniVoiceLM] 生成 {T_gen} 帧 ({T_gen * 0.04f:F1}s @ 24kHz)");
        return output;
    }

    // ── CFG 采样 ──────────────────────────────────────────────
    float[] SampleWithCFG(float[] logits, int S, float guidanceScale)
    {
        // logits: [2, 8, S, 1025]  batch=0=cond, batch=1=uncond
        var result = new float[NUM_CODEBOOKS];
        int stride_B = NUM_CODEBOOKS * S * VOCAB_SIZE;
        int stride_CB = S * VOCAB_SIZE;
        int t = S - 1; // 最后一步

        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
        {
            // CFG: logit = uncond + scale * (cond - uncond)
            var cfgLogits = new float[VOCAB_SIZE];
            for (int v = 0; v < VOCAB_SIZE; v++)
            {
                int idxCond = 0 * stride_B + cb * stride_CB + t * VOCAB_SIZE + v;
                int idxUncond = 1 * stride_B + cb * stride_CB + t * VOCAB_SIZE + v;
                float cond = logits[idxCond];
                float uncond = logits[idxUncond];
                cfgLogits[v] = uncond + guidanceScale * (cond - uncond);
            }

            // Temperature + Top-K 采样
            result[cb] = SampleTopK(cfgLogits, Temperature, TopK);
        }
        return result;
    }

    float SampleTopK(float[] logits, float temp, int k)
    {
        // 温度缩放
        if (Math.Abs(temp - 1f) > 1e-6f)
            for (int i = 0; i < logits.Length; i++) logits[i] /= temp;

        // Top-K 筛选：找前 k 个最大值
        var indexed = new List<(float val, int idx)>(logits.Length);
        for (int i = 0; i < logits.Length; i++) indexed.Add((logits[i], i));
        indexed.Sort((a, b) => b.val.CompareTo(a.val));
        indexed = indexed.GetRange(0, Math.Min(k, indexed.Count));

        // Softmax over top-k
        float maxV = indexed[0].val;
        float sum = 0f;
        var probs = new float[indexed.Count];
        for (int i = 0; i < indexed.Count; i++) { probs[i] = (float)Math.Exp(indexed[i].val - maxV); sum += probs[i]; }
        for (int i = 0; i < probs.Length; i++) probs[i] /= sum;

        // 按概率采样
        double r = _rng.NextDouble();
        double cum = 0;
        for (int i = 0; i < probs.Length - 1; i++) { cum += probs[i]; if (r < cum) return indexed[i].idx; }
        return indexed[^1].idx;
    }

    // ── 工具方法 ─────────────────────────────────────────────
    bool[,,,] BuildCausalMask(int B, int S)
    {
        var m = new bool[B, 1, S, S];
        for (int b = 0; b < B; b++)
            for (int i = 0; i < S; i++)
                for (int j = 0; j <= i; j++)
                    m[b, 0, i, j] = true;
        return m;
    }

    long[,] BuildPositionIds(int B, int S)
    {
        var p = new long[B, S];
        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++) p[b, s] = s;
        return p;
    }

    // CFG: 沿 batch 维复制一份作为 uncond（全 PAD 无条件）
    long[,,] DoubleForCFG(long[,,] ids, int S)
    {
        int CB = ids.GetLength(1);
        var d = new long[2, CB, S];
        for (int cb = 0; cb < CB; cb++)
            for (int s = 0; s < S; s++) { d[0, cb, s] = ids[0, cb, s]; d[1, cb, s] = PAD_TOKEN; }
        return d;
    }
    bool[,] DoubleForCFG(bool[,] mask, int S)
    {
        var d = new bool[2, S];
        for (int s = 0; s < S; s++) { d[0, s] = mask[0, s]; d[1, s] = false; }
        return d;
    }
    bool[,,,] DoubleForCFG(bool[,,,] attn)
    {
        int S = attn.GetLength(2);
        var d = new bool[2, 1, S, S];
        for (int i = 0; i < S; i++) for (int j = 0; j < S; j++) { d[0, 0, i, j] = attn[0, 0, i, j]; d[1, 0, i, j] = attn[0, 0, i, j]; }
        return d;
    }
    long[,] DoubleForCFG(long[,] pos, int S)
    {
        var d = new long[2, S];
        for (int s = 0; s < S; s++) { d[0, s] = pos[0, s]; d[1, s] = pos[0, s]; }
        return d;
    }

    long[,,] AppendToken(long[,,] ids, long[] newCodes, int curLen)
    {
        var n = new long[1, NUM_CODEBOOKS, curLen + 1];
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
        {
            for (int s = 0; s < curLen; s++) n[0, cb, s] = ids[0, cb, s];
            n[0, cb, curLen] = newCodes[cb];
        }
        return n;
    }

    bool[,] AppendAudioMask(bool[,] mask, int curLen)
    {
        var n = new bool[1, curLen + 1];
        for (int s = 0; s < curLen; s++) n[0, s] = mask[0, s];
        n[0, curLen] = true;
        return n;
    }

    // 多维数组展平辅助
    static long[] Flatten3D(long[,,] a) { var r = new long[a.Length]; Buffer.BlockCopy(a, 0, r, 0, a.Length * 8); return r; }
    static long[] Flatten2D(long[,] a) { var r = new long[a.Length]; Buffer.BlockCopy(a, 0, r, 0, a.Length * 8); return r; }
    static bool[] FlattenBool2D(bool[,] a) { var r = new bool[a.Length]; int i = 0; foreach (var v in a) r[i++] = v; return r; }
    static bool[] FlattenBool4D(bool[,,,] a) { var r = new bool[a.Length]; int i = 0; foreach (var v in a) r[i++] = v; return r; }

    public void Dispose() => _session?.Dispose();
}
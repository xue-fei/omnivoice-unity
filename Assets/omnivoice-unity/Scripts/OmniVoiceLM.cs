using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

public class OmniVoiceLM : IDisposable
{
    // ─── 基础常量 ────────────────────────────────────────
    public const int NUM_CODEBOOKS = 8;
    public const int AUDIO_BOS = 1024;
    public const int EOS_TOKEN = 1024;      // 多数 OmniVoice 版本使用 1024 作为音频 EOS
    public const int PAD_TOKEN = 0;

    InferenceSession _session;
    public int TopK = 50;
    public float Temperature = 1.0f;
    public float GuidanceScale = 0.0f;      // ⚠️ 首测务必设为 0，避免 CFG 概率坍缩
    public int MaxNewTokens = 200;          // ⚠️ 首测调低，验证通路后再调高
    System.Random _rng = new System.Random(42);

    public OmniVoiceLM(string modelPath)
    {
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        _session = new InferenceSession(modelPath, opts);
        Debug.Log($"[OmniVoiceLM] Loaded: {modelPath}");
    }

    // ── 单步 LM 前向 ──────────────────────────────────────
    float[] LMForward(long[,,] inputIds, bool[,] audioMask, bool[,,,] attnMask, long[,] posIds)
    {
        int B = inputIds.GetLength(0);
        int S = inputIds.GetLength(2);

        // 安全截断输入，防止 /audio_embeddings/Gather 越界
        for (int b = 0; b < B; b++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                for (int s = 0; s < S; s++)
                    inputIds[b, cb, s] = Math.Clamp(inputIds[b, cb, s], 0, 8199);

        var tIds = new DenseTensor<long>(Flatten3D(inputIds), new[] { B, NUM_CODEBOOKS, S });
        var tAudio = new DenseTensor<bool>(FlattenBool2D(audioMask), new[] { B, S });
        var tAttn = new DenseTensor<bool>(FlattenBool4D(attnMask), new[] { B, 1, S, S });
        var tPos = new DenseTensor<long>(Flatten2D(posIds), new[] { B, S });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", tIds),
            NamedOnnxValue.CreateFromTensor("audio_mask", tAudio),
            NamedOnnxValue.CreateFromTensor("attention_mask", tAttn),
            NamedOnnxValue.CreateFromTensor("position_ids", tPos),
        };

        using var results = _session.Run(inputs);
        var logitsTensor = results[0].AsTensor<float>();

        // 🔑 动态获取实际词表维度（避免硬编码 1025/8200 错位）
        int vocabSize = (int)logitsTensor.Dimensions[3];
        int total = B * NUM_CODEBOOKS * S * vocabSize;
        var flat = new float[total];
        int idx = 0;
        foreach (var v in logitsTensor) flat[idx++] = v;
        return flat;
    }

    // ── 语音克隆生成入口 ──────────────────────────────────
    public long[,] Generate(int[] textTokenIds, long[,] refCodes = null)
    {
        int T_ref = refCodes != null ? refCodes.GetLength(1) : 0;
        int prefixLen = 1 + T_ref + 1; // BOS + ref + BOS(生成起点)

        var prefix = new long[1, NUM_CODEBOOKS, prefixLen];
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++) prefix[0, cb, 0] = AUDIO_BOS;
        for (int t = 0; t < T_ref; t++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                prefix[0, cb, 1 + t] = Math.Clamp(refCodes[cb, t], 0, 8199);
        int genStart = 1 + T_ref;
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++) prefix[0, cb, genStart] = AUDIO_BOS;

        var audioMask = new bool[1, prefixLen];
        for (int i = 0; i < prefixLen; i++) audioMask[0, i] = true;

        var generatedCodes = new List<long[]>();
        var currentIds = prefix;
        int curLen = prefixLen;

        for (int step = 0; step < MaxNewTokens; step++)
        {
            // 🔍 每 20 步打印进度，确认未卡死
            if (step % 20 == 0) Debug.Log($"[OmniVoiceLM] 生成进度: {step}/{MaxNewTokens} (S={curLen})");

            var attnMask = BuildCausalMask(1, curLen);
            var posIds = BuildPositionIds(1, curLen);

            var condUncondIds = DoubleForCFG(currentIds, curLen);
            var condUncondAudio = DoubleForCFG(audioMask, curLen);
            var condUncondAttn = DoubleForCFG(attnMask);
            var condUncondPos = DoubleForCFG(posIds, curLen);

            var logits = LMForward(condUncondIds, condUncondAudio, condUncondAttn, condUncondPos);
            float[] sampledCodes = SampleWithCFG(logits, curLen, GuidanceScale);

            // 打印首步采样值，辅助调试 EOS
            if (step == 0) Debug.Log($"[OmniVoiceLM] 首帧采样 Codebook0: {(long)sampledCodes[0]} (EOS={EOS_TOKEN})");

            // 安全终止：首帧跳过 EOS 判断，后续匹配则 break
            if (step > 0 && (long)sampledCodes[0] == EOS_TOKEN)
            {
                Debug.Log($"[OmniVoiceLM] 触发 EOS，提前终止于 step={step}");
                break;
            }

            // 🔒 强制安全截断采样值，防止下一轮输入越界
            var stepCodes = new long[NUM_CODEBOOKS];
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                stepCodes[cb] = Math.Clamp((long)sampledCodes[cb], 0, 8199);
            generatedCodes.Add(stepCodes);

            currentIds = AppendToken(currentIds, stepCodes, curLen);
            audioMask = AppendAudioMask(audioMask, curLen);
            curLen++;
        }

        if (generatedCodes.Count == 0)
        {
            Debug.LogWarning("[OmniVoiceLM] 未生成任何 token");
            return new long[NUM_CODEBOOKS, 0];
        }

        int T_gen = generatedCodes.Count;
        var output = new long[NUM_CODEBOOKS, T_gen];
        for (int t = 0; t < T_gen; t++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                output[cb, t] = generatedCodes[t][cb];

        Debug.Log($"[OmniVoiceLM] ✅ 完成: 生成 {T_gen} 帧 ({T_gen * 0.04f:F1}s)");
        return output;
    }

    // ── CFG 采样 ──────────────────────────────────────────
    float[] SampleWithCFG(float[] logits, int S, float guidanceScale)
    {
        var result = new float[NUM_CODEBOOKS];
        int vocabSize = logits.Length / (2 * NUM_CODEBOOKS * S); // 动态计算
        int stride_B = NUM_CODEBOOKS * S * vocabSize;
        int stride_CB = S * vocabSize;
        int t = S - 1;

        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
        {
            var cfgLogits = new float[vocabSize];
            for (int v = 0; v < vocabSize; v++)
            {
                int idxCond = 0 * stride_B + cb * stride_CB + t * vocabSize + v;
                int idxUncond = 1 * stride_B + cb * stride_CB + t * vocabSize + v;
                float cond = logits[idxCond];
                float uncond = logits[idxUncond];
                cfgLogits[v] = uncond + guidanceScale * (cond - uncond);
            }
            result[cb] = SampleTopK(cfgLogits, Temperature, TopK);
        }
        return result;
    }

    float SampleTopK(float[] logits, float temp, int k)
    {
        if (Math.Abs(temp - 1f) > 1e-6f)
            for (int i = 0; i < logits.Length; i++) logits[i] /= temp;

        var indexed = new List<(float val, int idx)>(logits.Length);
        for (int i = 0; i < logits.Length; i++) indexed.Add((logits[i], i));
        indexed.Sort((a, b) => b.val.CompareTo(a.val));
        indexed = indexed.GetRange(0, Math.Min(k, indexed.Count));

        float maxV = indexed[0].val;
        float sum = 0f;
        var probs = new float[indexed.Count];
        for (int i = 0; i < indexed.Count; i++) { probs[i] = (float)Math.Exp(indexed[i].val - maxV); sum += probs[i]; }
        for (int i = 0; i < probs.Length; i++) probs[i] /= sum;

        double r = _rng.NextDouble();
        double cum = 0;
        for (int i = 0; i < probs.Length - 1; i++) { cum += probs[i]; if (r < cum) return indexed[i].idx; }
        return indexed[^1].idx;
    }

    // ── 工具方法 ──────────────────────────────────────────
    bool[,,,] BuildCausalMask(int B, int S)
    {
        var m = new bool[B, 1, S, S];
        for (int b = 0; b < B; b++)
            for (int i = 0; i < S; i++)
                for (int j = 0; j <= i; j++) m[b, 0, i, j] = true;
        return m;
    }

    long[,] BuildPositionIds(int B, int S)
    {
        var p = new long[B, S];
        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++) p[b, s] = s;
        return p;
    }

    long[,,] DoubleForCFG(long[,,] ids, int S)
    {
        var d = new long[2, NUM_CODEBOOKS, S];
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            for (int s = 0; s < S; s++) { d[0, cb, s] = ids[0, cb, s]; d[1, cb, s] = PAD_TOKEN; }
        return d;
    }

    bool[,] DoubleForCFG(bool[,] mask, int S)
    {
        var d = new bool[2, S];
        for (int s = 0; s < S; s++) { d[0, s] = mask[0, s]; d[1, s] = true; }
        return d;
    }

    bool[,,,] DoubleForCFG(bool[,,,] attn)
    {
        int S = attn.GetLength(2);
        var d = new bool[2, 1, S, S];
        for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++) { d[0, 0, i, j] = attn[0, 0, i, j]; d[1, 0, i, j] = true; }
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

    static long[] Flatten3D(long[,,] a) { var r = new long[a.Length]; Buffer.BlockCopy(a, 0, r, 0, a.Length * 8); return r; }
    static long[] Flatten2D(long[,] a) { var r = new long[a.Length]; Buffer.BlockCopy(a, 0, r, 0, a.Length * 8); return r; }
    static bool[] FlattenBool2D(bool[,] a) { var r = new bool[a.Length]; int i = 0; foreach (var v in a) r[i++] = v; return r; }
    static bool[] FlattenBool4D(bool[,,,] a) { var r = new bool[a.Length]; int i = 0; foreach (var v in a) r[i++] = v; return r; }

    public void Dispose() => _session?.Dispose();
}
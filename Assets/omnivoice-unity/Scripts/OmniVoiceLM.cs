// ============================================================
// OmniVoiceLM.cs  — 对齐 Python omnivoice.py _generate_iterative
//
// 与 Python 对齐的关键修改：
//   1. CFG batch：uncond 分支仅保留生成区域，其余 PAD（匹配 Python）
//   2. CFG log_probs：
//      ★ 公式：log_softmax(c + scale*(c - u))，基准为 c_log_probs
//        （原先错误地使用 u + scale*(c-u)）
//   3. MASK_TOKEN 的 log_prob 设为 -inf（Python: log_probs[..., mask_id] = -inf）
//   4. Token 采样：ClassTemperature=0.0（greedy argmax，匹配 Python 默认）
//      ClassTemperature>0 时使用 top-k ratio(0.1) + Gumbel 采样
//   5. 调度追踪剩余 mask，最后一步 unmask 全部残余
//   6. PositionTemperature 替代 MaskTemperature（匹配 Python position_temperature）
//   7. LayerPenaltyFactor 默认 5.0（匹配 Python layer_penalty_factor）
// ============================================================

using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

public class OmniVoiceLM : IDisposable
{
    public const int NUM_CODEBOOKS = 8;
    public const int VOCAB_SIZE = 1025;  // 1024 audio codes + 1 mask token
    public const int MASK_TOKEN = 1024;
    public const int PAD_TOKEN = 0;

    InferenceSession _session;
    System.Random _rng;

    // ─── 生成参数（与 Python OmniVoiceGenerationConfig 对齐）─────────
    public int NumStep = 32;
    public float GuidanceScale = 2.0f;
    public float TShift = 0.1f;
    /// <summary>
    /// position_temperature: 位置选择的 Gumbel 温度。Python 默认 5.0。
    /// </summary>
    public float PositionTemperature = 5.0f;
    /// <summary>
    /// class_temperature: token 采样温度。0 = greedy argmax（Python 默认）；
    /// &gt;0 时先 top-k ratio(0.1) 过滤再 Gumbel 采样。
    /// </summary>
    public float ClassTemperature = 0.0f;
    /// <summary>
    /// layer_penalty_factor: 层惩罚系数，控制 codebook 从低到高逐层解 mask。
    /// Python 默认 5.0。
    /// </summary>
    public float LayerPenaltyFactor = 5.0f;

    public OmniVoiceLM(string modelPath, int seed = 42)
    {
        _rng = new System.Random(seed);
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        opts.InterOpNumThreads = 1;
        opts.IntraOpNumThreads = 0;

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

        Debug.Log($"[OmniVoiceLM] 开始扩散: T_text={T_text} T_ref={T_ref} T_gen={targetLen} S={S} " +
                  $"steps={NumStep} GS={GuidanceScale} LayerPenalty={LayerPenaltyFactor} " +
                  $"ClassTemp={ClassTemperature} PosTemp={PositionTemperature}");

        // ──────────────────────────────────────────────
        // 构建 cond 序列的 inputIds 和 audioMask
        // ──────────────────────────────────────────────
        var inputIds = new long[1, NUM_CODEBOOKS, S];
        var audioMask = new bool[1, S];

        // 文本区（audioMask = false）
        for (int s = 0; s < T_text; s++)
        {
            long tid = textTokenIds[s];
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = tid;
            audioMask[0, s] = false;
        }

        // 参考音频区（audioMask = true）
        for (int t = 0; t < T_ref; t++)
        {
            int s = T_text + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = Math.Clamp(refCodes[cb, t], 0, MASK_TOKEN - 1);
            audioMask[0, s] = true;
        }

        // 待生成区（全部 MASK；audioMask = true）
        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = MASK_TOKEN;
            audioMask[0, s] = true;
        }

        // ── 流量调度（time-shift cosine schedule）─────────────────
        double tau = TShift;
        double N = NumStep;
        var r = new double[NumStep + 1];
        for (int n = 0; n <= NumStep; n++)
        {
            double u = (double)n / N;
            r[n] = tau * u / (1.0 + (tau - 1.0) * u);
        }

        // ── 计算每步的 schedule（与 Python schedules 对齐）──────────
        int totalMasks = targetLen * NUM_CODEBOOKS;
        int remMasks = totalMasks;

        // ── 主扩散循环 ─────────────────────────────────────────────
        for (int step = 0; step < NumStep; step++)
        {
            int kNew;
            if (step == NumStep - 1)
            {
                // 最后一步：解所有剩余 mask（Python: k = rem）
                kNew = remMasks;
            }
            else
            {
                double kRatio = r[step + 1] - r[step];
                kNew = (int)Math.Ceiling(kRatio * totalMasks);
                kNew = Math.Min(kNew, remMasks);
            }

            if (kNew <= 0) continue;

            if (step % 8 == 0)
                Debug.Log($"[OmniVoiceLM] step {step}/{NumStep}  kNew={kNew}  rem={remMasks}");

            float[] logProbs = LMForwardWithCFG(
                inputIds, audioMask, S, genStart, T_ref, T_text, targetLen);

            if (IsCorrupted(logProbs))
            {
                Debug.LogError($"[OmniVoiceLM] 步 {step} 检测到 NaN/Inf，尝试恢复...");
                PositionTemperature = Mathf.Max(0.1f, PositionTemperature * 0.5f);
                logProbs = LMForwardWithCFG(
                    inputIds, audioMask, S, genStart, T_ref, T_text, targetLen);
                if (IsCorrupted(logProbs))
                {
                    Debug.LogError("[OmniVoiceLM] 恢复失败，终止生成");
                    break;
                }
            }

            int unmasked = DiffusionStep(
                inputIds, logProbs, genStart, targetLen, S, kNew);
            remMasks -= unmasked;
        }

        // ── 最终强制解 mask ────────────────────────────────────────
        FinalUnmaskAll(inputIds, audioMask, S, genStart, targetLen, T_ref, T_text);

        // ── 提取结果 ───────────────────────────────────────────────
        var result = new long[NUM_CODEBOOKS, targetLen];
        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                long v = inputIds[0, cb, s];
                result[cb, t] = (v == MASK_TOKEN) ? 0L : Math.Clamp(v, 0L, (long)(MASK_TOKEN - 1));
            }
        }

        float durSec = targetLen * 960f / 24000f;
        Debug.Log($"[OmniVoiceLM] 完成: {targetLen} 帧 = {durSec:F1}s");
        return result;
    }

    // ═══════════════════════════════════════════════════════════════
    // CFG 前向传播 — 与 Python _generate_iterative 对齐
    //
    // Python 中 cond 和 uncond 分支序列布局不同：
    //   cond:   [style | text | ref_audio | gen_tokens]
    //   uncond: [gen_tokens | PAD]   （仅生成区域，其余填充）
    //
    // Logits 提取位置也不同：
    //   cond logits for gen: [b=0, C, c_len-t_len : c_len, V]
    //   uncond logits for gen: [b=1, C, 0 : t_len, V]
    // ═══════════════════════════════════════════════════════════════
    float[] LMForwardWithCFG(
        long[,,] inputIds, bool[,] audioMask,
        int S, int genStart, int T_ref, int T_text, int targetLen)
    {
        if (GuidanceScale > 0f)
        {
            // ── 构建 CFG batch ──────────────────────────────────
            var (batchIds, batchAudio, batchAttn) = BuildCFGBatch(
                inputIds, audioMask, genStart, S, T_text, T_ref, targetLen);
            var posIds = BuildPositionIds(2, S);

            float[] rawLogits = LMForward(
                batchIds, batchAudio, batchAttn, posIds, batchSize: 2, S: S);

            // ── 从 cond 和 uncond 分支分别提取生成区域的 logits ──
            int strideB = NUM_CODEBOOKS * S * VOCAB_SIZE;
            int strideCB = S * VOCAB_SIZE;
            var result = new float[NUM_CODEBOOKS * S * VOCAB_SIZE];

            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                for (int t = 0; t < targetLen; t++)
                {
                    int sCond = genStart + t;    // cond 中生成区的位置
                    int sUncond = t;             // uncond 中生成区的位置（从 0 开始）

                    var condLogits = new float[VOCAB_SIZE];
                    var uncondLogits = new float[VOCAB_SIZE];
                    int condBase = cb * strideCB + sCond * VOCAB_SIZE;
                    int uncondBase = cb * strideCB + sUncond * VOCAB_SIZE;

                    for (int v = 0; v < VOCAB_SIZE; v++)
                    {
                        condLogits[v] = rawLogits[0 * strideB + condBase + v];
                        uncondLogits[v] = rawLogits[1 * strideB + uncondBase + v];
                    }

                    // Python: c_log_probs = log_softmax(c_logits)
                    float[] condLSM = LogSoftmax(condLogits);
                    // Python: u_log_probs = log_softmax(u_logits)
                    float[] uncondLSM = LogSoftmax(uncondLogits);

                    // Python: log_softmax(c_log_probs + guidance_scale * (c_log_probs - u_log_probs))
                    // ★ 修复：基准是 c_log_probs，不是 u_log_probs
                    var cfgValues = new float[VOCAB_SIZE];
                    for (int v = 0; v < VOCAB_SIZE; v++)
                        cfgValues[v] = condLSM[v] + GuidanceScale * (condLSM[v] - uncondLSM[v]);
                    float[] finalLSM = LogSoftmax(cfgValues);

                    // Python: log_probs[..., audio_mask_id] = -inf
                    finalLSM[MASK_TOKEN] = float.NegativeInfinity;

                    // 存入结果（对应 cond 的生成区位置，方便 DiffusionStep 索引）
                    int resultOff = cb * strideCB + sCond * VOCAB_SIZE;
                    for (int v = 0; v < VOCAB_SIZE; v++)
                        result[resultOff + v] = finalLSM[v];
                }
            }

            return result;
        }
        else
        {
            // 无 CFG：直接 log_softmax
            var attnMask = BuildFullMask(1, S);
            var posIds = BuildPositionIds(1, S);
            float[] rawLogits = LMForward(
                inputIds, audioMask, attnMask, posIds, batchSize: 1, S: S);

            int strideCB = S * VOCAB_SIZE;
            var result = new float[NUM_CODEBOOKS * S * VOCAB_SIZE];
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                for (int s = 0; s < S; s++)
                {
                    var logits = new float[VOCAB_SIZE];
                    int baseOff = cb * strideCB + s * VOCAB_SIZE;
                    for (int v = 0; v < VOCAB_SIZE; v++)
                        logits[v] = rawLogits[baseOff + v];

                    float[] lsm = LogSoftmax(logits);
                    lsm[MASK_TOKEN] = float.NegativeInfinity;

                    for (int v = 0; v < VOCAB_SIZE; v++)
                        result[baseOff + v] = lsm[v];
                }

            return result;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // CFG Batch 构建 — 与 Python _generate_iterative 对齐
    //
    //   cond   (b=0): [text | ref_audio | gen_tokens]  — 完整条件序列
    //   uncond (b=1): [gen_tokens | PAD]               — 仅生成区域在前
    //
    // Python 原始逻辑：
    //   batch_input_ids[B+i, :, :u_len] = inp["input_ids"][..., -u_len:]
    //   batch_audio_mask[B+i, :u_len] = inp["audio_mask"][..., -u_len:]
    //   batch_attention_mask[B+i, :, :u_len, :u_len] = True
    //   pad_diag: batch_attention_mask[B+i, :, pad_diag, pad_diag] = True
    // ═══════════════════════════════════════════════════════════════
    static (long[,,] ids, bool[,] audio, bool[,,,] attn) BuildCFGBatch(
        long[,,] srcIds, bool[,] srcAudio,
        int genStart, int S, int T_text, int T_ref, int targetLen)
    {
        var ids = new long[2, NUM_CODEBOOKS, S];
        var audio = new bool[2, S];
        var attn = new bool[2, 1, S, S];

        // ── cond 分支（b=0）：原样复制 ──────────────────────────
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            for (int s = 0; s < S; s++)
            {
                ids[0, cb, s] = srcIds[0, cb, s];
                audio[0, s] = srcAudio[0, s];
            }

        // cond 的 attention：全序列可互相 attend
        for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++)
                attn[0, 0, i, j] = true;

        // ── uncond 分支（b=1）：生成区在前 [0, targetLen)，其余 PAD ──
        // Python: batch_input_ids[B+i, :, :u_len] = inp["input_ids"][..., -u_len:]
        //         batch_audio_mask[B+i, :u_len] = inp["audio_mask"][..., -u_len:]
        // audio mask 与 codebook 无关，先统一设置
        for (int t = 0; t < targetLen; t++) audio[1, t] = true;
        for (int s = targetLen; s < S; s++) audio[1, s] = false;

        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
        {
            // 生成区：从 cond 的 [genStart, S) 拷贝到 uncond 的 [0, targetLen)
            for (int t = 0; t < targetLen; t++)
                ids[1, cb, t] = srcIds[0, cb, genStart + t];
            // 填充区：MASK_TOKEN
            for (int s = targetLen; s < S; s++)
                ids[1, cb, s] = MASK_TOKEN;
        }

        // uncond 的 attention mask：
        // [0, targetLen) 之间可互相 attend
        for (int i = 0; i < targetLen; i++)
            for (int j = 0; j < targetLen; j++)
                attn[1, 0, i, j] = true;

        // [targetLen, S) 仅对角 attend（Python: pad_diag）
        for (int s = targetLen; s < S; s++)
            attn[1, 0, s, s] = true;

        return (ids, audio, attn);
    }

    // ═══════════════════════════════════════════════════════════════
    // DiffusionStep — 与 Python _generate_iterative 对齐
    //
    // Python 流程：
    //   1. _predict_tokens_with_scoring(c_logits, u_logits, config)
    //      → pred_tokens, confidence_scores
    //   2. scores = confidence_scores - layer_ids * layer_penalty_factor
    //   3. scores = _gumbel_sample(scores, position_temperature)
    //   4. scores.masked_fill_(sample_tokens != MASK, -inf)
    //   5. topk(scores.flatten(), k) → 选中位置
    //   6. 填入 pred_tokens 到选中位置
    // ═══════════════════════════════════════════════════════════════
    int DiffusionStep(long[,,] inputIds, float[] logProbs,
                      int genStart, int targetLen, int S, int kNew)
    {
        int strideCB = S * VOCAB_SIZE;

        // 1. 预测 token 和置信度（对应 Python _predict_tokens_with_scoring）
        var predTokens = new long[targetLen, NUM_CODEBOOKS];
        var scores = new float[targetLen, NUM_CODEBOOKS];

        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                int baseOff = cb * strideCB + s * VOCAB_SIZE;

                // Token 预测
                if (ClassTemperature > 0.0f)
                {
                    // Python: _filter_top_k(log_probs, ratio=0.1)
                    //         + _gumbel_sample(filtered, class_temperature)
                    //         + argmax
                    predTokens[t, cb] = SampleTokenTopKRatio(
                        logProbs, baseOff, 0.1f, ClassTemperature);
                }
                else
                {
                    // Python: pred_tokens = log_probs.argmax(dim=-1)
                    predTokens[t, cb] = ArgmaxToken(logProbs, baseOff);
                }

                // Python: confidence_scores = log_probs.max(dim=-1)[0]
                scores[t, cb] = float.NegativeInfinity;
                for (int v = 0; v < VOCAB_SIZE; v++)
                {
                    float lp = logProbs[baseOff + v];
                    if (!float.IsNaN(lp) && !float.IsInfinity(lp) && lp > scores[t, cb])
                        scores[t, cb] = lp;
                }
            }
        }

        // 2. 层惩罚：scores -= layer_ids * layer_penalty_factor
        //    Python: scores = scores - (layer_ids * layer_penalty_factor)
        for (int t = 0; t < targetLen; t++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                scores[t, cb] -= cb * LayerPenaltyFactor;

        // 3. Position temperature（Gumbel 噪声）
        //    Python: if position_temperature > 0: scores = _gumbel_sample(scores, position_temperature)
        if (PositionTemperature > 0.0f)
        {
            for (int t = 0; t < targetLen; t++)
                for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                {
                    double u = Math.Max(1e-10, _rng.NextDouble());
                    double gumbel = -Math.Log(-Math.Log(u));
                    scores[t, cb] = (float)(scores[t, cb] / PositionTemperature + gumbel);
                }
        }

        // 4. 已解 mask 的位置得分设为 -inf
        //    Python: scores.masked_fill_(sample_tokens != MASK, -inf)
        int totalMasked = 0;
        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                if (inputIds[0, cb, s] != MASK_TOKEN)
                    scores[t, cb] = float.NegativeInfinity;
                else
                    totalMasked++;
            }
        }

        if (totalMasked == 0) return 0;
        kNew = Math.Min(kNew, totalMasked);

        // 5. Top-k 位置选择
        //    Python: _, topk_idx = torch.topk(scores.flatten(), k)
        var allScores = new (int t, int cb, float score)[totalMasked];
        int idx = 0;
        for (int t = 0; t < targetLen; t++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                int s = genStart + t;
                if (inputIds[0, cb, s] == MASK_TOKEN)
                    allScores[idx++] = (t, cb, scores[t, cb]);
            }

        Array.Sort(allScores, (a, b) => b.score.CompareTo(a.score));

        // 6. 对选中的位置填入预测 token
        //    Python: flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
        int unmasked = 0;
        for (int i = 0; i < kNew && i < allScores.Length; i++)
        {
            var (t, cb, _) = allScores[i];
            int s = genStart + t;
            inputIds[0, cb, s] = predTokens[t, cb];
            unmasked++;
        }

        return unmasked;
    }

    // ═══════════════════════════════════════════════════════════════
    // Token 采样 — 与 Python _predict_tokens_with_scoring 对齐
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// Greedy argmax（class_temperature == 0 时使用）
    /// Python: pred_tokens = log_probs.argmax(dim=-1)
    /// </summary>
    long ArgmaxToken(float[] logProbs, int baseOff)
    {
        float best = float.NegativeInfinity;
        long tok = 0;
        for (int v = 0; v < VOCAB_SIZE; v++)
        {
            float lp = logProbs[baseOff + v];
            if (float.IsNaN(lp) || float.IsInfinity(lp)) continue;
            if (lp > best) { best = lp; tok = v; }
        }
        return tok;
    }

    /// <summary>
    /// Top-k ratio 过滤 + Gumbel 采样（class_temperature &gt; 0 时使用）
    /// Python: _filter_top_k(log_probs, ratio=0.1)
    ///         + _gumbel_sample(filtered, class_temperature)
    ///         + .argmax(dim=-1)
    /// </summary>
    long SampleTokenTopKRatio(float[] logProbs, int baseOff, float ratio, float temperature)
    {
        int k = (int)Math.Ceiling(ratio * VOCAB_SIZE);

        // 找 top-k 的 log_probs
        var entries = new (float score, int idx)[VOCAB_SIZE];
        for (int v = 0; v < VOCAB_SIZE; v++)
            entries[v] = (logProbs[baseOff + v], v);

        Array.Sort(entries, (a, b) => b.score.CompareTo(a.score));

        // Python _filter_top_k: 创建过滤后的 logits（非 top-k 设为 -inf）
        var filtered = new float[VOCAB_SIZE];
        for (int v = 0; v < VOCAB_SIZE; v++)
            filtered[v] = float.NegativeInfinity;
        for (int i = 0; i < k && i < entries.Length; i++)
            filtered[entries[i].idx] = entries[i].score;

        // Python _gumbel_sample: scaled_logits = logits / T + Gumbel noise
        for (int v = 0; v < VOCAB_SIZE; v++)
        {
            if (float.IsNegativeInfinity(filtered[v])) continue;
            double u = Math.Max(1e-10, _rng.NextDouble());
            double gumbel = -Math.Log(-Math.Log(u));
            filtered[v] = (float)(filtered[v] / temperature + gumbel);
        }

        // argmax on Gumbel-perturbed logits
        float best = float.NegativeInfinity;
        long tok = 0;
        for (int v = 0; v < VOCAB_SIZE; v++)
        {
            if (filtered[v] > best) { best = filtered[v]; tok = v; }
        }
        return tok;
    }

    void FinalUnmaskAll(long[,,] inputIds, bool[,] audioMask,
                        int S, int genStart, int targetLen,
                        int T_ref, int T_text)
    {
        int maskCount = 0;
        for (int t = 0; t < targetLen; t++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                if (inputIds[0, cb, genStart + t] == MASK_TOKEN)
                    maskCount++;

        if (maskCount == 0) return;
        Debug.Log($"[OmniVoiceLM] 最终强制解 mask: 残余 {maskCount} 个位置");

        float[] logProbs = LMForwardWithCFG(
            inputIds, audioMask, S, genStart, T_ref, T_text, targetLen);

        int strideCB = S * VOCAB_SIZE;
        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                if (inputIds[0, cb, s] != MASK_TOKEN) continue;
                int baseOff = cb * strideCB + s * VOCAB_SIZE;
                inputIds[0, cb, s] = ArgmaxToken(logProbs, baseOff);
            }
        }
    }

    float[] LMForward(long[,,] inputIds, bool[,] audioMask,
                      bool[,,,] attnMask, long[,] posIds,
                      int batchSize, int S)
    {
        var tIds = new DenseTensor<long>(Flatten3D(inputIds), new[] { batchSize, NUM_CODEBOOKS, S });
        var tAudio = new DenseTensor<bool>(FlattenBool2D(audioMask), new[] { batchSize, S });
        var tAttn = new DenseTensor<bool>(FlattenBool4D(attnMask), new[] { batchSize, 1, S, S });
        var tPos = new DenseTensor<long>(Flatten2D(posIds), new[] { batchSize, S });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids",      tIds),
            NamedOnnxValue.CreateFromTensor("audio_mask",     tAudio),
            NamedOnnxValue.CreateFromTensor("attention_mask", tAttn),
            NamedOnnxValue.CreateFromTensor("position_ids",   tPos),
        };

        using var results = _session.Run(inputs);
        var logitsTensor = results[0].AsTensor<float>();
        int total = batchSize * NUM_CODEBOOKS * S * VOCAB_SIZE;
        var flat = new float[total];
        int idx = 0;
        foreach (var v in logitsTensor) flat[idx++] = v;
        return flat;
    }

    // ── 辅助 ──────────────────────────────────────────────────────

    bool IsCorrupted(float[] arr)
    {
        int bad = 0;
        for (int i = 0; i < arr.Length; i++)
            if (float.IsNaN(arr[i]) || float.IsInfinity(arr[i])) bad++;
        return bad > arr.Length / 100;
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
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > maxV) maxV = logits[i];

        if (float.IsInfinity(maxV) || float.IsNaN(maxV))
        {
            var fb = new float[logits.Length];
            for (int i = 0; i < fb.Length; i++) fb[i] = float.NegativeInfinity;
            return fb;
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
        int i = 0; foreach (var v in a) r[i++] = v;
        return r;
    }

    static bool[] FlattenBool4D(bool[,,,] a)
    {
        var r = new bool[a.Length];
        int i = 0; foreach (var v in a) r[i++] = v;
        return r;
    }

    public void Dispose() => _session?.Dispose();
}
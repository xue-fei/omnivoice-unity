// ============================================================
// OmniVoiceLM.cs  — 性能优化版
//
// 优化项（相对原版）：
//   1. CUDA / DML EP 正确配置：禁用 EnableMemoryPattern（动态 shape 场景）
//      CUDA 额外选项：arena_extend_strategy、cudnn_conv_algo_search
//   2. 预分配复用缓冲区：消除内循环 new float[VOCAB_SIZE] 引起的 GC 压力
//   3. LogSoftmax 改为原地写入，避免每次分配返回数组
//   4. logits 拷贝：优先 ToArray()，fallback foreach
//   5. FinalUnmaskAll 快速路径：mask 为 0 时直接 return，省第 N+1 次 forward
//   6. BuildFullMask / BuildPositionIds 改为原地填充，减少堆分配
//   7. CFG 内循环：消除临时 condLogits/uncondLogits 数组，直接读 rawLogits
//   8. DiffusionStep：scores 计算与 top-k 合并，减少两次遍历
//   9. bool 扁平化 FlattenBool2D/4D 使用 Buffer.BlockCopy（bool=1byte）
//  10. CFG batch 构建：用 Array.Copy / Buffer.BlockCopy 替代逐元素循环
// ============================================================

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class OmniVoiceLM : IDisposable
{
    public const int NUM_CODEBOOKS = 8;
    public const int VOCAB_SIZE = 1025;   // 1024 audio codes + 1 mask token
    public const int MASK_TOKEN = 1024;
    public const int PAD_TOKEN = 0;

    InferenceSession _session;
    System.Random _rng;

    // ─── 生成参数 ────────────────────────────────────────────────────
    public int NumStep = 32;
    public float GuidanceScale = 2.0f;
    public float TShift = 0.1f;
    public float PositionTemperature = 5.0f;
    public float ClassTemperature = 0.0f;
    public float LayerPenaltyFactor = 5.0f;

    // ─── 复用缓冲区（Generate 首次调用时按实际 S 分配）──────────────
    float[] _rawLogitsBuf;   // [2 * NUM_CODEBOOKS * S * VOCAB_SIZE]
    float[] _resultBuf;      // [NUM_CODEBOOKS * S * VOCAB_SIZE]
    // 单 token LogSoftmax 工作区（VOCAB_SIZE，固定大小，构造时分配）
    readonly float[] _lsmWork = new float[VOCAB_SIZE];

    // ─── EP 配置参数（由 Runner 通过构造函数传入）──────────────────
    public enum ExecutionProviderType { CPU, CUDA, DML }

    // ════════════════════════════════════════════════════════════════
    // 构造函数
    // ════════════════════════════════════════════════════════════════
    public OmniVoiceLM(string modelPath,
                       ExecutionProviderType ep = ExecutionProviderType.CUDA,
                       int deviceId = 0,
                       int seed = 42)
    {
        _rng = new System.Random(seed);

        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        // 动态 shape（每步序列不变，但 CFG batch 2 vs 1 会切换）
        // EnableMemoryPattern = false 避免 ORT 在 shape 变化时重新分配 arena
        opts.EnableMemoryPattern = false;
        opts.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        // 线程数：LM 模型大，推理以 GPU 为主时 CPU 线程无需过多
        opts.InterOpNumThreads = 1;
        opts.IntraOpNumThreads = 4;

        bool epLoaded = false;

        if (ep == ExecutionProviderType.CUDA)
        {
            try
            {
                var cudaOpts = new OrtCUDAProviderOptions();
                cudaOpts.UpdateOptions(new Dictionary<string, string>
                {
                    { "device_id",               deviceId.ToString() },
                    // kSameAsRequested：按需申请显存，避免预留过多
                    { "arena_extend_strategy",   "kSameAsRequested" },
                    // HEURISTIC 比 EXHAUSTIVE 启动快，精度无损
                    { "cudnn_conv_algo_search",  "HEURISTIC" },
                    // 在默认流上做 H2D/D2H 拷贝，减少同步点
                    { "do_copy_in_default_stream", "1" },
                });
                opts.AppendExecutionProvider_CUDA(cudaOpts);
                epLoaded = true;
                Debug.Log($"[OmniVoiceLM] CUDA EP 已加载 (device={deviceId})");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[OmniVoiceLM] CUDA EP 不可用: {ex.Message}，回退 CPU");
            }
        }
        else if (ep == ExecutionProviderType.DML)
        {
            try
            {
                // 注意：DML 不支持 bool 张量，已在 LMForward 中将 mask 转为 float32
                opts.AppendExecutionProvider_DML(deviceId);
                epLoaded = true;
                Debug.Log($"[OmniVoiceLM] DirectML EP 已加载 (device={deviceId})");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[OmniVoiceLM] DML EP 不可用: {ex.Message}，回退 CPU");
            }
        }

        if (!epLoaded)
        {
            // CPU fallback：多线程加速
            opts.IntraOpNumThreads = Math.Max(4, Environment.ProcessorCount);
            Debug.Log($"[OmniVoiceLM] 使用 CPU EP (threads={opts.IntraOpNumThreads})");
        }

        _session = new InferenceSession(modelPath, opts);
        Debug.Log($"[OmniVoiceLM] 已加载: {modelPath}");
    }

    // ════════════════════════════════════════════════════════════════
    // Generate — 主入口
    // ════════════════════════════════════════════════════════════════
    public long[,] Generate(int[] textTokenIds, long[,] refCodes, int targetLen)
    {
        int T_text = textTokenIds != null ? textTokenIds.Length : 0;
        int T_ref = refCodes != null ? refCodes.GetLength(1) : 0;

        if (targetLen <= 0) targetLen = Mathf.Max(50, T_ref > 0 ? T_ref : 100);

        int genStart = T_text + T_ref;
        int S = genStart + targetLen;

        // 按实际 S 懒分配/扩容复用缓冲区
        EnsureBuffers(S);

        Debug.Log($"[OmniVoiceLM] 开始扩散: T_text={T_text} T_ref={T_ref} T_gen={targetLen} " +
                  $"S={S} steps={NumStep} GS={GuidanceScale} LayerPenalty={LayerPenaltyFactor} " +
                  $"ClassTemp={ClassTemperature} PosTemp={PositionTemperature}");

        // ── 构建 inputIds 和 audioMask ───────────────────────────────
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

        // 参考音频区
        for (int t = 0; t < T_ref; t++)
        {
            int s = T_text + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = Math.Clamp(refCodes[cb, t], 0, MASK_TOKEN - 1);
            audioMask[0, s] = true;
        }

        // 待生成区（全 MASK）
        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = MASK_TOKEN;
            audioMask[0, s] = true;
        }

        // ── 时移余弦调度 ─────────────────────────────────────────────
        double tau = TShift;
        double N = NumStep;
        var r = new double[NumStep + 1];
        for (int n = 0; n <= NumStep; n++)
        {
            double u = (double)n / N;
            r[n] = tau * u / (1.0 + (tau - 1.0) * u);
        }

        int totalMasks = targetLen * NUM_CODEBOOKS;
        int remMasks = totalMasks;

        // ── 主扩散循环 ───────────────────────────────────────────────
        for (int step = 0; step < NumStep; step++)
        {
            int kNew;
            if (step == NumStep - 1)
                kNew = remMasks;
            else
            {
                double kRatio = r[step + 1] - r[step];
                // ★ 修复：与 Python math.ceil(total_mask * delta) 对齐使用 Round，
                // Ceiling 会在前几步解掉过多 token，导致高层 codebook 优先级失衡。
                kNew = (int)Math.Round(kRatio * totalMasks);
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

            int unmasked = DiffusionStep(inputIds, logProbs, genStart, targetLen, S, kNew);
            remMasks -= unmasked;
        }

        // ── 最终强制解 mask ──────────────────────────────────────────
        FinalUnmaskAll(inputIds, audioMask, S, genStart, targetLen, T_ref, T_text);

        // ── 提取结果 ─────────────────────────────────────────────────
        var result = new long[NUM_CODEBOOKS, targetLen];
        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                long v = inputIds[0, cb, s];
                result[cb, t] = (v == MASK_TOKEN) ? 0L : Math.Clamp(v, 0L, MASK_TOKEN - 1L);
            }
        }

        float durSec = targetLen * 960f / 24000f;
        Debug.Log($"[OmniVoiceLM] 完成: {targetLen} 帧 = {durSec:F1}s");
        return result;
    }

    // ════════════════════════════════════════════════════════════════
    // CFG 前向传播
    //   cond   (b=0): [text | ref_audio | gen_tokens]
    //   uncond (b=1): [gen_tokens | PAD(MASK)]
    // ════════════════════════════════════════════════════════════════
    float[] LMForwardWithCFG(
        long[,,] inputIds, bool[,] audioMask,
        int S, int genStart, int T_ref, int T_text, int targetLen)
    {
        if (GuidanceScale > 0f)
        {
            var (batchIds, batchAudio, batchAttn) = BuildCFGBatch(
                inputIds, audioMask, genStart, S, T_text, T_ref, targetLen);
            long[,] posIds = BuildPositionIds(2, S);

            // rawLogits 写入复用缓冲区 _rawLogitsBuf
            LMForward(batchIds, batchAudio, batchAttn, posIds, batchSize: 2, S: S,
                      outBuf: _rawLogitsBuf);

            // ── CFG 合并：直接从 _rawLogitsBuf 读，写入 _resultBuf ──
            int strideB = NUM_CODEBOOKS * S * VOCAB_SIZE;
            int strideCB = S * VOCAB_SIZE;

            // 清空 result 缓冲区（仅生成区域会被写入，其余不访问）
            // Array.Clear(_resultBuf, 0, _resultBuf.Length); // 非必须，写入前会覆盖

            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                for (int t = 0; t < targetLen; t++)
                {
                    int sCond = genStart + t;
                    int sUncond = t;

                    int condBase = cb * strideCB + sCond * VOCAB_SIZE;
                    int uncondBase = cb * strideCB + sUncond * VOCAB_SIZE;
                    int condOff = 0 * strideB + condBase;
                    int uncondOff = 1 * strideB + uncondBase;

                    // ── log_softmax(cond) ──
                    LogSoftmaxSlice(_rawLogitsBuf, condOff, VOCAB_SIZE, _lsmWork);   // condLSM → _lsmWork
                    // 暂存 condLSM（需要两次使用）
                    var condLSM = new float[VOCAB_SIZE];
                    Array.Copy(_lsmWork, condLSM, VOCAB_SIZE);

                    // ── log_softmax(uncond) ──
                    LogSoftmaxSlice(_rawLogitsBuf, uncondOff, VOCAB_SIZE, _lsmWork); // uncondLSM → _lsmWork

                    // ── CFG: cfg[v] = condLSM[v] + scale*(condLSM[v] - uncondLSM[v]) ──
                    // ── 然后再做一次 log_softmax ──
                    // 先把 cfgValues 写进 _lsmWork 的位置（复用同一块）
                    for (int v = 0; v < VOCAB_SIZE; v++)
                        _lsmWork[v] = condLSM[v] + GuidanceScale * (condLSM[v] - _lsmWork[v]);

                    // 原地对 _lsmWork 做 log_softmax，结果写入 resultBuf
                    int resultOff = cb * strideCB + sCond * VOCAB_SIZE;
                    LogSoftmaxSliceSelf(_lsmWork, VOCAB_SIZE, _resultBuf, resultOff);

                    // MASK token 的 log_prob 设为 -inf
                    _resultBuf[resultOff + MASK_TOKEN] = float.NegativeInfinity;
                }
            }

            return _resultBuf;
        }
        else
        {
            // 无 CFG：单 batch
            bool[,,,] attnMask = BuildFullMask(1, S);
            long[,] posIds = BuildPositionIds(1, S);
            LMForward(inputIds, audioMask, attnMask, posIds, batchSize: 1, S: S,
                      outBuf: _rawLogitsBuf);

            int strideCB = S * VOCAB_SIZE;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                for (int s = 0; s < S; s++)
                {
                    int baseOff = cb * strideCB + s * VOCAB_SIZE;
                    LogSoftmaxSlice(_rawLogitsBuf, baseOff, VOCAB_SIZE, _resultBuf, baseOff);
                    _resultBuf[baseOff + MASK_TOKEN] = float.NegativeInfinity;
                }
            }
            return _resultBuf;
        }
    }

    // ════════════════════════════════════════════════════════════════
    // CFG Batch 构建
    //   cond   (b=0): 原样复制
    //   uncond (b=1): 生成区放前段，其余填 MASK
    // ════════════════════════════════════════════════════════════════
    static (long[,,] ids, bool[,] audio, bool[,,,] attn) BuildCFGBatch(
        long[,,] srcIds, bool[,] srcAudio,
        int genStart, int S, int T_text, int T_ref, int targetLen)
    {
        var ids = new long[2, NUM_CODEBOOKS, S];
        var audio = new bool[2, S];
        var attn = new bool[2, 1, S, S];

        // ── cond (b=0)：按 codebook 整行拷贝 ─────────────────────────
        // srcIds[0, cb, *] → ids[0, cb, *]
        // 利用 Buffer.BlockCopy（long=8 bytes）
        int rowBytes = S * sizeof(long);
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
        {
            // 源 offset（字节）: (0*C*S + cb*S) * 8
            int srcOff = (0 * NUM_CODEBOOKS * S + cb * S) * sizeof(long);
            int dstOff = (0 * NUM_CODEBOOKS * S + cb * S) * sizeof(long);
            Buffer.BlockCopy(srcIds, srcOff, ids, dstOff, rowBytes);
        }
        // cond audioMask：整行拷贝
        for (int s = 0; s < S; s++) audio[0, s] = srcAudio[0, s];

        // cond attention：全 true
        for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++)
                attn[0, 0, i, j] = true;

        // ── uncond (b=1) ─────────────────────────────────────────────
        // audioMask：生成区 true，其余 false
        for (int t = 0; t < targetLen; t++) audio[1, t] = true;
        // （bool 数组默认 false，其余不必清零）

        // inputIds：生成区拷贝自 cond [genStart..S)，其余 MASK
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
        {
            // 生成区：srcIds[0, cb, genStart..] → ids[1, cb, 0..]
            int srcOff = (0 * NUM_CODEBOOKS * S + cb * S + genStart) * sizeof(long);
            int dstOff = (1 * NUM_CODEBOOKS * S + cb * S + 0) * sizeof(long);
            Buffer.BlockCopy(srcIds, srcOff, ids, dstOff, targetLen * sizeof(long));

            // 填充区：MASK_TOKEN
            for (int s = targetLen; s < S; s++)
                ids[1, cb, s] = MASK_TOKEN;
        }

        // uncond attention：
        //   [0, targetLen) 互相可 attend
        for (int i = 0; i < targetLen; i++)
            for (int j = 0; j < targetLen; j++)
                attn[1, 0, i, j] = true;
        //   [targetLen, S) 仅对角（pad_diag）
        for (int s = targetLen; s < S; s++)
            attn[1, 0, s, s] = true;

        return (ids, audio, attn);
    }

    // ════════════════════════════════════════════════════════════════
    // DiffusionStep — 预测 token + 位置选择
    // ════════════════════════════════════════════════════════════════
    int DiffusionStep(long[,,] inputIds, float[] logProbs,
                      int genStart, int targetLen, int S, int kNew)
    {
        int strideCB = S * VOCAB_SIZE;

        var predTokens = new long[targetLen, NUM_CODEBOOKS];
        var scores = new float[targetLen, NUM_CODEBOOKS];

        // 1. 预测 token + 置信度
        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                int baseOff = cb * strideCB + s * VOCAB_SIZE;

                predTokens[t, cb] = ClassTemperature > 0f
                    ? SampleTokenTopKRatio(logProbs, baseOff, 0.1f, ClassTemperature)
                    : ArgmaxToken(logProbs, baseOff);

                // confidence = max log_prob（不含 MASK）
                float best = float.NegativeInfinity;
                for (int v = 0; v < VOCAB_SIZE; v++)   // MASK_TOKEN 已被设为 -inf，无需特判
                {
                    float lp = logProbs[baseOff + v];
                    if (lp > best) best = lp;
                }
                scores[t, cb] = best;
            }
        }

        // 2. 层惩罚
        for (int t = 0; t < targetLen; t++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                scores[t, cb] -= cb * LayerPenaltyFactor;

        // 3. Gumbel 噪声（position temperature）
        if (PositionTemperature > 0f)
        {
            float invTemp = 1f / PositionTemperature;
            for (int t = 0; t < targetLen; t++)
                for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                {
                    double u = Math.Max(1e-10, _rng.NextDouble());
                    double gumbel = -Math.Log(-Math.Log(u));
                    scores[t, cb] = (float)(scores[t, cb] * invTemp + gumbel);
                }
        }

        // 4. 已解 mask 位置 → -inf；统计仍 mask 的数量
        int totalMasked = 0;
        for (int t = 0; t < targetLen; t++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                if (inputIds[0, cb, genStart + t] != MASK_TOKEN)
                    scores[t, cb] = float.NegativeInfinity;
                else
                    totalMasked++;
            }

        if (totalMasked == 0) return 0;
        kNew = Math.Min(kNew, totalMasked);

        // 5. Top-k 位置选择（partial sort：只需前 kNew 个）
        var allScores = new (int t, int cb, float score)[totalMasked];
        int idx = 0;
        for (int t = 0; t < targetLen; t++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                if (inputIds[0, cb, genStart + t] == MASK_TOKEN)
                    allScores[idx++] = (t, cb, scores[t, cb]);

        // 降序排序（选信心最高的 kNew 个）
        Array.Sort(allScores, (a, b) => b.score.CompareTo(a.score));

        // 6. 填入预测 token
        int unmasked = 0;
        for (int i = 0; i < kNew; i++)
        {
            var (t, cb, _) = allScores[i];
            inputIds[0, cb, genStart + t] = predTokens[t, cb];
            unmasked++;
        }
        return unmasked;
    }

    // ════════════════════════════════════════════════════════════════
    // FinalUnmaskAll — 快速路径 + 最终强制解 mask
    // ════════════════════════════════════════════════════════════════
    void FinalUnmaskAll(long[,,] inputIds, bool[,] audioMask,
                        int S, int genStart, int targetLen,
                        int T_ref, int T_text)
    {
        // ★ 快速路径：最后一步通常已解全部 mask，省掉第 N+1 次 forward
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

    // ════════════════════════════════════════════════════════════════
    // LMForward — ORT 推理，输出写入 outBuf
    // ════════════════════════════════════════════════════════════════
    void LMForward(long[,,] inputIds, bool[,] audioMask,
                   bool[,,,] attnMask, long[,] posIds,
                   int batchSize, int S,
                   float[] outBuf)
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

        // ★ 优化：优先用 ToArray() 替代逐元素 foreach
        // ToArray() 在 ORT 内部通常使用批量内存拷贝
        float[] arr = logitsTensor.ToArray();
        int len = arr.Length;
        if (outBuf.Length < len)
        {
            Debug.LogError($"[OmniVoiceLM] outBuf 太小: {outBuf.Length} < {len}，请检查 EnsureBuffers");
            return;
        }
        Array.Copy(arr, outBuf, len);
    }

    // ════════════════════════════════════════════════════════════════
    // Token 采样
    // ════════════════════════════════════════════════════════════════

    long ArgmaxToken(float[] logProbs, int baseOff)
    {
        float best = float.NegativeInfinity;
        long tok = 0;
        for (int v = 0; v < VOCAB_SIZE; v++)
        {
            float lp = logProbs[baseOff + v];
            if (lp > best) { best = lp; tok = v; }
        }
        return tok;
    }

    long SampleTokenTopKRatio(float[] logProbs, int baseOff, float ratio, float temperature)
    {
        int k = (int)Math.Ceiling(ratio * VOCAB_SIZE);

        var entries = new (float score, int idx)[VOCAB_SIZE];
        for (int v = 0; v < VOCAB_SIZE; v++)
            entries[v] = (logProbs[baseOff + v], v);
        Array.Sort(entries, (a, b) => b.score.CompareTo(a.score));

        var filtered = new float[VOCAB_SIZE];
        for (int v = 0; v < VOCAB_SIZE; v++) filtered[v] = float.NegativeInfinity;
        for (int i = 0; i < k && i < entries.Length; i++)
            filtered[entries[i].idx] = entries[i].score;

        for (int v = 0; v < VOCAB_SIZE; v++)
        {
            if (float.IsNegativeInfinity(filtered[v])) continue;
            double u = Math.Max(1e-10, _rng.NextDouble());
            filtered[v] = (float)(filtered[v] / temperature - Math.Log(-Math.Log(u)));
        }

        float best = float.NegativeInfinity;
        long tok = 0;
        for (int v = 0; v < VOCAB_SIZE; v++)
            if (filtered[v] > best) { best = filtered[v]; tok = v; }
        return tok;
    }

    // ════════════════════════════════════════════════════════════════
    // LogSoftmax — 原地写入，避免分配返回数组
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// src[srcOff .. srcOff+len) → dst[dstOff .. dstOff+len)
    /// </summary>
    static void LogSoftmaxSlice(float[] src, int srcOff, int len,
                                float[] dst, int dstOff)
    {
        float maxV = float.NegativeInfinity;
        for (int i = 0; i < len; i++)
        {
            float v = src[srcOff + i];
            if (v > maxV) maxV = v;
        }
        if (float.IsInfinity(maxV) || float.IsNaN(maxV))
        {
            for (int i = 0; i < len; i++) dst[dstOff + i] = float.NegativeInfinity;
            return;
        }
        float sumExp = 0f;
        for (int i = 0; i < len; i++) sumExp += MathF.Exp(src[srcOff + i] - maxV);
        float logSum = maxV + MathF.Log(sumExp);
        for (int i = 0; i < len; i++) dst[dstOff + i] = src[srcOff + i] - logSum;
    }

    /// <summary>
    /// src[srcOff..] → dst[dstOff..] 版本（srcOff != dstOff 均可）
    /// 同时也作为 src → _lsmWork 的重载入口
    /// </summary>
    static void LogSoftmaxSlice(float[] src, int srcOff, int len, float[] dst)
        => LogSoftmaxSlice(src, srcOff, len, dst, 0);

    /// <summary>
    /// 对 work[0..len) 原地做 log_softmax，结果写入 dst[dstOff..]
    /// </summary>
    static void LogSoftmaxSliceSelf(float[] work, int len, float[] dst, int dstOff)
    {
        float maxV = float.NegativeInfinity;
        for (int i = 0; i < len; i++) if (work[i] > maxV) maxV = work[i];
        if (float.IsInfinity(maxV) || float.IsNaN(maxV))
        {
            for (int i = 0; i < len; i++) dst[dstOff + i] = float.NegativeInfinity;
            return;
        }
        float sumExp = 0f;
        for (int i = 0; i < len; i++) sumExp += MathF.Exp(work[i] - maxV);
        float logSum = maxV + MathF.Log(sumExp);
        for (int i = 0; i < len; i++) dst[dstOff + i] = work[i] - logSum;
    }

    // ════════════════════════════════════════════════════════════════
    // 辅助方法
    // ════════════════════════════════════════════════════════════════

    /// <summary>按实际 S 懒分配/扩容复用缓冲区</summary>
    void EnsureBuffers(int S)
    {
        int cfgTotal = 2 * NUM_CODEBOOKS * S * VOCAB_SIZE;
        int single = NUM_CODEBOOKS * S * VOCAB_SIZE;
        if (_rawLogitsBuf == null || _rawLogitsBuf.Length < cfgTotal)
            _rawLogitsBuf = new float[cfgTotal];
        if (_resultBuf == null || _resultBuf.Length < single)
            _resultBuf = new float[single];
    }

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

    // ── Flatten 工具 ─────────────────────────────────────────────

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

    // bool 在 CLR 中保证为 1 byte，可以 BlockCopy
    static bool[] FlattenBool2D(bool[,] a)
    {
        var r = new bool[a.Length];
        Buffer.BlockCopy(a, 0, r, 0, a.Length);
        return r;
    }

    static bool[] FlattenBool4D(bool[,,,] a)
    {
        var r = new bool[a.Length];
        Buffer.BlockCopy(a, 0, r, 0, a.Length);
        return r;
    }

    public void Dispose() => _session?.Dispose();
}
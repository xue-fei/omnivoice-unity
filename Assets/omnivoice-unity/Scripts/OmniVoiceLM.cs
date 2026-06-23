using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class OmniVoiceLM : IDisposable
{
    public const int NUM_CODEBOOKS = 8;
    public const int VOCAB_SIZE = 1025;
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

    // ─── LMForward 零重建缓冲区（首次 Generate 时按 S 分配）─────────
    // 底层扁平数组
    long[] _idsBuf;    // [batchSize * NUM_CODEBOOKS * S]
    bool[] _audioBuf;  // [batchSize * S]
    bool[] _attnBuf;   // [batchSize * 1 * S * S]
    long[] _posBuf;    // [batchSize * S]
    float[] _rawLogitsBuf;
    float[] _resultBuf;

    // 复用 Tensor 对象（数据直接写入上面的 buffer，Tensor 不重建）
    DenseTensor<long> _tIds;
    DenseTensor<bool> _tAudio;
    DenseTensor<bool> _tAttn;
    DenseTensor<long> _tPos;
    IReadOnlyList<NamedOnnxValue> _inputList;
    int _tensorS = -1;  // 当前 Tensor 对应的 S，S 变化时重建
    int _tensorBatch = -1;

    // ─── LogSoftmax 工作区 ──────────────────────────────────────────
    readonly float[] _lsmWork = new float[VOCAB_SIZE];
    readonly float[] _lsmWork2 = new float[VOCAB_SIZE];  // condLSM 暂存

    // ─── DiffusionStep 复用缓冲 ─────────────────────────────────────
    (int t, int cb, float score)[] _allScoresBuf;
    long[] _predTokensBuf;
    float[] _scoresBuf;

    // ─── SampleToken 复用缓冲 ───────────────────────────────────────
    readonly (float score, int idx)[] _entriesBuf = new (float, int)[VOCAB_SIZE];
    readonly float[] _filteredBuf = new float[VOCAB_SIZE];

    // ─── attn mask / posIds 缓存 ────────────────────────────────────
    int _cachedAttnS = -1;
    int _cachedAttnTLen = -1;

    long[,] _cachedPosIds1;
    long[,] _cachedPosIds2;
    int _cachedPosS = -1;

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
        opts.EnableMemoryPattern = false;  // 动态 shape 场景
        opts.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
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
                    { "arena_extend_strategy",   "kSameAsRequested"  },
                    { "cudnn_conv_algo_search",  "HEURISTIC"         },
                    { "do_copy_in_default_stream","1"                },
                    { "use_tf32",                "1"                 },
                });
                opts.AppendExecutionProvider_CUDA(cudaOpts);
                epLoaded = true;
                Debug.Log($"[OmniVoiceLM] CUDA EP (device={deviceId})");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[OmniVoiceLM] CUDA EP 失败: {ex.Message}，回退 CPU");
            }
        }
        else if (ep == ExecutionProviderType.DML)
        {
            try
            {
                opts.AppendExecutionProvider_DML(deviceId);
                epLoaded = true;
                Debug.Log($"[OmniVoiceLM] DML EP (device={deviceId})");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[OmniVoiceLM] DML EP 失败: {ex.Message}，回退 CPU");
            }
        }

        if (!epLoaded)
        {
            opts.IntraOpNumThreads = Math.Max(4, Environment.ProcessorCount);
            Debug.Log($"[OmniVoiceLM] CPU EP (threads={opts.IntraOpNumThreads})");
        }

        _session = new InferenceSession(modelPath, opts);
        Debug.Log($"[OmniVoiceLM] 已加载: {modelPath}");
    }

    // ════════════════════════════════════════════════════════════════
    // WarmUp — 触发 cuDNN 算法缓存，消除首次推理卡顿
    // ════════════════════════════════════════════════════════════════
    public void WarmUp(int warmupS = 32)
    {
        Debug.Log($"[OmniVoiceLM] 预热 (S={warmupS})...");
        var sw = System.Diagnostics.Stopwatch.StartNew();

        int B = 2, S = warmupS;
        EnsureBuffers(S, B);
        EnsureStepBuffers(S);

        // audioMask: 全 true
        for (int i = 0; i < _audioBuf.Length; i++) _audioBuf[i] = true;
        // attn: 对角 true（最小合法）
        int attnStride = S * S;
        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
                _attnBuf[b * attnStride + s * S + s] = true;
        // ids: MASK
        for (int i = 0; i < _idsBuf.Length; i++) _idsBuf[i] = MASK_TOKEN;
        // pos
        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
                _posBuf[b * S + s] = s;

        for (int i = 0; i < 2; i++)
            _session.Run(_inputList);

        sw.Stop();
        Debug.Log($"[OmniVoiceLM] 预热完成 {sw.ElapsedMilliseconds}ms");
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

        EnsureBuffers(S, batchSize: 2);
        EnsureStepBuffers(targetLen);

        Debug.Log($"[OmniVoiceLM] 开始扩散: T_text={T_text} T_ref={T_ref} " +
                  $"T_gen={targetLen} S={S} steps={NumStep} GS={GuidanceScale}");

        // ── 构建 inputIds 和 audioMask（单 batch 原始数据）────────────
        var inputIds = new long[1, NUM_CODEBOOKS, S];
        var audioMask = new bool[1, S];

        for (int s = 0; s < T_text; s++)
        {
            long tid = textTokenIds[s];
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = tid;
            audioMask[0, s] = false;
        }
        for (int t = 0; t < T_ref; t++)
        {
            int s = T_text + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = Math.Clamp(refCodes[cb, t], 0, MASK_TOKEN - 1);
            audioMask[0, s] = true;
        }
        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                inputIds[0, cb, s] = MASK_TOKEN;
            audioMask[0, s] = true;
        }

        // ── 时移余弦调度 ─────────────────────────────────────────────
        double tau = TShift, N = NumStep;
        var r = new double[NumStep + 1];
        for (int n = 0; n <= NumStep; n++)
        {
            double u = (double)n / N;
            r[n] = tau * u / (1.0 + (tau - 1.0) * u);
        }

        int totalMasks = targetLen * NUM_CODEBOOKS;
        int remMasks = totalMasks;

        // ── 主扩散循环 ───────────────────────────────────────────────
        var swTotal = System.Diagnostics.Stopwatch.StartNew();
        long msForward = 0, msStep = 0;

        for (int step = 0; step < NumStep; step++)
        {
            int kNew;
            if (step == NumStep - 1)
                kNew = remMasks;
            else
            {
                kNew = (int)Math.Round((r[step + 1] - r[step]) * totalMasks);
                kNew = Math.Min(kNew, remMasks);
            }
            if (kNew <= 0) continue;

            var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            float[] logProbs = LMForwardWithCFG(
                inputIds, audioMask, S, genStart, T_ref, T_text, targetLen);
            msForward += (System.Diagnostics.Stopwatch.GetTimestamp() - t0)
                         * 1000 / System.Diagnostics.Stopwatch.Frequency;

            if (IsCorrupted(logProbs))
            {
                Debug.LogError($"[OmniVoiceLM] 步{step} NaN/Inf，降温重试");
                PositionTemperature = Mathf.Max(0.1f, PositionTemperature * 0.5f);
                logProbs = LMForwardWithCFG(
                    inputIds, audioMask, S, genStart, T_ref, T_text, targetLen);
                if (IsCorrupted(logProbs)) { Debug.LogError("恢复失败，终止"); break; }
            }

            t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            int unmasked = DiffusionStep(inputIds, logProbs, genStart, targetLen, S, kNew);
            msStep += (System.Diagnostics.Stopwatch.GetTimestamp() - t0)
                      * 1000 / System.Diagnostics.Stopwatch.Frequency;
            remMasks -= unmasked;

            if (step % 8 == 0)
                Debug.Log($"[OmniVoiceLM] step {step}/{NumStep} kNew={kNew} rem={remMasks} " +
                          $"fwd={msForward}ms step={msStep}ms");
        }

        FinalUnmaskAll(inputIds, audioMask, S, genStart, targetLen, T_ref, T_text);

        swTotal.Stop();
        Debug.Log($"[OmniVoiceLM] 完成: LMForward累计={msForward}ms " +
                  $"DiffusionStep累计={msStep}ms 总={swTotal.ElapsedMilliseconds}ms");

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
        Debug.Log($"[OmniVoiceLM] 音频={durSec:F1}s");
        return result;
    }

    // ════════════════════════════════════════════════════════════════
    // LMForwardWithCFG
    // ════════════════════════════════════════════════════════════════
    float[] LMForwardWithCFG(
        long[,,] inputIds, bool[,] audioMask,
        int S, int genStart, int T_ref, int T_text, int targetLen)
    {
        if (GuidanceScale > 0f)
        {
            FillCFGBatch(inputIds, audioMask, genStart, S, targetLen);
            var posIds = BuildPositionIds(2, S);
            FillPosBuf(posIds, 2, S);

            LMForward(batchSize: 2, S: S, outBuf: _rawLogitsBuf);

            int strideB = NUM_CODEBOOKS * S * VOCAB_SIZE;
            int strideCB = S * VOCAB_SIZE;

            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                for (int t = 0; t < targetLen; t++)
                {
                    int sCond = genStart + t;
                    int sUncond = t;

                    int condOff = 0 * strideB + cb * strideCB + sCond * VOCAB_SIZE;
                    int uncondOff = 1 * strideB + cb * strideCB + sUncond * VOCAB_SIZE;

                    LogSoftmaxSlice(_rawLogitsBuf, condOff, VOCAB_SIZE, _lsmWork2);
                    LogSoftmaxSlice(_rawLogitsBuf, uncondOff, VOCAB_SIZE, _lsmWork);

                    for (int v = 0; v < VOCAB_SIZE; v++)
                        _lsmWork[v] = _lsmWork2[v] + GuidanceScale * (_lsmWork2[v] - _lsmWork[v]);

                    int resultOff = cb * strideCB + sCond * VOCAB_SIZE;
                    LogSoftmaxSliceSelf(_lsmWork, VOCAB_SIZE, _resultBuf, resultOff);
                    _resultBuf[resultOff + MASK_TOKEN] = float.NegativeInfinity;
                }
            }
            return _resultBuf;
        }
        else
        {
            FillSingleBatch(inputIds, audioMask, S);
            var posIds = BuildPositionIds(1, S);
            FillPosBuf(posIds, 1, S);

            LMForward(batchSize: 1, S: S, outBuf: _rawLogitsBuf);

            int strideCB = S * VOCAB_SIZE;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                for (int s = 0; s < S; s++)
                {
                    int off = cb * strideCB + s * VOCAB_SIZE;
                    LogSoftmaxSlice(_rawLogitsBuf, off, VOCAB_SIZE, _resultBuf, off);
                    _resultBuf[off + MASK_TOKEN] = float.NegativeInfinity;
                }
            return _resultBuf;
        }
    }

    // ════════════════════════════════════════════════════════════════
    // LMForward — 直接使用复用 Tensor，零重建
    // ════════════════════════════════════════════════════════════════
    void LMForward(int batchSize, int S, float[] outBuf)
    {
        // Tensor shape 变化时重建（正常 Generate 内只建一次）
        if (_tensorS != S || _tensorBatch != batchSize)
        {
            _tIds = new DenseTensor<long>(_idsBuf, new[] { batchSize, NUM_CODEBOOKS, S });
            _tAudio = new DenseTensor<bool>(_audioBuf, new[] { batchSize, S });
            _tAttn = new DenseTensor<bool>(_attnBuf, new[] { batchSize, 1, S, S });
            _tPos = new DenseTensor<long>(_posBuf, new[] { batchSize, S });
            _inputList = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids",      _tIds),
                NamedOnnxValue.CreateFromTensor("audio_mask",     _tAudio),
                NamedOnnxValue.CreateFromTensor("attention_mask", _tAttn),
                NamedOnnxValue.CreateFromTensor("position_ids",   _tPos),
            };
            _tensorS = S;
            _tensorBatch = batchSize;
            Debug.Log($"[OmniVoiceLM] Tensor 重建: batch={batchSize} S={S}");
        }

        using var results = _session.Run(_inputList);

        // 直接从输出 Tensor 的底层 buffer 读取，跳过 ToArray()
        var logitsTensor = results[0].AsTensor<float>();
        float[] arr = logitsTensor.ToArray();  // ORT C# binding 暂无 Span 直接访问
        int len = arr.Length;
        if (outBuf.Length >= len)
            Buffer.BlockCopy(arr, 0, outBuf, 0, len * sizeof(float));
        else
            Debug.LogError($"[OmniVoiceLM] outBuf 太小 {outBuf.Length}<{len}");
    }

    // ════════════════════════════════════════════════════════════════
    // FillCFGBatch — 直接写入复用 _idsBuf/_audioBuf/_attnBuf，不 new 任何数组
    // ════════════════════════════════════════════════════════════════
    void FillCFGBatch(long[,,] srcIds, bool[,] srcAudio,
                      int genStart, int S, int targetLen)
    {
        int cbS = NUM_CODEBOOKS * S;
        int rowB = S * sizeof(long);

        // ── cond (b=0) ids：BlockCopy 每个 codebook 行 ──────────────
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
        {
            int srcOff = (cb * S) * sizeof(long);                  // srcIds[0,cb,0]
            int dstOff = (0 * cbS + cb * S) * sizeof(long);       // _idsBuf[0,cb,0]
            Buffer.BlockCopy(srcIds, srcOff, _idsBuf, dstOff, rowB);
        }

        // ── uncond (b=1) ids：生成区 + MASK 填充 ────────────────────
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
        {
            int srcOff = (cb * S + genStart) * sizeof(long);
            int dstOff = (1 * cbS + cb * S) * sizeof(long);
            Buffer.BlockCopy(srcIds, srcOff, _idsBuf, dstOff, targetLen * sizeof(long));
            int fillStart = 1 * cbS + cb * S + targetLen;
            for (int s = targetLen; s < S; s++) _idsBuf[fillStart++] = MASK_TOKEN;
        }

        // ── cond audioMask ──────────────────────────────────────────
        for (int s = 0; s < S; s++) _audioBuf[s] = srcAudio[0, s];

        // ── uncond audioMask：生成区 true，其余 false ─────────────────
        for (int s = 0; s < S; s++) _audioBuf[S + s] = s < targetLen;

        // ── attn mask（缓存：S 和 targetLen 不变时跳过重建）──────────
        if (_cachedAttnS != S || _cachedAttnTLen != targetLen)
        {
            _cachedAttnS = S;
            _cachedAttnTLen = targetLen;

            int stride = S * S;  // per-batch stride in _attnBuf

            // cond: 全 true
            for (int i = 0; i < stride; i++) _attnBuf[i] = true;

            // uncond: [0,targetLen)×[0,targetLen) true；pad 对角
            Array.Clear(_attnBuf, stride, stride);
            for (int i = 0; i < targetLen; i++)
                for (int j = 0; j < targetLen; j++)
                    _attnBuf[stride + i * S + j] = true;
            for (int s = targetLen; s < S; s++)
                _attnBuf[stride + s * S + s] = true;
        }
    }

    void FillSingleBatch(long[,,] srcIds, bool[,] srcAudio, int S)
    {
        int cbS = NUM_CODEBOOKS * S;
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
        {
            int srcOff = (cb * S) * sizeof(long);
            int dstOff = (cb * S) * sizeof(long);
            Buffer.BlockCopy(srcIds, srcOff, _idsBuf, dstOff, S * sizeof(long));
        }
        for (int s = 0; s < S; s++) _audioBuf[s] = srcAudio[0, s];
        // single batch attn: 全 true
        int stride = S * S;
        for (int i = 0; i < stride; i++) _attnBuf[i] = true;
    }

    void FillPosBuf(long[,] posIds, int B, int S)
    {
        Buffer.BlockCopy(posIds, 0, _posBuf, 0, B * S * sizeof(long));
    }

    // ════════════════════════════════════════════════════════════════
    // DiffusionStep — 复用缓冲，零 new[]
    // ════════════════════════════════════════════════════════════════
    int DiffusionStep(long[,,] inputIds, float[] logProbs,
                      int genStart, int targetLen, int S, int kNew)
    {
        int strideCB = S * VOCAB_SIZE;

        for (int t = 0; t < targetLen; t++)
        {
            int s = genStart + t;
            int tBase = t * NUM_CODEBOOKS;
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                int baseOff = cb * strideCB + s * VOCAB_SIZE;

                _predTokensBuf[tBase + cb] = ClassTemperature > 0f
                    ? SampleTokenTopKRatio(logProbs, baseOff, 0.1f, ClassTemperature)
                    : ArgmaxToken(logProbs, baseOff);

                float best = float.NegativeInfinity;
                for (int v = 0; v < VOCAB_SIZE; v++)
                {
                    float lp = logProbs[baseOff + v];
                    if (lp > best) best = lp;
                }
                _scoresBuf[tBase + cb] = best;
            }
        }

        for (int t = 0; t < targetLen; t++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                _scoresBuf[t * NUM_CODEBOOKS + cb] -= cb * LayerPenaltyFactor;

        if (PositionTemperature > 0f)
        {
            float invTemp = 1f / PositionTemperature;
            for (int t = 0; t < targetLen; t++)
                for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                {
                    double u = Math.Max(1e-10, _rng.NextDouble());
                    _scoresBuf[t * NUM_CODEBOOKS + cb] =
                        (float)(_scoresBuf[t * NUM_CODEBOOKS + cb] * invTemp
                                - Math.Log(-Math.Log(u)));
                }
        }

        int totalMasked = 0;
        int allScoresIdx = 0;
        for (int t = 0; t < targetLen; t++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            {
                int idx = t * NUM_CODEBOOKS + cb;
                if (inputIds[0, cb, genStart + t] != MASK_TOKEN)
                    _scoresBuf[idx] = float.NegativeInfinity;
                else
                    _allScoresBuf[allScoresIdx++] = (t, cb, _scoresBuf[idx]);
                totalMasked += (inputIds[0, cb, genStart + t] == MASK_TOKEN) ? 1 : 0;
            }

        if (totalMasked == 0) return 0;
        kNew = Math.Min(kNew, totalMasked);

        Array.Sort(_allScoresBuf, 0, totalMasked,
            Comparer<(int, int, float)>.Create((a, b) => b.Item3.CompareTo(a.Item3)));

        for (int i = 0; i < kNew; i++)
        {
            var (t, cb, _) = _allScoresBuf[i];
            inputIds[0, cb, genStart + t] = _predTokensBuf[t * NUM_CODEBOOKS + cb];
        }
        return kNew;
    }

    // ════════════════════════════════════════════════════════════════
    // FinalUnmaskAll
    // ════════════════════════════════════════════════════════════════
    void FinalUnmaskAll(long[,,] inputIds, bool[,] audioMask,
                        int S, int genStart, int targetLen, int T_ref, int T_text)
    {
        int maskCount = 0;
        for (int t = 0; t < targetLen; t++)
            for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
                if (inputIds[0, cb, genStart + t] == MASK_TOKEN) maskCount++;

        if (maskCount == 0) return;

        Debug.Log($"[OmniVoiceLM] 最终强制解 mask: 残余 {maskCount} 个");
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
        for (int v = 0; v < VOCAB_SIZE; v++) _entriesBuf[v] = (logProbs[baseOff + v], v);
        Array.Sort(_entriesBuf, (a, b) => b.score.CompareTo(a.score));
        for (int v = 0; v < VOCAB_SIZE; v++) _filteredBuf[v] = float.NegativeInfinity;
        for (int i = 0; i < k; i++) _filteredBuf[_entriesBuf[i].idx] = _entriesBuf[i].score;
        for (int v = 0; v < VOCAB_SIZE; v++)
        {
            if (float.IsNegativeInfinity(_filteredBuf[v])) continue;
            double u = Math.Max(1e-10, _rng.NextDouble());
            _filteredBuf[v] = (float)(_filteredBuf[v] / temperature - Math.Log(-Math.Log(u)));
        }
        float best = float.NegativeInfinity; long tok = 0;
        for (int v = 0; v < VOCAB_SIZE; v++)
            if (_filteredBuf[v] > best) { best = _filteredBuf[v]; tok = v; }
        return tok;
    }

    // ════════════════════════════════════════════════════════════════
    // LogSoftmax
    // ════════════════════════════════════════════════════════════════
    static void LogSoftmaxSlice(float[] src, int srcOff, int len, float[] dst, int dstOff)
    {
        float maxV = float.NegativeInfinity;
        for (int i = 0; i < len; i++) { float v = src[srcOff + i]; if (v > maxV) maxV = v; }
        if (float.IsInfinity(maxV) || float.IsNaN(maxV))
        { for (int i = 0; i < len; i++) dst[dstOff + i] = float.NegativeInfinity; return; }
        float sumExp = 0f;
        for (int i = 0; i < len; i++) sumExp += MathF.Exp(src[srcOff + i] - maxV);
        float logSum = maxV + MathF.Log(sumExp);
        for (int i = 0; i < len; i++) dst[dstOff + i] = src[srcOff + i] - logSum;
    }
    static void LogSoftmaxSlice(float[] src, int srcOff, int len, float[] dst)
        => LogSoftmaxSlice(src, srcOff, len, dst, 0);

    static void LogSoftmaxSliceSelf(float[] work, int len, float[] dst, int dstOff)
    {
        float maxV = float.NegativeInfinity;
        for (int i = 0; i < len; i++) if (work[i] > maxV) maxV = work[i];
        if (float.IsInfinity(maxV) || float.IsNaN(maxV))
        { for (int i = 0; i < len; i++) dst[dstOff + i] = float.NegativeInfinity; return; }
        float sumExp = 0f;
        for (int i = 0; i < len; i++) sumExp += MathF.Exp(work[i] - maxV);
        float logSum = maxV + MathF.Log(sumExp);
        for (int i = 0; i < len; i++) dst[dstOff + i] = work[i] - logSum;
    }

    // ════════════════════════════════════════════════════════════════
    // 辅助：缓冲区分配
    // ════════════════════════════════════════════════════════════════
    void EnsureBuffers(int S, int batchSize)
    {
        int idsSz = batchSize * NUM_CODEBOOKS * S;
        int audioSz = batchSize * S;
        int attnSz = batchSize * S * S;
        int posSz = batchSize * S;
        int rawSz = batchSize * NUM_CODEBOOKS * S * VOCAB_SIZE;
        int resSz = NUM_CODEBOOKS * S * VOCAB_SIZE;

        bool rebuild = false;
        if (_idsBuf == null || _idsBuf.Length < idsSz) { _idsBuf = new long[idsSz]; rebuild = true; }
        if (_audioBuf == null || _audioBuf.Length < audioSz) { _audioBuf = new bool[audioSz]; rebuild = true; }
        if (_attnBuf == null || _attnBuf.Length < attnSz) { _attnBuf = new bool[attnSz]; rebuild = true; }
        if (_posBuf == null || _posBuf.Length < posSz) { _posBuf = new long[posSz]; rebuild = true; }
        if (_rawLogitsBuf == null || _rawLogitsBuf.Length < rawSz)
            _rawLogitsBuf = new float[rawSz];
        if (_resultBuf == null || _resultBuf.Length < resSz)
            _resultBuf = new float[resSz];

        // buffer 扩容时强制重建 Tensor（使 _tensorS 失效）
        if (rebuild) _tensorS = -1;
    }

    void EnsureStepBuffers(int targetLen)
    {
        int need = targetLen * NUM_CODEBOOKS;
        if (_allScoresBuf == null || _allScoresBuf.Length < need) _allScoresBuf = new (int, int, float)[need];
        if (_predTokensBuf == null || _predTokensBuf.Length < need) _predTokensBuf = new long[need];
        if (_scoresBuf == null || _scoresBuf.Length < need) _scoresBuf = new float[need];
    }

    // ════════════════════════════════════════════════════════════════
    // 辅助：BuildPositionIds（缓存）
    // ════════════════════════════════════════════════════════════════
    long[,] BuildPositionIds(int B, int S)
    {
        if (_cachedPosS == S)
        {
            if (B == 1 && _cachedPosIds1 != null) return _cachedPosIds1;
            if (B == 2 && _cachedPosIds2 != null) return _cachedPosIds2;
        }
        var p = new long[B, S];
        for (int b = 0; b < B; b++)
            for (int s = 0; s < S; s++)
                p[b, s] = s;
        _cachedPosS = S;
        if (B == 1) _cachedPosIds1 = p;
        if (B == 2) _cachedPosIds2 = p;
        return p;
    }

    bool IsCorrupted(float[] arr)
    {
        int bad = 0;
        for (int i = 0; i < arr.Length; i++)
            if (float.IsNaN(arr[i]) || float.IsInfinity(arr[i])) bad++;
        return bad > arr.Length / 100;
    }

    public void Dispose() => _session?.Dispose();
}

public enum ExecutionProviderType { CPU, CUDA, DML }
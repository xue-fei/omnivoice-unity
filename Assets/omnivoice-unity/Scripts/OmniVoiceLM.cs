using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

/// <summary>
/// OmniVoice 语言模型推理引擎 — 基于 ONNX Runtime 的扩散式语音生成器
/// 
/// 核心功能：
///   1. 加载 ONNX 格式的 OmniVoice LM 模型
///   2. 执行多步扩散去噪循环，逐步预测音频 token
///   3. 支持 CFG（Classifier-Free Guidance）引导生成
///   4. 支持多 codebook 并行采样与 Layer Penalty
/// </summary>
public class OmniVoiceLM : IDisposable
{
    // ════════════════════════════════════════════════════════════════
    // 常量定义
    // ════════════════════════════════════════════════════════════════

    /// <summary>音频 codebook 数量（EnCodec 结构）</summary>
    public const int NUM_CODEBOOKS = 8;

    /// <summary>词表大小（1024 个有效 token + 1 个 MASK）</summary>
    public const int VOCAB_SIZE = 1025;

    /// <summary>MASK token ID，用于扩散模型的待预测位置</summary>
    public const int MASK_TOKEN = 1024;

    /// <summary>PAD token ID</summary>
    public const int PAD_TOKEN = 0;

    // ════════════════════════════════════════════════════════════════
    // 核心组件
    // ════════════════════════════════════════════════════════════════

    /// <summary>ONNX Runtime 推理会话</summary>
    InferenceSession _session;

    /// <summary>随机数生成器（用于 Gumbel 采样）</summary>
    System.Random _rng;

    // ════════════════════════════════════════════════════════════════
    // 生成参数
    // ════════════════════════════════════════════════════════════════

    /// <summary>扩散去噪步数（默认 32，减少可提速但可能降低质量）</summary>
    public int NumStep = 32;

    /// <summary>CFG 引导强度（0=关闭，越大越贴近条件）</summary>
    public float GuidanceScale = 2.0f;

    /// <summary>调度时移 τ（控制扩散速率）</summary>
    public float TShift = 0.1f;

    /// <summary>位置选择温度（Gumbel 噪声强度）</summary>
    public float PositionTemperature = 5.0f;

    /// <summary>类别采样温度（0=greedy argmax）</summary>
    public float ClassTemperature = 0.0f;

    /// <summary>层惩罚系数（控制 codebook 从低到高逐层解 mask）</summary>
    public float LayerPenaltyFactor = 5.0f;

    // ════════════════════════════════════════════════════════════════
    // LMForward 复用缓冲区
    // ════════════════════════════════════════════════════════════════

    /// <summary>输入 token IDs 扁平缓冲区 [batchSize * NUM_CODEBOOKS * S]</summary>
    long[] _idsBuf;

    /// <summary>音频掩码缓冲区 [batchSize * S]</summary>
    bool[] _audioBuf;

    /// <summary>注意力掩码缓冲区 [batchSize * 1 * S * S]</summary>
    bool[] _attnBuf;

    /// <summary>位置 IDs 缓冲区 [batchSize * S]</summary>
    long[] _posBuf;

    /// <summary>原始 logits 输出缓冲区 [batchSize * NUM_CODEBOOKS * S * VOCAB_SIZE]</summary>
    float[] _rawLogitsBuf;

    /// <summary>LogSoftmax 结果缓冲区 [NUM_CODEBOOKS * S * VOCAB_SIZE]</summary>
    float[] _resultBuf;

    /// <summary>复用 Tensor 对象 — input_ids</summary>
    DenseTensor<long> _tIds;

    /// <summary>复用 Tensor 对象 — audio_mask</summary>
    DenseTensor<bool> _tAudio;

    /// <summary>复用 Tensor 对象 — attention_mask</summary>
    DenseTensor<bool> _tAttn;

    /// <summary>复用 Tensor 对象 — position_ids</summary>
    DenseTensor<long> _tPos;

    /// <summary>ONNX 输入列表（复用避免重建）</summary>
    IReadOnlyList<NamedOnnxValue> _inputList;

    /// <summary>当前 Tensor 对应的序列长度 S（变化时重建）</summary>
    int _tensorS = -1;

    /// <summary>当前 Tensor 对应的 batch size</summary>
    int _tensorBatch = -1;

    // ════════════════════════════════════════════════════════════════
    // LogSoftmax 工作区
    // ════════════════════════════════════════════════════════════════

    /// <summary>串行 LogSoftmax 工作区</summary>
    readonly float[] _lsmWork = new float[VOCAB_SIZE];

    /// <summary>串行 LogSoftmax 暂存（条件分支）</summary>
    readonly float[] _lsmWork2 = new float[VOCAB_SIZE];

    /// <summary>【方案 D】并行 LogSoftmax 工作区 [NUM_CODEBOOKS][VOCAB_SIZE]</summary>
    float[][] _lsmWorkPar;

    /// <summary>【方案 D】并行 LogSoftmax 暂存 [NUM_CODEBOOKS][VOCAB_SIZE]</summary>
    float[][] _lsmWork2Par;

    // ════════════════════════════════════════════════════════════════
    // DiffusionStep 复用缓冲
    // ════════════════════════════════════════════════════════════════

    /// <summary>候选分数缓冲区（用于 Top-K 排序）</summary>
    (int t, int cb, float score)[] _allScoresBuf;

    /// <summary>预测 token 缓冲区</summary>
    long[] _predTokensBuf;

    /// <summary>分数临时缓冲区</summary>
    float[] _scoresBuf;

    // ════════════════════════════════════════════════════════════════
    // Token 采样复用缓冲
    // ════════════════════════════════════════════════════════════════

    /// <summary>Top-K 排序条目缓冲区</summary>
    readonly (float score, int idx)[] _entriesBuf = new (float, int)[VOCAB_SIZE];

    /// <summary>Top-K 过滤后 log-prob 缓冲区</summary>
    readonly float[] _filteredBuf = new float[VOCAB_SIZE];

    // ════════════════════════════════════════════════════════════════
    // 缓存状态
    // ════════════════════════════════════════════════════════════════

    /// <summary>缓存的 attn mask 序列长度</summary>
    int _cachedAttnS = -1;

    /// <summary>缓存的 attn mask 目标长度</summary>
    int _cachedAttnTLen = -1;

    /// <summary>缓存的 position IDs（batch=1）</summary>
    long[,] _cachedPosIds1;

    /// <summary>缓存的 position IDs（batch=2）</summary>
    long[,] _cachedPosIds2;

    /// <summary>缓存的 position IDs 序列长度</summary>
    int _cachedPosS = -1;

    // ════════════════════════════════════════════════════════════════
    // 构造函数
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 构造 OmniVoiceLM 推理引擎
    /// </summary>
    /// <param name="modelPath">ONNX 模型文件路径</param>
    /// <param name="executionProvider">执行提供程序类型（CUDA/DML/CPU）</param>
    /// <param name="deviceId">GPU 设备索引</param>
    /// <param name="seed">随机数种子</param>
    public OmniVoiceLM(
        string modelPath,
        ExecutionProviderType executionProvider = ExecutionProviderType.CUDA,
        int deviceId = 0,
        int seed = 42)
    {
        _rng = new System.Random(seed);

        // 构建 SessionOptions
        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.EnableMemoryPattern = false;  // 动态 shape 场景关闭
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        options.InterOpNumThreads = 1;
        options.IntraOpNumThreads = 4;

        bool executionProviderLoaded = false;

        // 按优先级尝试 EP：CUDA → DML → CPU
        if (executionProvider == ExecutionProviderType.CUDA)
        {
            try
            {
                var cudaOptions = new OrtCUDAProviderOptions();
                cudaOptions.UpdateOptions(new Dictionary<string, string>
                {
                    { "device_id",                deviceId.ToString() },
                    { "arena_extend_strategy",    "kSameAsRequested"  },
                    { "do_copy_in_default_stream","1"                },
                });
                options.AppendExecutionProvider_CUDA(cudaOptions);
                executionProviderLoaded = true;
                Debug.Log($"[OmniVoiceLM] CUDA EP (device={deviceId})");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[OmniVoiceLM] CUDA EP 失败: {ex.Message}，回退 CPU");
            }
        }
        else if (executionProvider == ExecutionProviderType.DML)
        {
            try
            {
                options.AppendExecutionProvider_DML(deviceId);
                executionProviderLoaded = true;
                Debug.Log($"[OmniVoiceLM] DML EP (device={deviceId})");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[OmniVoiceLM] DML EP 失败: {ex.Message}，回退 CPU");
            }
        }

        // 回退 CPU
        if (!executionProviderLoaded)
        {
            options.IntraOpNumThreads = Math.Max(4, Environment.ProcessorCount);
            Debug.Log($"[OmniVoiceLM] CPU EP (threads={options.IntraOpNumThreads})");
        }

        _session = new InferenceSession(modelPath, options);
        Debug.Log($"[OmniVoiceLM] 已加载: {modelPath}");
    }

    // ════════════════════════════════════════════════════════════════
    // WarmUp — 触发 cuDNN 算法缓存，消除首次推理卡顿
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 预热推理引擎，触发 cuDNN/ORT 内部算法缓存
    /// </summary>
    /// <param name="warmupSequenceLength">预热使用的序列长度</param>
    public void WarmUp(int warmupSequenceLength = 32)
    {
        Debug.Log($"[OmniVoiceLM] 预热 (S={warmupSequenceLength})...");
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        int batchSize = 2, sequenceLength = warmupSequenceLength;
        EnsureBuffers(sequenceLength, batchSize);
        EnsureStepBuffers(sequenceLength);

        // audioMask: 全 true
        for (int i = 0; i < _audioBuf.Length; i++) _audioBuf[i] = true;

        // attn: 对角 true（最小合法）
        int attentionStride = sequenceLength * sequenceLength;
        for (int b = 0; b < batchSize; b++)
            for (int s = 0; s < sequenceLength; s++)
                _attnBuf[b * attentionStride + s * sequenceLength + s] = true;

        // ids: MASK
        for (int i = 0; i < _idsBuf.Length; i++) _idsBuf[i] = MASK_TOKEN;

        // position IDs
        for (int b = 0; b < batchSize; b++)
            for (int s = 0; s < sequenceLength; s++)
                _posBuf[b * sequenceLength + s] = s;

        // 执行 2 次预热推理
        for (int i = 0; i < 2; i++)
            _session.Run(_inputList);

        stopwatch.Stop();
        Debug.Log($"[OmniVoiceLM] 预热完成 {stopwatch.ElapsedMilliseconds}ms");
    }

    // ════════════════════════════════════════════════════════════════
    // Generate — 主入口
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 执行扩散式语音 token 生成
    /// </summary>
    /// <param name="textTokenIds">文本 token IDs（可为空）</param>
    /// <param name="refCodes">参考音频 codes [NUM_CODEBOOKS, T_ref]（可为空）</param>
    /// <param name="targetLen">目标生成长度（帧数）</param>
    /// <returns>生成的 audio codes [NUM_CODEBOOKS, targetLen]</returns>
    public long[,] Generate(int[] textTokenIds, long[,] refCodes, int targetLen)
    {
        int textLength = textTokenIds != null ? textTokenIds.Length : 0;
        int refLength = refCodes != null ? refCodes.GetLength(1) : 0;

        // 自动估算目标长度
        if (targetLen <= 0) targetLen = Mathf.Max(50, refLength > 0 ? refLength : 100);

        int generateStart = textLength + refLength;
        int sequenceLength = generateStart + targetLen;

        // 分配缓冲区
        EnsureBuffers(sequenceLength, batchSize: 2);
        EnsureStepBuffers(targetLen);

        Debug.Log($"[OmniVoiceLM] 开始扩散: T_text={textLength} T_ref={refLength} " +
                  $"T_gen={targetLen} S={sequenceLength} steps={NumStep} GS={GuidanceScale}");

        // ════════════════════════════════════════════════════════════
        // 构建 inputIds 和 audioMask
        // ════════════════════════════════════════════════════════════
        var inputIds = new long[1, NUM_CODEBOOKS, sequenceLength];
        var audioMask = new bool[1, sequenceLength];

        // 文本段：所有 codebook 共享同一 token，audioMask=false
        for (int s = 0; s < textLength; s++)
        {
            long tokenId = textTokenIds[s];
            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
                inputIds[0, codebook, s] = tokenId;
            audioMask[0, s] = false;
        }

        // 参考音频段：填入 refCodes，audioMask=true
        for (int t = 0; t < refLength; t++)
        {
            int s = textLength + t;
            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
                inputIds[0, codebook, s] = Math.Clamp(refCodes[codebook, t], 0, MASK_TOKEN - 1);
            audioMask[0, s] = true;
        }

        // 生成段：初始化为 MASK_TOKEN，audioMask=true
        for (int t = 0; t < targetLen; t++)
        {
            int s = generateStart + t;
            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
                inputIds[0, codebook, s] = MASK_TOKEN;
            audioMask[0, s] = true;
        }

        // ════════════════════════════════════════════════════════════
        // 时移余弦调度 r[n]
        // ════════════════════════════════════════════════════════════
        double tau = TShift, totalSteps = NumStep;
        var schedule = new double[NumStep + 1];
        for (int n = 0; n <= NumStep; n++)
        {
            double progress = n / totalSteps;
            schedule[n] = tau * progress / (1.0 + (tau - 1.0) * progress);
        }

        int totalMaskCount = targetLen * NUM_CODEBOOKS;
        int remainingMasks = totalMaskCount;

        // ════════════════════════════════════════════════════════════
        // 主扩散循环
        // ════════════════════════════════════════════════════════════
        var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();
        long forwardMs = 0, stepMs = 0;

        for (int step = 0; step < NumStep; step++)
        {
            // 计算本轮要解 mask 的 token 数
            int newUnmaskCount;
            if (step == NumStep - 1)
                newUnmaskCount = remainingMasks;  // 最后一步全部解完
            else
            {
                newUnmaskCount = (int)Math.Round((schedule[step + 1] - schedule[step]) * totalMaskCount);
                newUnmaskCount = Math.Min(newUnmaskCount, remainingMasks);
            }
            if (newUnmaskCount <= 0) continue;

            // 执行 LM Forward + CFG
            var timestamp = System.Diagnostics.Stopwatch.GetTimestamp();
            float[] logProbabilities = LMForwardWithCFG(
                inputIds, audioMask, sequenceLength, generateStart, refLength, textLength, targetLen);
            forwardMs += (System.Diagnostics.Stopwatch.GetTimestamp() - timestamp)
                         * 1000 / System.Diagnostics.Stopwatch.Frequency;

            // NaN/Inf 检测与恢复
            if (IsCorrupted(logProbabilities))
            {
                Debug.LogError($"[OmniVoiceLM] 步{step} NaN/Inf，降温重试");
                PositionTemperature = Mathf.Max(0.1f, PositionTemperature * 0.5f);
                logProbabilities = LMForwardWithCFG(
                    inputIds, audioMask, sequenceLength, generateStart, refLength, textLength, targetLen);
                if (IsCorrupted(logProbabilities))
                {
                    Debug.LogError("恢复失败，终止");
                    break;
                }
            }

            // 执行扩散采样步骤
            timestamp = System.Diagnostics.Stopwatch.GetTimestamp();
            int unmaskedCount = DiffusionStep(
                inputIds, logProbabilities, generateStart, targetLen, sequenceLength, newUnmaskCount);
            stepMs += (System.Diagnostics.Stopwatch.GetTimestamp() - timestamp)
                      * 1000 / System.Diagnostics.Stopwatch.Frequency;

            remainingMasks -= unmaskedCount;

            // 日志输出
            if (step % 8 == 0)
                Debug.Log($"[OmniVoiceLM] step {step}/{NumStep} kNew={newUnmaskCount} rem={remainingMasks} " +
                          $"fwd={forwardMs}ms step={stepMs}ms");
        }

        // 强制解剩余 mask（兜底）
        FinalUnmaskAll(inputIds, audioMask, sequenceLength, generateStart, targetLen, refLength, textLength);

        totalStopwatch.Stop();
        Debug.Log($"[OmniVoiceLM] 完成: LMForward累计={forwardMs}ms " +
                  $"DiffusionStep累计={stepMs}ms 总={totalStopwatch.ElapsedMilliseconds}ms");

        // ════════════════════════════════════════════════════════════
        // 提取结果
        // ════════════════════════════════════════════════════════════
        var result = new long[NUM_CODEBOOKS, targetLen];
        for (int t = 0; t < targetLen; t++)
        {
            int s = generateStart + t;
            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
            {
                long value = inputIds[0, codebook, s];
                result[codebook, t] = (value == MASK_TOKEN) ? 0L : Math.Clamp(value, 0L, MASK_TOKEN - 1L);
            }
        }

        float durationSeconds = targetLen * 960f / 24000f;
        Debug.Log($"[OmniVoiceLM] 音频={durationSeconds:F1}s");
        return result;
    }

    // ════════════════════════════════════════════════════════════════
    // LMForwardWithCFG — 带 Classifier-Free Guidance 的前向推理
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 执行带 CFG 引导的 LM 前向推理
    /// </summary>
    /// <param name="inputIds">输入 token IDs</param>
    /// <param name="audioMask">音频掩码</param>
    /// <param name="sequenceLength">序列总长度 S</param>
    /// <param name="generateStart">生成段起始位置</param>
    /// <param name="refLength">参考音频长度</param>
    /// <param name="textLength">文本长度</param>
    /// <param name="targetLen">目标生成长度</param>
    /// <returns>LogSoftmax 后的概率 [NUM_CODEBOOKS * S * VOCAB_SIZE]</returns>
    float[] LMForwardWithCFG(
        long[,,] inputIds, bool[,] audioMask,
        int sequenceLength, int generateStart, int refLength, int textLength, int targetLen)
    {
        if (GuidanceScale > 0f)
        {
            // ════════════════════════════════════════════════════════
            // CFG 模式：构造 cond/uncond 双 batch
            // ════════════════════════════════════════════════════════
            FillCFGBatch(inputIds, audioMask, generateStart, sequenceLength, targetLen);
            var positionIds = BuildPositionIds(2, sequenceLength);
            FillPosBuf(positionIds, 2, sequenceLength);

            LMForward(batchSize: 2, sequenceLength: sequenceLength, outBuf: _rawLogitsBuf);

            int batchStride = NUM_CODEBOOKS * sequenceLength * VOCAB_SIZE;
            int codebookStride = sequenceLength * VOCAB_SIZE;

            // ★ 方案 D：并行 LogSoftmax —— 按 codebook 维度并行
            // 8 个 codebook 相互独立，完美并行；targetLen 循环内串行保证缓存友好
            EnsureParallelLsmWorkBuffers();
            Parallel.For(0, NUM_CODEBOOKS, codebookIndex =>
            {
                var workBuffer = _lsmWorkPar[codebookIndex];
                var workBuffer2 = _lsmWork2Par[codebookIndex];

                for (int t = 0; t < targetLen; t++)
                {
                    int conditionalPosition = generateStart + t;
                    int unconditionalPosition = t;

                    int conditionalOffset = 0 * batchStride + codebookIndex * codebookStride + conditionalPosition * VOCAB_SIZE;
                    int unconditionalOffset = 1 * batchStride + codebookIndex * codebookStride + unconditionalPosition * VOCAB_SIZE;

                    // 分别计算 cond 和 uncond 的 log-softmax
                    LogSoftmaxSlice(_rawLogitsBuf, conditionalOffset, VOCAB_SIZE, workBuffer2);
                    LogSoftmaxSlice(_rawLogitsBuf, unconditionalOffset, VOCAB_SIZE, workBuffer);

                    // CFG 融合：logP_cond + scale * (logP_cond - logP_uncond)
                    for (int v = 0; v < VOCAB_SIZE; v++)
                        workBuffer[v] = workBuffer2[v] + GuidanceScale * (workBuffer2[v] - workBuffer[v]);

                    // 写回结果并屏蔽 MASK token
                    int resultOffset = codebookIndex * codebookStride + conditionalPosition * VOCAB_SIZE;
                    LogSoftmaxSliceSelf(workBuffer, VOCAB_SIZE, _resultBuf, resultOffset);
                    _resultBuf[resultOffset + MASK_TOKEN] = float.NegativeInfinity;
                }
            });

            return _resultBuf;
        }
        else
        {
            // ════════════════════════════════════════════════════════
            // 无 CFG 模式：单 batch
            // ════════════════════════════════════════════════════════
            FillSingleBatch(inputIds, audioMask, sequenceLength);
            var positionIds = BuildPositionIds(1, sequenceLength);
            FillPosBuf(positionIds, 1, sequenceLength);

            LMForward(batchSize: 1, sequenceLength: sequenceLength, outBuf: _rawLogitsBuf);

            int codebookStride = sequenceLength * VOCAB_SIZE;
            for (int codebookIndex = 0; codebookIndex < NUM_CODEBOOKS; codebookIndex++)
                for (int s = 0; s < sequenceLength; s++)
                {
                    int offset = codebookIndex * codebookStride + s * VOCAB_SIZE;
                    LogSoftmaxSlice(_rawLogitsBuf, offset, VOCAB_SIZE, _resultBuf, offset);
                    _resultBuf[offset + MASK_TOKEN] = float.NegativeInfinity;
                }

            return _resultBuf;
        }
    }

    // ════════════════════════════════════════════════════════════════
    // LMForward — ONNX 推理执行
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 执行 ONNX 推理并读取输出 logits
    /// 【方案 B】单次 ToArray + BlockCopy（兼容 Unity ORT，无 GetAsSpan）
    /// </summary>
    /// <param name="batchSize">批次大小</param>
    /// <param name="sequenceLength">序列长度 S</param>
    /// <param name="outBuf">输出缓冲区</param>
    void LMForward(int batchSize, int sequenceLength, float[] outBuf)
    {
        // Tensor shape 变化时重建（正常 Generate 内只建一次）
        if (_tensorS != sequenceLength || _tensorBatch != batchSize)
        {
            _tIds = new DenseTensor<long>(_idsBuf, new[] { batchSize, NUM_CODEBOOKS, sequenceLength });
            _tAudio = new DenseTensor<bool>(_audioBuf, new[] { batchSize, sequenceLength });
            _tAttn = new DenseTensor<bool>(_attnBuf, new[] { batchSize, 1, sequenceLength, sequenceLength });
            _tPos = new DenseTensor<long>(_posBuf, new[] { batchSize, sequenceLength });
            _inputList = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids",      _tIds),
                NamedOnnxValue.CreateFromTensor("audio_mask",     _tAudio),
                NamedOnnxValue.CreateFromTensor("attention_mask", _tAttn),
                NamedOnnxValue.CreateFromTensor("position_ids",   _tPos),
            };
            _tensorS = sequenceLength;
            _tensorBatch = batchSize;
            Debug.Log($"[OmniVoiceLM] Tensor 重建: batch={batchSize} S={sequenceLength}");
        }

        using var results = _session.Run(_inputList);

        // 方案 B：通过 AsTensor<float>() 获取 DenseTensor，单次 ToArray + BlockCopy
        var logitsTensor = results[0].AsTensor<float>();
        int length = (int)logitsTensor.Length;
        if (outBuf.Length >= length)
        {
            var array = logitsTensor.ToArray();
            Buffer.BlockCopy(array, 0, outBuf, 0, length * sizeof(float));
        }
        else
            Debug.LogError($"[OmniVoiceLM] outBuf 太小 {outBuf.Length}<{length}");
    }

    // ════════════════════════════════════════════════════════════════
    // FillCFGBatch — 构造 CFG 双 batch 输入
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 构造 CFG 双 batch 输入：cond(batch=0) + uncond(batch=1)
    /// </summary>
    void FillCFGBatch(
        long[,,] sourceIds, bool[,] sourceAudio,
        int generateStart, int sequenceLength, int targetLength)
    {
        int codebookStride = NUM_CODEBOOKS * sequenceLength;
        int rowBytes = sequenceLength * sizeof(long);

        // ── cond (batch=0) ids：直接复制 ──
        for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
        {
            int sourceOffset = (codebook * sequenceLength) * sizeof(long);
            int destOffset = (0 * codebookStride + codebook * sequenceLength) * sizeof(long);
            Buffer.BlockCopy(sourceIds, sourceOffset, _idsBuf, destOffset, rowBytes);
        }

        // ── uncond (batch=1) ids：生成区 + MASK 填充 ──
        for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
        {
            int sourceOffset = (codebook * sequenceLength + generateStart) * sizeof(long);
            int destOffset = (1 * codebookStride + codebook * sequenceLength) * sizeof(long);
            Buffer.BlockCopy(sourceIds, sourceOffset, _idsBuf, destOffset, targetLength * sizeof(long));

            int fillStart = 1 * codebookStride + codebook * sequenceLength + targetLength;
            for (int s = targetLength; s < sequenceLength; s++) _idsBuf[fillStart++] = MASK_TOKEN;
        }

        // ── cond audioMask ──
        for (int s = 0; s < sequenceLength; s++) _audioBuf[s] = sourceAudio[0, s];

        // ── uncond audioMask：生成区 true，其余 false ──
        for (int s = 0; s < sequenceLength; s++) _audioBuf[sequenceLength + s] = s < targetLength;

        // ── attn mask（缓存：S 和 targetLen 不变时跳过重建）──
        if (_cachedAttnS != sequenceLength || _cachedAttnTLen != targetLength)
        {
            _cachedAttnS = sequenceLength;
            _cachedAttnTLen = targetLength;

            int stride = sequenceLength * sequenceLength;

            // cond: 全 true
            for (int i = 0; i < stride; i++) _attnBuf[i] = true;

            // uncond: [0,targetLen)×[0,targetLen) true；pad 对角
            Array.Clear(_attnBuf, stride, stride);
            for (int i = 0; i < targetLength; i++)
                for (int j = 0; j < targetLength; j++)
                    _attnBuf[stride + i * sequenceLength + j] = true;
            for (int s = targetLength; s < sequenceLength; s++)
                _attnBuf[stride + s * sequenceLength + s] = true;
        }
    }

    /// <summary>
    /// 构造单 batch 输入（无 CFG 模式）
    /// </summary>
    void FillSingleBatch(long[,,] sourceIds, bool[,] sourceAudio, int sequenceLength)
    {
        int codebookStride = NUM_CODEBOOKS * sequenceLength;
        for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
        {
            int sourceOffset = (codebook * sequenceLength) * sizeof(long);
            int destOffset = (codebook * sequenceLength) * sizeof(long);
            Buffer.BlockCopy(sourceIds, sourceOffset, _idsBuf, destOffset, sequenceLength * sizeof(long));
        }
        for (int s = 0; s < sequenceLength; s++) _audioBuf[s] = sourceAudio[0, s];

        // single batch attn: 全 true
        int stride = sequenceLength * sequenceLength;
        for (int i = 0; i < stride; i++) _attnBuf[i] = true;
    }

    /// <summary>
    /// 填充 position IDs 缓冲区
    /// </summary>
    void FillPosBuf(long[,] positionIds, int batchSize, int sequenceLength)
    {
        Buffer.BlockCopy(positionIds, 0, _posBuf, 0, batchSize * sequenceLength * sizeof(long));
    }

    // ════════════════════════════════════════════════════════════════
    // DiffusionStep — 扩散采样步骤
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 执行单步扩散采样：预测 token → 计算分数 → Top-K 选择 → 解 mask
    /// 【方案 C】Top-K 部分选择（自定义 MinHeap 替代全排序）
    /// </summary>
    /// <param name="inputIds">当前 token IDs（原地修改）</param>
    /// <param name="logProbabilities">LogSoftmax 后的概率</param>
    /// <param name="generateStart">生成段起始位置</param>
    /// <param name="targetLength">目标生成长度</param>
    /// <param name="sequenceLength">序列总长度 S</param>
    /// <param name="newUnmaskCount">本轮要解 mask 的 token 数</param>
    /// <returns>实际解 mask 的 token 数</returns>
    int DiffusionStep(
        long[,,] inputIds, float[] logProbabilities,
        int generateStart, int targetLength, int sequenceLength, int newUnmaskCount)
    {
        int codebookStride = sequenceLength * VOCAB_SIZE;

        // ════════════════════════════════════════════════════════════
        // 1. 预测每个位置的 token 并计算置信度分数
        // ════════════════════════════════════════════════════════════
        for (int t = 0; t < targetLength; t++)
        {
            int position = generateStart + t;
            int baseIndex = t * NUM_CODEBOOKS;

            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
            {
                int offset = codebook * codebookStride + position * VOCAB_SIZE;

                // 采样或 argmax 预测 token
                _predTokensBuf[baseIndex + codebook] = ClassTemperature > 0f
                    ? SampleTokenTopKRatio(logProbabilities, offset, 0.1f, ClassTemperature)
                    : ArgmaxToken(logProbabilities, offset);

                // 计算最大 log-prob 作为置信度
                float bestScore = float.NegativeInfinity;
                for (int v = 0; v < VOCAB_SIZE; v++)
                {
                    float logProb = logProbabilities[offset + v];
                    if (logProb > bestScore) bestScore = logProb;
                }
                _scoresBuf[baseIndex + codebook] = bestScore;
            }
        }

        // ════════════════════════════════════════════════════════════
        // 2. Layer Penalty：逐层递减分数（鼓励从低到高解 mask）
        // ════════════════════════════════════════════════════════════
        for (int t = 0; t < targetLength; t++)
            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
                _scoresBuf[t * NUM_CODEBOOKS + codebook] -= codebook * LayerPenaltyFactor;

        // ════════════════════════════════════════════════════════════
        // 3. Gumbel 噪声（Position Temperature > 0 时）
        // ════════════════════════════════════════════════════════════
        if (PositionTemperature > 0f)
        {
            float inverseTemperature = 1f / PositionTemperature;
            for (int t = 0; t < targetLength; t++)
                for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
                {
                    double uniform = Math.Max(1e-10, _rng.NextDouble());
                    _scoresBuf[t * NUM_CODEBOOKS + codebook] =
                        (float)(_scoresBuf[t * NUM_CODEBOOKS + codebook] * inverseTemperature
                                - Math.Log(-Math.Log(uniform)));
                }
        }

        // ════════════════════════════════════════════════════════════
        // 4. 收集待解 mask 的位置
        // ════════════════════════════════════════════════════════════
        int totalMasked = 0;
        int candidateIndex = 0;
        for (int t = 0; t < targetLength; t++)
            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
            {
                int index = t * NUM_CODEBOOKS + codebook;
                if (inputIds[0, codebook, generateStart + t] != MASK_TOKEN)
                    _scoresBuf[index] = float.NegativeInfinity;  // 已解 mask 的位置排除
                else
                    _allScoresBuf[candidateIndex++] = (t, codebook, _scoresBuf[index]);
                totalMasked += (inputIds[0, codebook, generateStart + t] == MASK_TOKEN) ? 1 : 0;
            }

        if (totalMasked == 0) return 0;
        newUnmaskCount = Math.Min(newUnmaskCount, totalMasked);

        // ════════════════════════════════════════════════════════════
        // 5. Top-K 选择并解 mask
        // ════════════════════════════════════════════════════════════
        if (newUnmaskCount < totalMasked / 3)
        {
            // ★ 方案 C：最小堆 Top-K 选择（O(n log k)，k 远小于 n 时优势显著）
            var heap = new MinHeap(_allScoresBuf, newUnmaskCount);
            for (int i = 0; i < totalMasked; i++)
            {
                var item = _allScoresBuf[i];
                if (heap.Count < newUnmaskCount)
                    heap.Add(item);
                else if (item.score > heap.Peek().score)
                    heap.ReplaceTop(item);
            }
            // 出堆应用
            for (int i = 0; i < newUnmaskCount; i++)
            {
                var item = heap.Pop();
                inputIds[0, item.cb, generateStart + item.t] = _predTokensBuf[item.t * NUM_CODEBOOKS + item.cb];
            }
        }
        else
        {
            // k 接近 n 时全排序更快
            Array.Sort(_allScoresBuf, 0, totalMasked,
                Comparer<(int, int, float)>.Create((a, b) => b.Item3.CompareTo(a.Item3)));
            for (int i = 0; i < newUnmaskCount; i++)
            {
                var (t, cb, _) = _allScoresBuf[i];
                inputIds[0, cb, generateStart + t] = _predTokensBuf[t * NUM_CODEBOOKS + cb];
            }
        }

        return newUnmaskCount;
    }

    // ════════════════════════════════════════════════════════════════
    // MinHeap — 自定义小根堆（零 GC，兼容 Unity .NET Standard 2.1）
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 小根堆结构 —— 用于 Top-K 选择，零 GC 分配
    /// 复用外部数组的前 capacity 个位置作为堆存储
    /// </summary>
    struct MinHeap
    {
        private readonly (int t, int cb, float score)[] _data;
        private readonly int _capacity;
        private int _count;

        /// <summary>
        /// 构造最小堆
        /// </summary>
        /// <param name="source">源数组（复用其前 capacity 个位置）</param>
        /// <param name="capacity">堆容量（即 K 值）</param>
        public MinHeap((int, int, float)[] source, int capacity)
        {
            _data = source;
            _capacity = capacity;
            _count = 0;
        }

        /// <summary>当前堆中元素数量</summary>
        public int Count => _count;

        /// <summary>查看堆顶元素（最小值）</summary>
        public (int t, int cb, float score) Peek() => _data[0];

        /// <summary>添加元素</summary>
        public void Add((int t, int cb, float score) item)
        {
            _data[_count] = item;
            SiftUp(_count);
            _count++;
        }

        /// <summary>替换堆顶元素（新元素必须大于原堆顶）</summary>
        public void ReplaceTop((int t, int cb, float score) item)
        {
            _data[0] = item;
            SiftDown(0);
        }

        /// <summary>弹出堆顶元素</summary>
        public (int t, int cb, float score) Pop()
        {
            var top = _data[0];
            _count--;
            _data[0] = _data[_count];
            SiftDown(0);
            return top;
        }

        /// <summary>上浮调整</summary>
        private void SiftUp(int index)
        {
            while (index > 0)
            {
                int parent = (index - 1) >> 1;
                if (_data[index].score >= _data[parent].score) break;
                Swap(index, parent);
                index = parent;
            }
        }

        /// <summary>下沉调整</summary>
        private void SiftDown(int index)
        {
            while (true)
            {
                int left = (index << 1) + 1;
                int right = left + 1;
                int smallest = index;

                if (left < _count && _data[left].score < _data[smallest].score) smallest = left;
                if (right < _count && _data[right].score < _data[smallest].score) smallest = right;
                if (smallest == index) break;

                Swap(index, smallest);
                index = smallest;
            }
        }

        /// <summary>交换元素</summary>
        private void Swap(int i, int j)
        {
            var temp = _data[i];
            _data[i] = _data[j];
            _data[j] = temp;
        }
    }

    // ════════════════════════════════════════════════════════════════
    // FinalUnmaskAll — 强制解剩余 mask（兜底）
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 最终强制解 mask：对仍未解的位置执行 argmax
    /// </summary>
    void FinalUnmaskAll(
        long[,,] inputIds, bool[,] audioMask,
        int sequenceLength, int generateStart, int targetLength, int refLength, int textLength)
    {
        int maskCount = 0;
        for (int t = 0; t < targetLength; t++)
            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
                if (inputIds[0, codebook, generateStart + t] == MASK_TOKEN) maskCount++;

        if (maskCount == 0) return;

        Debug.Log($"[OmniVoiceLM] 最终强制解 mask: 残余 {maskCount} 个");
        float[] logProbabilities = LMForwardWithCFG(
            inputIds, audioMask, sequenceLength, generateStart, refLength, textLength, targetLength);

        int codebookStride = sequenceLength * VOCAB_SIZE;
        for (int t = 0; t < targetLength; t++)
        {
            int position = generateStart + t;
            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
            {
                if (inputIds[0, codebook, position] != MASK_TOKEN) continue;
                int offset = codebook * codebookStride + position * VOCAB_SIZE;
                inputIds[0, codebook, position] = ArgmaxToken(logProbabilities, offset);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Token 采样
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// Argmax 采样：选择概率最高的 token
    /// </summary>
    /// <param name="logProbabilities">LogSoftmax 后的概率数组</param>
    /// <param name="baseOffset">起始偏移量</param>
    /// <returns>概率最高的 token ID</returns>
    long ArgmaxToken(float[] logProbabilities, int baseOffset)
    {
        float bestScore = float.NegativeInfinity;
        long bestToken = 0;
        for (int v = 0; v < VOCAB_SIZE; v++)
        {
            float logProb = logProbabilities[baseOffset + v];
            if (logProb > bestScore) { bestScore = logProb; bestToken = v; }
        }
        return bestToken;
    }

    /// <summary>
    /// Top-K Ratio + Gumbel 采样
    /// </summary>
    /// <param name="logProbabilities">LogSoftmax 后的概率数组</param>
    /// <param name="baseOffset">起始偏移量</param>
    /// <param name="ratio">Top-K 比例（如 0.1 = 前 10%）</param>
    /// <param name="temperature">采样温度</param>
    /// <returns>采样的 token ID</returns>
    long SampleTokenTopKRatio(float[] logProbabilities, int baseOffset, float ratio, float temperature)
    {
        int topK = (int)Math.Ceiling(ratio * VOCAB_SIZE);

        // 复制并排序
        for (int v = 0; v < VOCAB_SIZE; v++) _entriesBuf[v] = (logProbabilities[baseOffset + v], v);
        Array.Sort(_entriesBuf, (a, b) => b.score.CompareTo(a.score));

        // 构建过滤后的 log-prob 数组
        for (int v = 0; v < VOCAB_SIZE; v++) _filteredBuf[v] = float.NegativeInfinity;
        for (int i = 0; i < topK; i++) _filteredBuf[_entriesBuf[i].idx] = _entriesBuf[i].score;

        // Gumbel 噪声
        for (int v = 0; v < VOCAB_SIZE; v++)
        {
            if (float.IsNegativeInfinity(_filteredBuf[v])) continue;
            double uniform = Math.Max(1e-10, _rng.NextDouble());
            _filteredBuf[v] = (float)(_filteredBuf[v] / temperature - Math.Log(-Math.Log(uniform)));
        }

        // 选择最大值
        float bestScore = float.NegativeInfinity;
        long bestToken = 0;
        for (int v = 0; v < VOCAB_SIZE; v++)
            if (_filteredBuf[v] > bestScore) { bestScore = _filteredBuf[v]; bestToken = v; }
        return bestToken;
    }

    // ════════════════════════════════════════════════════════════════
    // LogSoftmax
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 计算 LogSoftmax（指定源和目标偏移）
    /// </summary>
    /// <param name="source">源数组</param>
    /// <param name="sourceOffset">源起始偏移</param>
    /// <param name="length">计算长度</param>
    /// <param name="destination">目标数组</param>
    /// <param name="destinationOffset">目标起始偏移</param>
    static void LogSoftmaxSlice(float[] source, int sourceOffset, int length, float[] destination, int destinationOffset)
    {
        // 1. 找最大值（数值稳定性）
        float maxValue = float.NegativeInfinity;
        for (int i = 0; i < length; i++)
        {
            float value = source[sourceOffset + i];
            if (value > maxValue) maxValue = value;
        }

        // 2. 处理全 -Inf 情况
        if (float.IsInfinity(maxValue) || float.IsNaN(maxValue))
        {
            for (int i = 0; i < length; i++) destination[destinationOffset + i] = float.NegativeInfinity;
            return;
        }

        // 3. 计算 log-sum-exp
        float sumExp = 0f;
        for (int i = 0; i < length; i++) sumExp += MathF.Exp(source[sourceOffset + i] - maxValue);
        float logSumExp = maxValue + MathF.Log(sumExp);

        // 4. 计算 log-softmax
        for (int i = 0; i < length; i++) destination[destinationOffset + i] = source[sourceOffset + i] - logSumExp;
    }

    /// <summary>
    /// 计算 LogSoftmax（写入目标数组起始位置）</summary>
    static void LogSoftmaxSlice(float[] source, int sourceOffset, int length, float[] destination)
        => LogSoftmaxSlice(source, sourceOffset, length, destination, 0);

    /// <summary>
    /// 原地 LogSoftmax（工作区 → 目标）
    /// </summary>
    /// <param name="work">工作区数组</param>
    /// <param name="length">计算长度</param>
    /// <param name="destination">目标数组</param>
    /// <param name="destinationOffset">目标起始偏移</param>
    static void LogSoftmaxSliceSelf(float[] work, int length, float[] destination, int destinationOffset)
    {
        float maxValue = float.NegativeInfinity;
        for (int i = 0; i < length; i++) if (work[i] > maxValue) maxValue = work[i];

        if (float.IsInfinity(maxValue) || float.IsNaN(maxValue))
        {
            for (int i = 0; i < length; i++) destination[destinationOffset + i] = float.NegativeInfinity;
            return;
        }

        float sumExp = 0f;
        for (int i = 0; i < length; i++) sumExp += MathF.Exp(work[i] - maxValue);
        float logSumExp = maxValue + MathF.Log(sumExp);

        for (int i = 0; i < length; i++) destination[destinationOffset + i] = work[i] - logSumExp;
    }

    // ════════════════════════════════════════════════════════════════
    // 辅助：缓冲区分配
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 确保主缓冲区足够大
    /// </summary>
    /// <param name="sequenceLength">序列长度 S</param>
    /// <param name="batchSize">批次大小</param>
    void EnsureBuffers(int sequenceLength, int batchSize)
    {
        int idsSize = batchSize * NUM_CODEBOOKS * sequenceLength;
        int audioSize = batchSize * sequenceLength;
        int attnSize = batchSize * sequenceLength * sequenceLength;
        int posSize = batchSize * sequenceLength;
        int rawLogitsSize = batchSize * NUM_CODEBOOKS * sequenceLength * VOCAB_SIZE;
        int resultSize = NUM_CODEBOOKS * sequenceLength * VOCAB_SIZE;

        bool rebuild = false;

        if (_idsBuf == null || _idsBuf.Length < idsSize) { _idsBuf = new long[idsSize]; rebuild = true; }
        if (_audioBuf == null || _audioBuf.Length < audioSize) { _audioBuf = new bool[audioSize]; rebuild = true; }
        if (_attnBuf == null || _attnBuf.Length < attnSize) { _attnBuf = new bool[attnSize]; rebuild = true; }
        if (_posBuf == null || _posBuf.Length < posSize) { _posBuf = new long[posSize]; rebuild = true; }
        if (_rawLogitsBuf == null || _rawLogitsBuf.Length < rawLogitsSize)
            _rawLogitsBuf = new float[rawLogitsSize];
        if (_resultBuf == null || _resultBuf.Length < resultSize)
            _resultBuf = new float[resultSize];

        // buffer 扩容时强制重建 Tensor（使 _tensorS 失效）
        if (rebuild) _tensorS = -1;
    }

    /// <summary>
    /// 确保扩散步骤缓冲区足够大
    /// </summary>
    /// <param name="targetLength">目标生成长度</param>
    void EnsureStepBuffers(int targetLength)
    {
        int needed = targetLength * NUM_CODEBOOKS;
        if (_allScoresBuf == null || _allScoresBuf.Length < needed) _allScoresBuf = new (int, int, float)[needed];
        if (_predTokensBuf == null || _predTokensBuf.Length < needed) _predTokensBuf = new long[needed];
        if (_scoresBuf == null || _scoresBuf.Length < needed) _scoresBuf = new float[needed];
    }

    /// <summary>
    /// 【方案 D 辅助】分配并行 LogSoftmax 工作区
    /// </summary>
    void EnsureParallelLsmWorkBuffers()
    {
        if (_lsmWorkPar == null)
        {
            _lsmWorkPar = new float[NUM_CODEBOOKS][];
            _lsmWork2Par = new float[NUM_CODEBOOKS][];
            for (int i = 0; i < NUM_CODEBOOKS; i++)
            {
                _lsmWorkPar[i] = new float[VOCAB_SIZE];
                _lsmWork2Par[i] = new float[VOCAB_SIZE];
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 辅助：BuildPositionIds（缓存）
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 构建位置 IDs（带缓存）
    /// </summary>
    /// <param name="batchSize">批次大小</param>
    /// <param name="sequenceLength">序列长度</param>
    /// <returns>位置 IDs [batchSize, sequenceLength]</returns>
    long[,] BuildPositionIds(int batchSize, int sequenceLength)
    {
        if (_cachedPosS == sequenceLength)
        {
            if (batchSize == 1 && _cachedPosIds1 != null) return _cachedPosIds1;
            if (batchSize == 2 && _cachedPosIds2 != null) return _cachedPosIds2;
        }

        var positions = new long[batchSize, sequenceLength];
        for (int b = 0; b < batchSize; b++)
            for (int s = 0; s < sequenceLength; s++)
                positions[b, s] = s;

        _cachedPosS = sequenceLength;
        if (batchSize == 1) _cachedPosIds1 = positions;
        if (batchSize == 2) _cachedPosIds2 = positions;
        return positions;
    }

    /// <summary>
    /// 检测数组是否包含过多 NaN/Inf
    /// </summary>
    /// <param name="array">待检测数组</param>
    /// <returns>损坏比例超过 1% 返回 true</returns>
    bool IsCorrupted(float[] array)
    {
        int badCount = 0;
        for (int i = 0; i < array.Length; i++)
            if (float.IsNaN(array[i]) || float.IsInfinity(array[i])) badCount++;
        return badCount > array.Length / 100;
    }

    // ════════════════════════════════════════════════════════════════
    // Dispose
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 释放 ONNX 会话资源
    /// </summary>
    public void Dispose() => _session?.Dispose();
}

/// <summary>
/// 执行提供程序类型
/// </summary>
public enum ExecutionProviderType
{
    /// <summary>CPU 多线程</summary>
    CPU,
    /// <summary>NVIDIA CUDA GPU</summary>
    CUDA,
    /// <summary>DirectML (Windows/AMD/Intel)</summary>
    DML
}
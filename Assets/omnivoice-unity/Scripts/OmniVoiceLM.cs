using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
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
///
/// 性能优化清单：
///   [OPT-1] 消除 LMForward 中 ToArray() 的每步 GC 分配（反射获取 DenseTensor 底层数组）
///   [OPT-2] 跳过 CFG 融合后的冗余 LogSoftmax（单调变换不影响 argmax / Gumbel 采样）
///   [OPT-3] 融合 Argmax + 置信度计算到 LogSoftmax 并行循环（消除 DiffusionStep 重复遍历）
///   [OPT-4] IsCorrupted 改为采样检测（检查量降低 ~99%）
///   [OPT-5] 一维数组替代多维数组（消除边界检查开销）
///   [OPT-6] SampleTokenTopKRatio 用 QuickSelect 替代全排序（O(n) 替代 O(n·log n)）
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

    /// <summary>LogSoftmax / CFG 融合结果缓冲区 [NUM_CODEBOOKS * S * VOCAB_SIZE]</summary>
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
    // [OPT-1] 反射缓存：DenseTensor 底层数组字段
    // ════════════════════════════════════════════════════════════════

    /// <summary>缓存的 DenseTensor 内部 float[] 字段（如果存在）</summary>
    static FieldInfo _denseTensorArrayField;

    /// <summary>是否已搜索过反射字段</summary>
    static bool _denseTensorFieldSearched;

    // ════════════════════════════════════════════════════════════════
    // LogSoftmax 并行工作区
    // ════════════════════════════════════════════════════════════════

    /// <summary>并行 LogSoftmax 工作区 [NUM_CODEBOOKS][VOCAB_SIZE]</summary>
    float[][] _lsmWorkPar;

    /// <summary>并行 LogSoftmax 暂存 [NUM_CODEBOOKS][VOCAB_SIZE]</summary>
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
    // [OPT-3] 融合缓冲区：在 LogSoftmax 循环中同时输出 argmax + 置信度
    // ════════════════════════════════════════════════════════════════

    /// <summary>融合预测 token [NUM_CODEBOOKS * targetLen]</summary>
    long[] _fusedPredBuf;

    /// <summary>融合置信度分数 [NUM_CODEBOOKS * targetLen]</summary>
    float[] _fusedScoreBuf;

    // ════════════════════════════════════════════════════════════════
    // Token 采样复用缓冲
    // ════════════════════════════════════════════════════════════════

    /// <summary>Top-K / QuickSelect 条目缓冲区</summary>
    readonly (float score, int idx)[] _entriesBuf = new (float, int)[VOCAB_SIZE];

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
        options.EnableMemoryPattern = false;   // 动态 shape 场景关闭
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
                    { "device_id",                 deviceId.ToString() },
                    { "arena_extend_strategy",     "kSameAsRequested"  },
                    { "do_copy_in_default_stream", "1"                 },
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

        // 创建 Tensor 和输入列表（修复原始代码中 WarmUp 未初始化 _inputList 的问题）
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
        EnsureFusedBuffers(targetLen);   // [OPT-3]

        Debug.Log($"[OmniVoiceLM] 开始扩散: T_text={textLength} T_ref={refLength} " +
                  $"T_gen={targetLen} S={sequenceLength} steps={NumStep} GS={GuidanceScale}");

        // ════════════════════════════════════════════════════════════
        // [OPT-5] 构建 inputIds（一维）和 audioMask（一维）
        // ════════════════════════════════════════════════════════════
        var inputIds = new long[NUM_CODEBOOKS * sequenceLength];
        var audioMask = new bool[sequenceLength];

        // 文本段：所有 codebook 共享同一 token，audioMask=false
        for (int s = 0; s < textLength; s++)
        {
            long tokenId = textTokenIds[s];
            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
                inputIds[codebook * sequenceLength + s] = tokenId;
            audioMask[s] = false;
        }

        // 参考音频段：填入 refCodes，audioMask=true
        for (int t = 0; t < refLength; t++)
        {
            int s = textLength + t;
            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
                inputIds[codebook * sequenceLength + s] = Math.Clamp(refCodes[codebook, t], 0, MASK_TOKEN - 1);
            audioMask[s] = true;
        }

        // 生成段：初始化为 MASK_TOKEN，audioMask=true
        for (int t = 0; t < targetLen; t++)
        {
            int s = generateStart + t;
            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
                inputIds[codebook * sequenceLength + s] = MASK_TOKEN;
            audioMask[s] = true;
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
                newUnmaskCount = remainingMasks;   // 最后一步全部解完
            else
            {
                newUnmaskCount = (int)Math.Round((schedule[step + 1] - schedule[step]) * totalMaskCount);
                newUnmaskCount = Math.Min(newUnmaskCount, remainingMasks);
            }
            if (newUnmaskCount <= 0) continue;

            // 执行 LM Forward + CFG + 融合 Argmax
            var timestamp = System.Diagnostics.Stopwatch.GetTimestamp();
            float[] logProbabilities = LMForwardWithCFG(
                inputIds, audioMask, sequenceLength, generateStart, refLength, textLength, targetLen);
            forwardMs += (System.Diagnostics.Stopwatch.GetTimestamp() - timestamp)
                         * 1000 / System.Diagnostics.Stopwatch.Frequency;

            // [OPT-4] NaN/Inf 采样检测与恢复
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
                long value = inputIds[codebook * sequenceLength + s];
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
    /// [OPT-2] 跳过 CFG 融合后的冗余 LogSoftmax
    /// [OPT-3] 融合 Argmax + 置信度到并行循环
    /// </summary>
    /// <param name="inputIds">[OPT-5] 一维输入 token IDs [NUM_CODEBOOKS * S]</param>
    /// <param name="audioMask">[OPT-5] 一维音频掩码 [S]</param>
    /// <param name="sequenceLength">序列总长度 S</param>
    /// <param name="generateStart">生成段起始位置</param>
    /// <param name="refLength">参考音频长度</param>
    /// <param name="textLength">文本长度</param>
    /// <param name="targetLen">目标生成长度</param>
    /// <returns>融合分数缓冲区 [NUM_CODEBOOKS * S * VOCAB_SIZE]</returns>
    float[] LMForwardWithCFG(
        long[] inputIds, bool[] audioMask,
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

            // ★ 并行 LogSoftmax + CFG 融合 + Argmax（按 codebook 维度并行）
            EnsureParallelLsmWorkBuffers();

            Parallel.For(0, NUM_CODEBOOKS, codebookIndex =>
            {
                var workBuffer = _lsmWorkPar[codebookIndex];
                var workBuffer2 = _lsmWork2Par[codebookIndex];

                for (int t = 0; t < targetLen; t++)
                {
                    int conditionalPosition = generateStart + t;
                    int unconditionalPosition = t;

                    int condOffset = codebookIndex * codebookStride
                                     + conditionalPosition * VOCAB_SIZE;
                    int uncondOffset = batchStride
                                     + codebookIndex * codebookStride
                                     + unconditionalPosition * VOCAB_SIZE;

                    // 分别计算 cond 和 uncond 的 log-softmax
                    LogSoftmaxSlice(_rawLogitsBuf, condOffset, VOCAB_SIZE, workBuffer2);
                    LogSoftmaxSlice(_rawLogitsBuf, uncondOffset, VOCAB_SIZE, workBuffer);

                    int resultOffset = codebookIndex * codebookStride
                                     + conditionalPosition * VOCAB_SIZE;
                    int fusedIndex = codebookIndex * targetLen + t;

                    // ──────────────────────────────────────────────
                    // [OPT-2] CFG 融合 → 直接写入 _resultBuf
                    //         跳过第二次 LogSoftmax（单调变换不影响
                    //         argmax 和 Gumbel-max 采样的正确性）
                    // [OPT-3] 同时计算 argmax + 置信度
                    // ──────────────────────────────────────────────
                    float bestScore = float.NegativeInfinity;
                    long bestToken = 0;

                    for (int v = 0; v < VOCAB_SIZE; v++)
                    {
                        float score = workBuffer2[v]
                                    + GuidanceScale * (workBuffer2[v] - workBuffer[v]);
                        _resultBuf[resultOffset + v] = score;

                        // 跳过 MASK token 的 argmax 比较
                        if (v != MASK_TOKEN && score > bestScore)
                        {
                            bestScore = score;
                            bestToken = v;
                        }
                    }

                    _resultBuf[resultOffset + MASK_TOKEN] = float.NegativeInfinity;

                    // 写入融合缓冲区
                    _fusedPredBuf[fusedIndex] = bestToken;
                    _fusedScoreBuf[fusedIndex] = bestScore;
                }
            });

            return _resultBuf;
        }
        else
        {
            // ════════════════════════════════════════════════════════
            // 无 CFG 模式：单 batch（同样并行化 + 融合）
            // ════════════════════════════════════════════════════════
            FillSingleBatch(inputIds, audioMask, sequenceLength);

            var positionIds = BuildPositionIds(1, sequenceLength);
            FillPosBuf(positionIds, 1, sequenceLength);
            LMForward(batchSize: 1, sequenceLength: sequenceLength, outBuf: _rawLogitsBuf);

            int codebookStride = sequenceLength * VOCAB_SIZE;

            EnsureParallelLsmWorkBuffers();

            Parallel.For(0, NUM_CODEBOOKS, codebookIndex =>
            {
                for (int s = 0; s < sequenceLength; s++)
                {
                    int offset = codebookIndex * codebookStride + s * VOCAB_SIZE;
                    LogSoftmaxSlice(_rawLogitsBuf, offset, VOCAB_SIZE, _resultBuf, offset);
                    _resultBuf[offset + MASK_TOKEN] = float.NegativeInfinity;

                    // [OPT-3] 仅对生成区域计算融合 argmax + 置信度
                    if (s >= generateStart && s < generateStart + targetLen)
                    {
                        int t = s - generateStart;
                        int fusedIndex = codebookIndex * targetLen + t;

                        float bestScore = float.NegativeInfinity;
                        long bestToken = 0;
                        for (int v = 0; v < MASK_TOKEN; v++)
                        {
                            float val = _resultBuf[offset + v];
                            if (val > bestScore) { bestScore = val; bestToken = v; }
                        }
                        _fusedPredBuf[fusedIndex] = bestToken;
                        _fusedScoreBuf[fusedIndex] = bestScore;
                    }
                }
            });

            return _resultBuf;
        }
    }

    // ════════════════════════════════════════════════════════════════
    // LMForward — ONNX 推理执行
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 执行 ONNX 推理并读取输出 logits
    /// [OPT-1] 通过反射获取 DenseTensor 底层 float[] 实现零中间分配；
    ///         反射失败时回退到 ToArray + BlockCopy（兼容所有 Unity ORT 版本）
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
        var logitsTensor = results[0].AsTensor<float>();
        int length = (int)logitsTensor.Length;

        if (outBuf.Length < length)
        {
            Debug.LogError($"[OmniVoiceLM] outBuf 太小 {outBuf.Length}<{length}");
            return;
        }

        // ── [OPT-1] 零分配拷贝 ──

        // 首次调用：遍历 DenseTensor<float> 所有非公开字段，找 float[] 类型
        if (!_denseTensorFieldSearched)
        {
            _denseTensorFieldSearched = true;
            foreach (var field in typeof(DenseTensor<float>).GetFields(
                         BindingFlags.NonPublic | BindingFlags.Instance))
            {
                if (field.FieldType == typeof(float[]))
                {
                    _denseTensorArrayField = field;
                    Debug.Log($"[OmniVoiceLM] 找到 DenseTensor 内部数组字段: {field.Name}");
                    break;
                }
            }
            if (_denseTensorArrayField == null)
                Debug.Log("[OmniVoiceLM] 未找到 DenseTensor 内部 float[] 字段，使用 ToArray 回退");
        }

        bool copied = false;

        // 路径 A：反射获取底层 float[] → 直接 BlockCopy（零 GC）
        if (_denseTensorArrayField != null)
        {
            if (_denseTensorArrayField.GetValue(logitsTensor) is float[] backingArray
                && backingArray.Length >= length)
            {
                Buffer.BlockCopy(backingArray, 0, outBuf, 0, length * sizeof(float));
                copied = true;
            }
        }

        // 路径 B：回退 — ToArray + BlockCopy（有 GC，但保证兼容）
        if (!copied)
        {
            var array = logitsTensor.ToArray();
            Buffer.BlockCopy(array, 0, outBuf, 0, length * sizeof(float));
        }
    }

    // ════════════════════════════════════════════════════════════════
    // FillCFGBatch — 构造 CFG 双 batch 输入
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 构造 CFG 双 batch 输入：cond(batch=0) + uncond(batch=1)
    /// [OPT-5] 源数据为一维数组
    /// </summary>
    void FillCFGBatch(
        long[] sourceIds, bool[] sourceAudio,
        int generateStart, int sequenceLength, int targetLength)
    {
        int codebookStride = NUM_CODEBOOKS * sequenceLength;

        // ── cond (batch=0) ids：整块复制 ──
        Array.Copy(sourceIds, 0, _idsBuf, 0, codebookStride);

        // ── uncond (batch=1) ids：生成区复制 + 尾部 MASK 填充 ──
        for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
        {
            int srcOffset = codebook * sequenceLength + generateStart;
            int dstOffset = codebookStride + codebook * sequenceLength;

            Array.Copy(sourceIds, srcOffset, _idsBuf, dstOffset, targetLength);

            for (int s = targetLength; s < sequenceLength; s++)
                _idsBuf[dstOffset + s] = MASK_TOKEN;
        }

        // ── cond audioMask ──
        Array.Copy(sourceAudio, 0, _audioBuf, 0, sequenceLength);

        // ── uncond audioMask：生成区 true，其余 false ──
        for (int s = 0; s < sequenceLength; s++)
            _audioBuf[sequenceLength + s] = s < targetLength;

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
    /// [OPT-5] 源数据为一维数组
    /// </summary>
    void FillSingleBatch(long[] sourceIds, bool[] sourceAudio, int sequenceLength)
    {
        Array.Copy(sourceIds, 0, _idsBuf, 0, NUM_CODEBOOKS * sequenceLength);
        Array.Copy(sourceAudio, 0, _audioBuf, 0, sequenceLength);

        // single batch attn: 全 true
        int stride = sequenceLength * sequenceLength;
        for (int i = 0; i < stride; i++) _attnBuf[i] = true;
    }

    /// <summary>
    /// 填充 position IDs 缓冲区
    /// </summary>
    void FillPosBuf(long[,] positionIds, int batchSize, int sequenceLength)
    {
        Buffer.BlockCopy(positionIds, 0, _posBuf, 0,
            batchSize * sequenceLength * sizeof(long));
    }

    // ════════════════════════════════════════════════════════════════
    // DiffusionStep — 扩散采样步骤
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 执行单步扩散采样：读取融合预测 → 计算分数 → Top-K 选择 → 解 mask
    /// [OPT-3] 直接从 _fusedPredBuf / _fusedScoreBuf 读取，跳过重复遍历
    /// [OPT-5] inputIds 为一维数组
    /// </summary>
    /// <param name="inputIds">[OPT-5] 一维 token IDs [NUM_CODEBOOKS * S]（原地修改）</param>
    /// <param name="logProbabilities">融合分数缓冲区</param>
    /// <param name="generateStart">生成段起始位置</param>
    /// <param name="targetLength">目标生成长度</param>
    /// <param name="sequenceLength">序列总长度 S</param>
    /// <param name="newUnmaskCount">本轮要解 mask 的 token 数</param>
    /// <returns>实际解 mask 的 token 数</returns>
    int DiffusionStep(
        long[] inputIds, float[] logProbabilities,
        int generateStart, int targetLength, int sequenceLength, int newUnmaskCount)
    {
        int codebookStride = sequenceLength * VOCAB_SIZE;

        // ════════════════════════════════════════════════════════════
        // 1. [OPT-3] 从融合缓冲区读取预测 token 和置信度
        //    greedy 模式：直接使用 _fusedPredBuf
        //    采样模式：仍需调用 SampleTokenTopKRatio（但 score 已预计算）
        // ════════════════════════════════════════════════════════════
        bool useSampling = ClassTemperature > 0f;

        for (int t = 0; t < targetLength; t++)
        {
            int baseIndex = t * NUM_CODEBOOKS;

            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
            {
                int fusedIndex = codebook * targetLength + t;

                // 置信度：始终从融合缓冲区读取（省去 1025 次比较）
                _scoresBuf[baseIndex + codebook] = _fusedScoreBuf[fusedIndex];

                // 预测 token
                if (useSampling)
                {
                    int offset = codebook * codebookStride
                               + (generateStart + t) * VOCAB_SIZE;
                    _predTokensBuf[baseIndex + codebook] =
                        SampleTokenTopKRatio(logProbabilities, offset, 0.1f, ClassTemperature);
                }
                else
                {
                    _predTokensBuf[baseIndex + codebook] = _fusedPredBuf[fusedIndex];
                }
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
                    int idx = t * NUM_CODEBOOKS + codebook;
                    double uniform = Math.Max(1e-10, _rng.NextDouble());
                    _scoresBuf[idx] = (float)(_scoresBuf[idx] * inverseTemperature
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
                bool isMasked = inputIds[codebook * sequenceLength + generateStart + t] == MASK_TOKEN;

                if (!isMasked)
                    _scoresBuf[index] = float.NegativeInfinity;   // 已解 mask 排除
                else
                    _allScoresBuf[candidateIndex++] = (t, codebook, _scoresBuf[index]);

                if (isMasked) totalMasked++;
            }

        if (totalMasked == 0) return 0;
        newUnmaskCount = Math.Min(newUnmaskCount, totalMasked);

        // ════════════════════════════════════════════════════════════
        // 5. Top-K 选择并解 mask
        // ════════════════════════════════════════════════════════════
        if (newUnmaskCount < totalMasked / 3)
        {
            // 最小堆 Top-K 选择（O(n log k)）
            var heap = new MinHeap(_allScoresBuf, newUnmaskCount);
            for (int i = 0; i < totalMasked; i++)
            {
                var item = _allScoresBuf[i];
                if (heap.Count < newUnmaskCount)
                    heap.Add(item);
                else if (item.score > heap.Peek().score)
                    heap.ReplaceTop(item);
            }

            for (int i = 0; i < newUnmaskCount; i++)
            {
                var item = heap.Pop();
                inputIds[item.cb * sequenceLength + generateStart + item.t] =
                    _predTokensBuf[item.t * NUM_CODEBOOKS + item.cb];
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
                inputIds[cb * sequenceLength + generateStart + t] =
                    _predTokensBuf[t * NUM_CODEBOOKS + cb];
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

        public MinHeap((int, int, float)[] source, int capacity)
        {
            _data = source;
            _capacity = capacity;
            _count = 0;
        }

        public int Count => _count;

        public (int t, int cb, float score) Peek() => _data[0];

        public void Add((int t, int cb, float score) item)
        {
            _data[_count] = item;
            SiftUp(_count);
            _count++;
        }

        public void ReplaceTop((int t, int cb, float score) item)
        {
            _data[0] = item;
            SiftDown(0);
        }

        public (int t, int cb, float score) Pop()
        {
            var top = _data[0];
            _count--;
            _data[0] = _data[_count];
            SiftDown(0);
            return top;
        }

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

        private void SiftDown(int index)
        {
            while (true)
            {
                int left = (index << 1) + 1;
                int right = left + 1;
                int smallest = index;

                if (left < _count && _data[left].score < _data[smallest].score)
                    smallest = left;
                if (right < _count && _data[right].score < _data[smallest].score)
                    smallest = right;
                if (smallest == index) break;

                Swap(index, smallest);
                index = smallest;
            }
        }

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
    /// [OPT-5] 一维数组索引
    /// </summary>
    void FinalUnmaskAll(
        long[] inputIds, bool[] audioMask,
        int sequenceLength, int generateStart, int targetLength,
        int refLength, int textLength)
    {
        int maskCount = 0;
        for (int t = 0; t < targetLength; t++)
            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
                if (inputIds[codebook * sequenceLength + generateStart + t] == MASK_TOKEN)
                    maskCount++;

        if (maskCount == 0) return;

        Debug.Log($"[OmniVoiceLM] 最终强制解 mask: 残余 {maskCount} 个");

        float[] logProbabilities = LMForwardWithCFG(
            inputIds, audioMask, sequenceLength, generateStart,
            refLength, textLength, targetLength);

        int codebookStride = sequenceLength * VOCAB_SIZE;

        for (int t = 0; t < targetLength; t++)
        {
            int position = generateStart + t;
            for (int codebook = 0; codebook < NUM_CODEBOOKS; codebook++)
            {
                if (inputIds[codebook * sequenceLength + position] != MASK_TOKEN) continue;

                int offset = codebook * codebookStride + position * VOCAB_SIZE;
                inputIds[codebook * sequenceLength + position] =
                    ArgmaxToken(logProbabilities, offset);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Token 采样
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// Argmax 采样：选择概率最高的 token
    /// </summary>
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
    /// [OPT-6] 用 QuickSelect O(n) 替代 Array.Sort O(n·log n)
    /// </summary>
    /// <param name="logProbabilities">分数数组</param>
    /// <param name="baseOffset">起始偏移量</param>
    /// <param name="ratio">Top-K 比例（如 0.1 = 前 10%）</param>
    /// <param name="temperature">采样温度</param>
    /// <returns>采样的 token ID</returns>
    long SampleTokenTopKRatio(float[] logProbabilities, int baseOffset,
                              float ratio, float temperature)
    {
        int topK = (int)Math.Ceiling(ratio * VOCAB_SIZE);

        // 1. 复制到工作区
        for (int v = 0; v < VOCAB_SIZE; v++)
            _entriesBuf[v] = (logProbabilities[baseOffset + v], v);

        // 2. [OPT-6] QuickSelect 找第 topK 大的阈值（O(n) 平均）
        float threshold = QuickSelectThreshold(_entriesBuf, VOCAB_SIZE, topK);

        // 3. 过滤 + Gumbel 噪声（只处理 >= threshold 的候选）
        float bestScore = float.NegativeInfinity;
        long bestToken = 0;

        for (int v = 0; v < VOCAB_SIZE; v++)
        {
            float score = logProbabilities[baseOffset + v];
            if (score < threshold) continue;

            double uniform = Math.Max(1e-10, _rng.NextDouble());
            float noisy = (float)(score / temperature - Math.Log(-Math.Log(uniform)));
            if (noisy > bestScore) { bestScore = noisy; bestToken = v; }
        }

        return bestToken;
    }

    /// <summary>
    /// [OPT-6] QuickSelect：找数组中第 k 大的 score 值
    /// 原地三路 partition，O(n) 平均时间复杂度
    /// </summary>
    static float QuickSelectThreshold((float score, int idx)[] entries, int length, int k)
    {
        int left = 0, right = length - 1;

        while (left < right)
        {
            // LCG 随机 pivot 避免退化
            _lcgSeed = _lcgSeed * 1103515245 + 12345;
            int pivotIndex = left + (int)((uint)_lcgSeed % (uint)(right - left + 1));
            float pivotValue = entries[pivotIndex].score;

            // 三路 partition（降序：> pivot | == pivot | < pivot）
            int lt = left, gt = right, i = left;
            while (i <= gt)
            {
                if (entries[i].score > pivotValue)
                    SwapEntries(entries, lt++, i++);
                else if (entries[i].score < pivotValue)
                    SwapEntries(entries, i, gt--);
                else
                    i++;
            }

            if (k <= lt)
                right = lt - 1;
            else if (k > gt + 1)
                left = gt + 1;
            else
                return pivotValue;
        }

        return entries[left].score;
    }

    /// <summary>QuickSelect 用 LCG 种子（避免每次 new Random）</summary>
    static int _lcgSeed = 42;

    static void SwapEntries((float, int)[] arr, int i, int j)
    {
        var tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    // ════════════════════════════════════════════════════════════════
    // LogSoftmax
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 计算 LogSoftmax（指定源和目标偏移）
    /// </summary>
    static void LogSoftmaxSlice(float[] source, int sourceOffset, int length,
                                float[] destination, int destinationOffset)
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
            for (int i = 0; i < length; i++)
                destination[destinationOffset + i] = float.NegativeInfinity;
            return;
        }

        // 3. 计算 log-sum-exp
        float sumExp = 0f;
        for (int i = 0; i < length; i++)
            sumExp += MathF.Exp(source[sourceOffset + i] - maxValue);
        float logSumExp = maxValue + MathF.Log(sumExp);

        // 4. 计算 log-softmax
        for (int i = 0; i < length; i++)
            destination[destinationOffset + i] = source[sourceOffset + i] - logSumExp;
    }

    /// <summary>
    /// 计算 LogSoftmax（写入目标数组起始位置）
    /// </summary>
    static void LogSoftmaxSlice(float[] source, int sourceOffset, int length,
                                float[] destination)
        => LogSoftmaxSlice(source, sourceOffset, length, destination, 0);

    // ════════════════════════════════════════════════════════════════
    // 辅助：缓冲区分配
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 确保主缓冲区足够大
    /// </summary>
    void EnsureBuffers(int sequenceLength, int batchSize)
    {
        int idsSize = batchSize * NUM_CODEBOOKS * sequenceLength;
        int audioSize = batchSize * sequenceLength;
        int attnSize = batchSize * sequenceLength * sequenceLength;
        int posSize = batchSize * sequenceLength;
        int rawLogitsSize = batchSize * NUM_CODEBOOKS * sequenceLength * VOCAB_SIZE;
        int resultSize = NUM_CODEBOOKS * sequenceLength * VOCAB_SIZE;

        bool rebuild = false;

        if (_idsBuf == null || _idsBuf.Length < idsSize)
        { _idsBuf = new long[idsSize]; rebuild = true; }

        if (_audioBuf == null || _audioBuf.Length < audioSize)
        { _audioBuf = new bool[audioSize]; rebuild = true; }

        if (_attnBuf == null || _attnBuf.Length < attnSize)
        { _attnBuf = new bool[attnSize]; rebuild = true; }

        if (_posBuf == null || _posBuf.Length < posSize)
        { _posBuf = new long[posSize]; rebuild = true; }

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
    void EnsureStepBuffers(int targetLength)
    {
        int needed = targetLength * NUM_CODEBOOKS;
        if (_allScoresBuf == null || _allScoresBuf.Length < needed)
            _allScoresBuf = new (int, int, float)[needed];
        if (_predTokensBuf == null || _predTokensBuf.Length < needed)
            _predTokensBuf = new long[needed];
        if (_scoresBuf == null || _scoresBuf.Length < needed)
            _scoresBuf = new float[needed];
    }

    /// <summary>
    /// [OPT-3] 确保融合缓冲区足够大
    /// </summary>
    void EnsureFusedBuffers(int targetLength)
    {
        int size = NUM_CODEBOOKS * targetLength;
        if (_fusedPredBuf == null || _fusedPredBuf.Length < size)
            _fusedPredBuf = new long[size];
        if (_fusedScoreBuf == null || _fusedScoreBuf.Length < size)
            _fusedScoreBuf = new float[size];
    }

    /// <summary>
    /// 分配并行 LogSoftmax 工作区
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

    // ════════════════════════════════════════════════════════════════
    // [OPT-4] IsCorrupted — 采样检测
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 检测数组是否包含过多 NaN/Inf
    /// [OPT-4] 采样检查（~4096 个点），替代全量遍历
    /// </summary>
    /// <param name="array">待检测数组</param>
    /// <returns>损坏比例超过 10% 返回 true</returns>
    bool IsCorrupted(float[] array)
    {
        int badCount = 0;
        int checkCount = 0;
        int step = Math.Max(1, array.Length / 4096);

        for (int i = 0; i < array.Length; i += step)
        {
            if (float.IsNaN(array[i]) || float.IsInfinity(array[i]))
                badCount++;
            checkCount++;
        }

        return checkCount > 0 && badCount > checkCount / 10;
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
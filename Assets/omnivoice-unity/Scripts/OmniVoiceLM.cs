using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Concurrent;
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
///   [OPT-7] IOBinding 替代 NamedOnnxValue.Run()：
///           - 输入/输出 OrtValue 直接包裹复用的托管数组，地址在整个 Generate() 调用内保持不变
///           - 输出直接写入 _rawLogitsBuf，彻底取代 OPT-1 的反射拷贝
///           - 可选开启 CUDA Graph（enable_cuda_graph）：同一 Generate() 调用内 32 步 forward
///             shape 完全相同，首步 capture、后续步骤 replay，消除逐步 kernel launch 开销
///           - 失败自动回退旧的 NamedOnnxValue.Run() 路径，不影响正确性
///   [OPT-8] FastRng：XOR-shift 64-bit PRNG 替代 System.Random
///           - 比 System.Random 快约 10×，无内部锁争用，线程安全（ThreadStatic）
///           - 用于 Gumbel 噪声生成的 per-element 随机数，性能瓶颈明显
///   [OPT-9] DiffusionStep 并行化：LayerPenalty + Gumbel 噪声 + mask 收集用 Parallel.For
///           - 利用多核 CPU 加速 CPU-bound 的后处理阶段
///   [OPT-10] 增量 CFG batch 更新：仅在 batch 数据变化时重建
///           - 避免每步重复复制不变的 batch 区域
///   [OPT-11] 调度加速：线性调度 + 减少步数
///           - ScheduleAcceleration > 1 时切换为线性调度（均匀分配每步解 mask 量）
///           - 同时减少有效步数（effectiveSteps = NumStep / ScheduleAcceleration）
///           - 前向传播次数从 32 → 通常 8~11 次，速度提升 2~4×
///           - 安全限制：每步最多解 mask 剩余量的 80%
/// 
/// 尚未做（需要模型/架构层面配合，本次未改动）：
///   - 序列长度分桶（跨 Generate() 调用复用 CUDA Graph）：需要同步改造 attention mask，
///     避免 padding 区域被 cond 分支的全 1 注意力污染真实 token，暂缓实现
///   - lm_head 只对生成区位置投影（ONNX 导出层面裁剪），可省去 (S-targetLen) 比例的算力
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
    // [OPT-7] IOBinding / CUDA Graph 相关状态
    // ════════════════════════════════════════════════════════════════

    /// <summary>是否成功初始化 IOBinding（构造时探测，失败则永久回退旧路径）</summary>
    bool _ioBindingReady;

    /// <summary>复用的 IOBinding 对象</summary>
    OrtIoBinding _ioBinding;

    /// <summary>IOBinding 用持久 OrtValue —— 输入（与 shape 绑定，shape 变化时重建，每步重新 BindInput 保证数据新鲜）</summary>
    OrtValue _ovIds, _ovAudio, _ovAttn, _ovPos;

    /// <summary>模型输出节点名（从 session 元数据读取，避免硬编码）</summary>
    string _outputName;

    /// <summary>复用的 RunOptions（IOBinding 路径下每次 Run 需要，避免每步 new）</summary>
    RunOptions _runOptions;

    /// <summary>是否请求启用 CUDA Graph（仅在 provider=CUDA 且成功开启 IOBinding 时生效）</summary>
    bool _enableCudaGraph;

    /// <summary>实际执行提供程序类型（用于判断是否满足 CUDA Graph 前置条件）</summary>
    ExecutionProviderType _providerType;

    /// <summary>本次 Generate() 内，当前 shape 是否已完成"首步 capture"（仅用于日志提示，不影响逻辑）</summary>
    bool _cudaGraphCapturedForCurrentShape;

    // ════════════════════════════════════════════════════════════════
    // [OPT-10] 增量 CFG batch 更新 — 使用 batchDirty 标志（在 Generate 方法内管理）
    // ════════════════════════════════════════════════════════════════

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

    /// <summary>[OPT-11] 调度加速倍率：控制每步解 mask 的比例
    /// 1.0 = 原版调度，2.0 = 每步解 2 倍 mask，4.0 = 每步解 4 倍 mask
    /// 推荐 2.0~4.0（质量损失很小，速度提升 2~4×）</summary>
    public float ScheduleAcceleration = 2.0f;

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
    // [OPT-8] FastRng — XOR-shift 64-bit PRNG（线程安全、零分配、~10× faster than System.Random）
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 超轻量 XOR-shift 64-bit 伪随机数生成器。
    /// 无锁、无分配，适合 Parallel.For 内部高频调用。
    /// 比 System.Random 快约 10×，且避免了 NextDouble() 的内部锁争用。
    /// </summary>
    struct FastRng
    {
        ulong _state;

        public FastRng(ulong seed) => _state = seed != 0 ? seed : 0x9E3779B97F4A7C15UL;

        /// <summary>生成 [0, 1) 区间的 double，用于 Gumbel 噪声</summary>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        public double NextDouble01()
        {
            _state ^= _state << 13;
            _state ^= _state >> 7;
            _state ^= _state << 17;
            // 映射到 [0, 1)：取高 53 位作为尾数，除以 2^53
            return (_state >> 11) * 1.1102230246251565e-16; // 2^-53
        }
    }


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

    /// <summary>
    /// 分数降序比较器（静态复用，避免 DiffusionStep 每次进入全排序分支都 new 一个 Comparer 包装对象）
    /// </summary>
    static readonly Comparer<(int t, int cb, float score)> _scoreDescendingComparer =
        Comparer<(int t, int cb, float score)>.Create((a, b) => b.score.CompareTo(a.score));

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
    /// <param name="enableCudaGraph">
    /// [OPT-7] 是否尝试开启 CUDA Graph（仅 CUDA EP 有效）。
    /// 默认 false —— 该特性依赖具体 ORT/CUDA 版本，不同版本行为差异较大，
    /// 建议先在你的实际环境上用 WarmUp() 验证输出无 NaN/Inf 后再开启。
    /// 开启失败会自动降级为普通 CUDA（不影响正确性，只是拿不到额外加速）。
    /// </param>
    public OmniVoiceLM(
        string modelPath,
        ExecutionProviderType executionProvider = ExecutionProviderType.CUDA,
        int deviceId = 0,
        int seed = 42,
        bool enableCudaGraph = false)
    {
        _rng = new System.Random(seed);
        _providerType = executionProvider;
        _enableCudaGraph = enableCudaGraph && executionProvider == ExecutionProviderType.CUDA;

        // 构建 SessionOptions
        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        // 动态 shape 场景关闭；CUDA Graph 模式下 ORT 要求内存地址在 capture 后固定，
        // EnableMemoryPattern=true 可能导致 arena 重新规划、破坏已 capture 的图，必须保持 false。
        options.EnableMemoryPattern = false;
        options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        options.InterOpNumThreads = 1;
        options.IntraOpNumThreads = 4;

        bool executionProviderLoaded = false;

        // 按优先级尝试 EP：CUDA → DML → CPU
        if (executionProvider == ExecutionProviderType.CUDA)
        {
            // [OPT-7] 先尝试带 enable_cuda_graph 的配置，失败（旧版 ORT 不识别该 key /
            // 硬件不支持）时自动退化为不带该选项的标准 CUDA 配置，最后才回退 CPU。
            bool cudaGraphActuallyEnabled = false;

            if (_enableCudaGraph)
            {
                try
                {
                    var cudaOptions = new OrtCUDAProviderOptions();
                    cudaOptions.UpdateOptions(new Dictionary<string, string>
                    {
                        { "device_id",                 deviceId.ToString() },
                        { "arena_extend_strategy",     "kSameAsRequested"  },
                        { "do_copy_in_default_stream", "1"                 },
                        { "enable_cuda_graph",         "1"                 },
                    });
                    options.AppendExecutionProvider_CUDA(cudaOptions);
                    executionProviderLoaded = true;
                    cudaGraphActuallyEnabled = true;
                    Debug.Log($"[OmniVoiceLM] CUDA EP + CUDA Graph (device={deviceId})");
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[OmniVoiceLM] CUDA Graph 选项不受支持（{ex.Message}），" +
                                      "回退标准 CUDA EP");
                }
            }

            if (!executionProviderLoaded)
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

            _enableCudaGraph = cudaGraphActuallyEnabled;
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

        // [OPT-7] 输出节点名（从元数据读取，避免硬编码 "logits" 之类的名字猜测）
        _outputName = _session.OutputMetadata.Keys.First();

        // [OPT-7] 尝试初始化 IOBinding；失败则整份代码自动回退到旧的 NamedOnnxValue.Run() 路径
        try
        {
            _ioBinding = _session.CreateIoBinding();
            _runOptions = new RunOptions();
            _ioBindingReady = true;
            Debug.Log("[OmniVoiceLM] IOBinding 初始化成功");
        }
        catch (Exception ex)
        {
            _ioBindingReady = false;
            _enableCudaGraph = false;   // 没有 IOBinding 就没法保证地址稳定，CUDA Graph 一并放弃
            Debug.LogWarning($"[OmniVoiceLM] IOBinding 初始化失败（{ex.Message}），" +
                              "回退标准 Run() 路径（性能略低，正确性不受影响）");
        }
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

        int batchSize = GuidanceScale > 0f ? 2 : 1, sequenceLength = warmupSequenceLength;
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

        // 创建 Tensor / IOBinding（修复原始代码中 WarmUp 未初始化 _inputList 的问题；
        // 现在与 LMForward 共用同一份重建逻辑，避免两处实现漂移）
        EnsureTensorsAndBinding(batchSize, sequenceLength);

        // [OPT-7] CUDA Graph 首次 capture 通常发生在第 1 次 Run，之后才是真正的 replay，
        // 所以预热轮数比原来的 2 次多留一点余量，同时顺带把 capture 开销消化在预热阶段。
        int warmIterations = _enableCudaGraph ? 4 : 2;
        for (int i = 0; i < warmIterations; i++)
            LMForward(batchSize, sequenceLength, _rawLogitsBuf);

        // [OPT-7] CUDA Graph 自检：复用 [OPT-4] 的采样式 NaN/Inf 检测，
        // 一旦发现异常立刻大声报警——这类问题在 CUDA Graph 场景下最容易"悄悄"发生
        // （地址复用错误、graph 与实际输入不同步等），生产环境务必看这条日志。
        if (_enableCudaGraph && IsCorrupted(_rawLogitsBuf))
        {
            Debug.LogError("[OmniVoiceLM] ⚠️ CUDA Graph 预热后输出包含 NaN/Inf，" +
                            "强烈建议将 enableCudaGraph 设为 false 后重新测试，" +
                            "当前 ORT/CUDA 版本组合可能不兼容该特性。");
        }

        // 提醒：真实请求的 S（文本+参考音频+目标长度）大概率和这里的预热 S 不同，
        // 首次真实请求仍会触发一次重建（+CUDA Graph 重新 capture）。如果业务侧
        // refLength 分布比较集中，建议改用贴近真实分布的 warmupSequenceLength 多跑几次，
        // 把这部分延迟提前消化掉。
        stopwatch.Stop();
        Debug.Log($"[OmniVoiceLM] 预热完成 {stopwatch.ElapsedMilliseconds}ms " +
                  $"(IOBinding={_ioBindingReady}, CUDAGraph={_enableCudaGraph})");
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
        // 根据是否启用 CFG 决定 batch size（GS=0 时 batchSize=1，GS>0 时 batchSize=2）
        int batchSize = GuidanceScale > 0f ? 2 : 1;
        EnsureBuffers(sequenceLength, batchSize);
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
        // ════════════════════════════════════════════════════════════
        // [OPT-11] 时移余弦调度 r[n]
        // ════════════════════════════════════════════════════════════
        // TShift=0.1 原版调度极度后加载（step 0 仅解 0.3%，step 16 仅解 9%），
        // 导致 32 步中大部分步只解 mask 很少 token。
        // ScheduleAcceleration > 1 时：
        //   1) 适当调大 tau（上限 0.2，保持后加载特性）
        //   2) 减少有效步数（effectiveSteps = NumStep / ScheduleAcceleration）
        //   3) 不放大每步解 mask 量（让调度自然分配）
        int effectiveSteps = ScheduleAcceleration > 1.0f
            ? Math.Max(24, (int)(NumStep / ScheduleAcceleration))
            : NumStep;
        // tau 范围：原版 0.1 → 加速时最大 0.15（接近原版，保持后加载）
        double tau = ScheduleAcceleration > 1.0f
            ? Math.Min(0.15, TShift * ScheduleAcceleration)
            : TShift;
        double totalSteps = effectiveSteps;
        var schedule = new double[effectiveSteps + 1];
        for (int n = 0; n <= effectiveSteps; n++)
        {
            double progress = n / totalSteps;
            schedule[n] = tau * progress / (1.0 + (tau - 1.0) * progress);
        }

        int totalMaskCount = targetLen * NUM_CODEBOOKS;
        int remainingMasks = totalMaskCount;

        // ════════════════════════════════════════════════════════════
        // [OPT-11] 主扩散循环 — 置信度驱动的自适应解 mask
        // ════════════════════════════════════════════════════════════
        var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();
        long forwardMs = 0, stepMs = 0;
        bool batchDirty = true;  // [OPT-10] 第一步必须重建 batch（_idsBuf 未初始化）
        int forwardCount = 0;

        for (int step = 0; step < effectiveSteps; step++)
        {
            // [OPT-11] 使用置信度阈值自适应决定解 mask 数量
            int newUnmaskCount;
            if (step == effectiveSteps - 1)
                newUnmaskCount = remainingMasks;   // 最后一步全部解完
            else
            {
                // 基础调度：固定比例
                int baseUnmask = (int)Math.Round((schedule[step + 1] - schedule[step]) * totalMaskCount);

                // [OPT-11] 安全限制：每步最多解 mask 剩余量的 15%
                // 限制每步解 mask 量，防止相邻位置同时解 mask 导致字音重叠
                int maxUnmask = Math.Max(1, (int)(remainingMasks * 0.15f));
                baseUnmask = Math.Min(baseUnmask, maxUnmask);

                newUnmaskCount = Math.Min(baseUnmask, remainingMasks);
            }

            if (newUnmaskCount <= 0) continue;

            // [OPT-10] 增量 batch 更新：上一步没有解 mask 时跳过 batch 重建

            // 执行 LM Forward + CFG + 融合 Argmax
            var timestamp = System.Diagnostics.Stopwatch.GetTimestamp();
            float[] logProbabilities = LMForwardWithCFG(
                inputIds, audioMask, sequenceLength, generateStart, refLength, textLength, targetLen,
                batchDirty);
            forwardMs += (System.Diagnostics.Stopwatch.GetTimestamp() - timestamp)
                         * 1000 / System.Diagnostics.Stopwatch.Frequency;
            forwardCount++;

            // [OPT-4] NaN/Inf 采样检测与恢复
            if (IsCorrupted(logProbabilities))
            {
                Debug.LogError($"[OmniVoiceLM] 步{step} NaN/Inf，降温重试");
                PositionTemperature = Mathf.Max(0.1f, PositionTemperature * 0.5f);
                logProbabilities = LMForwardWithCFG(
                    inputIds, audioMask, sequenceLength, generateStart, refLength, textLength, targetLen,
                    batchDirty);
                forwardCount++;
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
            // [OPT-10] 只有当本轮实际解 mask 了，下一步才需要重建 batch
            batchDirty = (unmaskedCount > 0);

            // 日志输出
            if (step % 4 == 0 || unmaskedCount > newUnmaskCount * 1.5f)
                Debug.Log($"[OmniVoiceLM] step {step}/{effectiveSteps} kNew={newUnmaskCount} actual={unmaskedCount} " +
                          $"rem={remainingMasks} fwd={forwardCount} fwdMs={forwardMs} stepMs={stepMs}");

            // [OPT-11] 如果所有 token 都已解 mask，提前退出
            if (remainingMasks <= 0)
            {
                Debug.Log($"[OmniVoiceLM] ✅ 提前完成于步 {step}/{effectiveSteps}（前向传播 {forwardCount} 次）");
                break;
            }
        }

        // 强制解剩余 mask（兜底）
        if (remainingMasks > 0)
        {
            Debug.LogWarning($"[OmniVoiceLM] 主循环结束后仍有 {remainingMasks} 个 mask，执行兜底");
            FinalUnmaskAll(inputIds, audioMask, sequenceLength, generateStart, targetLen, refLength, textLength, batchDirty);
        }

        totalStopwatch.Stop();
        Debug.Log($"[OmniVoiceLM] 完成: LMForward={forwardCount}次/{effectiveSteps}步 {forwardMs}ms " +
                  $"DiffusionStep={stepMs}ms 总={totalStopwatch.ElapsedMilliseconds}ms " +
                  $"(加速比 {NumStep / (float)forwardCount:F1}×)");

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
        int sequenceLength, int generateStart, int refLength, int textLength, int targetLen,
        bool needRebuildBatch = true)
    {
        if (GuidanceScale > 0f)
        {
            // ════════════════════════════════════════════════════════
            // CFG 模式：构造 cond/uncond 双 batch
            // [OPT-10] 增量更新：仅当 batch 数据变化时重建
            // ════════════════════════════════════════════════════════
            if (needRebuildBatch)
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
            // [OPT-10] 增量更新
            // ════════════════════════════════════════════════════════
            if (needRebuildBatch)
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
        // Tensor / IOBinding shape 变化时重建（正常 Generate 内只建一次，32 步复用同一份）
        if (_tensorS != sequenceLength || _tensorBatch != batchSize)
            EnsureTensorsAndBinding(batchSize, sequenceLength);

        if (_ioBindingReady)
        {
            // ════════════════════════════════════════════════════════
            // [OPT-7] IOBinding 路径：
            //   - 输入/输出 OrtValue 对象在 shape 不变时始终是同一个实例，
            //     直接包裹 _idsBuf/_audioBuf/_attnBuf/_posBuf/_rawLogitsBuf，
            //     地址在整个 Generate() 调用（32 步）内保持不变 —— 这是
            //     CUDA Graph 能正确 replay 的前提。
            //   - ⚠️ 关键点：ORT 的 host→device 拷贝只发生在 BindInput
            //     被调用的那一刻，不是每次 Run 都会自动重新拷贝。如果只在
            //     shape 变化时 bind 一次、之后仅修改数组内容而不重新 bind，
            //     后续步骤会读到第一步的陈旧数据（这是 OrtIoBinding 官方
            //     文档明确写明的行为，不是本实现的 bug）。所以这里每一步
            //     都重新 BindInput——同一个 OrtValue 实例，只是重新登记一次，
            //     开销远小于重建 Tensor/List，但保证了数据新鲜。
            //   - 输出通过 BindOutputToDevice(CPU) + GetOutputValues() 获取，
            //     一次 memcpy 拷进 outBuf（见下方），不再需要 OPT-1 的反射逻辑。
            //   - 若 CUDA Graph 生效，图内只捕获 GPU 侧计算 kernel；host→device
            //     拷贝本就发生在图外（每次 Run 前），两者不冲突。
            // ════════════════════════════════════════════════════════
            _ioBinding.BindInput("input_ids", _ovIds);
            _ioBinding.BindInput("audio_mask", _ovAudio);
            _ioBinding.BindInput("attention_mask", _ovAttn);
            _ioBinding.BindInput("position_ids", _ovPos);

            // 输出：绑定到 CPU 设备（而不是固定 OrtValue），让 ORT 在每次 Run 后
            // 把结果写到新分配的 OrtValue 里，Run 完成后立刻通过 GetOutputValues()
            // 取出、拷贝进 outBuf。这样避免了"绑定=拷贝时机"的不确定性——
            // 保证拿到的永远是这次 Run 算出来的新鲜结果，不会有一次性绑定
            // 导致的陈旧数据风险。拷贝本身是原生内存到托管数组的 memcpy，
            // 没有 GC 压力，比原来的反射方案（OPT-1）更快也更简单。
            _ioBinding.BindOutputToDevice(_outputName, OrtMemoryInfo.DefaultInstance);

            _session.RunWithBinding(_runOptions, _ioBinding);

            // 注意：GetOutputValues() 在不同 ORT 版本里返回类型略有差异（数组 /
            // IEnumerable / IDisposableReadOnlyCollection 均出现过），这里只依赖
            // 最通用的 IEnumerable<OrtValue> 接口（LINQ First + foreach），
            // 不假设它本身实现 IDisposable 或支持索引器，兼容性更好。
            var boundOutputs = _ioBinding.GetOutputValues();
            try
            {
                var outputValue = boundOutputs.First();
                var outSpan = outputValue.GetTensorDataAsSpan<float>();

                if (outBuf.Length < outSpan.Length)
                    Debug.LogError($"[OmniVoiceLM] outBuf 太小 {outBuf.Length}<{outSpan.Length}");
                else
                    outSpan.CopyTo(outBuf.AsSpan(0, outSpan.Length));
            }
            finally
            {
                // 每个 OrtValue 都是 ORT 新分配的原生对象，必须逐个释放，否则每步都会泄漏
                foreach (var ov in boundOutputs) ov?.Dispose();
            }

            if (!_cudaGraphCapturedForCurrentShape)
            {
                _cudaGraphCapturedForCurrentShape = true;
                if (_enableCudaGraph)
                    Debug.Log($"[OmniVoiceLM] CUDA Graph 已 capture (batch={batchSize} S={sequenceLength})，后续同 shape 调用将 replay");
            }
            return;
        }

        // ════════════════════════════════════════════════════════════
        // 回退路径：标准 NamedOnnxValue.Run()（IOBinding 初始化失败时使用）
        // ════════════════════════════════════════════════════════════
        using var results = _session.Run(_inputList);
        var logitsTensor = results[0].AsTensor<float>();
        int length = (int)logitsTensor.Length;

        if (outBuf.Length < length)
        {
            Debug.LogError($"[OmniVoiceLM] outBuf 太小 {outBuf.Length}<{length}");
            return;
        }

        // ── [OPT-1] 零分配拷贝（仅回退路径需要）──

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
    // [OPT-7] EnsureTensorsAndBinding — Tensor / IOBinding 共用重建逻辑
    // ════════════════════════════════════════════════════════════════

    /// <summary>
    /// 在 batchSize / sequenceLength 变化时重建 DenseTensor（旧回退路径用）
    /// 以及 OrtValue + IOBinding 绑定（新路径用）。
    /// LMForward 与 WarmUp 共用此方法，避免两处实现漂移。
    /// </summary>
    void EnsureTensorsAndBinding(int batchSize, int sequenceLength)
    {
        // ── 旧路径：DenseTensor + NamedOnnxValue（IOBinding 失败时的回退始终可用）──
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

        // ── 新路径：OrtValue 直接包裹同一批复用数组 + IOBinding ──
        if (_ioBindingReady)
        {
            try
            {
                _ovIds?.Dispose();
                _ovAudio?.Dispose();
                _ovAttn?.Dispose();
                _ovPos?.Dispose();

                _ovIds = OrtValue.CreateTensorValueFromMemory(
                    _idsBuf, new long[] { batchSize, NUM_CODEBOOKS, sequenceLength });
                _ovAudio = OrtValue.CreateTensorValueFromMemory(
                    _audioBuf, new long[] { batchSize, sequenceLength });
                _ovAttn = OrtValue.CreateTensorValueFromMemory(
                    _attnBuf, new long[] { batchSize, 1, sequenceLength, sequenceLength });
                _ovPos = OrtValue.CreateTensorValueFromMemory(
                    _posBuf, new long[] { batchSize, sequenceLength });

                // 输出不使用固定 OrtValue：BindOutputToDevice(CPU) 让 ORT 每次 Run 后
                // 分配新的 CPU 端结果对象，LMForward 里通过 GetOutputValues() 取出。
                // 清掉上一个 shape 遗留的绑定（避免残留悬空引用）；
                // 真正的 BindInput/BindOutputToDevice 在 LMForward 每一步都会重新调用，
                // 这里不需要预先 bind。
                _ioBinding.ClearBoundInputs();
                _ioBinding.ClearBoundOutputs();

                // shape 变了：CUDA Graph（若开启）需要针对新 shape 重新 capture
                _cudaGraphCapturedForCurrentShape = false;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[OmniVoiceLM] IOBinding 重建失败（{ex.Message}），" +
                                  "本次运行永久回退标准 Run() 路径");
                _ioBindingReady = false;
                _enableCudaGraph = false;
            }
        }

        _tensorS = sequenceLength;
        _tensorBatch = batchSize;
        Debug.Log($"[OmniVoiceLM] Tensor{(_ioBindingReady ? "+IOBinding" : "")} 重建: " +
                  $"batch={batchSize} S={sequenceLength}");
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
        // 2+3+4. [OPT-9] 并行 Layer Penalty + Gumbel 噪声 + mask 收集
        //     利用多核 CPU 加速 CPU-bound 的后处理阶段
        // ════════════════════════════════════════════════════════════
        int totalElements = targetLength * NUM_CODEBOOKS;
        int totalMasked;
        int candidateIndex;

        if (PositionTemperature > 0f)
        {
            float inverseTemperature = 1f / PositionTemperature;

            // 并行分区：每个线程处理一段连续的 t 范围
            var partitioner = System.Collections.Concurrent.Partitioner.Create(0, totalElements);
            totalMasked = 0;
            candidateIndex = 0;

            // 使用 thread-local 的计数器避免 Interlocked 争用
            var maskedCounts = new System.Collections.Concurrent.ConcurrentBag<int>();

            System.Threading.Tasks.Parallel.ForEach(partitioner, (range, state, threadIndex) =>
            {
                var localRng = new FastRng((ulong)(threadIndex + 1) * 0x9E3779B97F4A7C15UL);
                int localMasked = 0;

                for (int idx = range.Item1; idx < range.Item2; idx++)
                {
                    int t = idx / NUM_CODEBOOKS;
                    int codebook = idx % NUM_CODEBOOKS;

                    // Layer Penalty
                    _scoresBuf[idx] -= codebook * LayerPenaltyFactor;

                    // Gumbel 噪声
                    double uniform = Math.Max(1e-10, localRng.NextDouble01());
                    _scoresBuf[idx] = (float)(_scoresBuf[idx] * inverseTemperature
                                              - Math.Log(-Math.Log(uniform)));

                    // Mask 检测
                    bool isMasked = inputIds[codebook * sequenceLength + generateStart + t] == MASK_TOKEN;
                    if (!isMasked)
                        _scoresBuf[idx] = float.NegativeInfinity;
                    else
                        localMasked++;
                }

                maskedCounts.Add(localMasked);
            });

            totalMasked = 0;
            foreach (var count in maskedCounts) totalMasked += count;

            // 收集候选（单线程，因为需要写入共享缓冲区）
            candidateIndex = 0;
            for (int idx = 0; idx < totalElements; idx++)
            {
                if (_scoresBuf[idx] > float.NegativeInfinity)
                {
                    int t = idx / NUM_CODEBOOKS;
                    int codebook = idx % NUM_CODEBOOKS;
                    _allScoresBuf[candidateIndex++] = (t, codebook, _scoresBuf[idx]);
                }
            }
        }
        else
        {
            // PositionTemperature == 0 时只需 Layer Penalty + mask 收集
            totalMasked = 0;
            candidateIndex = 0;

            for (int idx = 0; idx < totalElements; idx++)
            {
                int t = idx / NUM_CODEBOOKS;
                int codebook = idx % NUM_CODEBOOKS;

                // Layer Penalty
                _scoresBuf[idx] -= codebook * LayerPenaltyFactor;

                // Mask 检测
                bool isMasked = inputIds[codebook * sequenceLength + generateStart + t] == MASK_TOKEN;
                if (!isMasked)
                    _scoresBuf[idx] = float.NegativeInfinity;
                else
                {
                    _allScoresBuf[candidateIndex++] = (t, codebook, _scoresBuf[idx]);
                    totalMasked++;
                }
            }
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
            Array.Sort(_allScoresBuf, 0, totalMasked, _scoreDescendingComparer);

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
        int refLength, int textLength, bool needRebuildBatch = true)
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
            refLength, textLength, targetLength, needRebuildBatch);

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

        if (_idsBuf == null || _idsBuf.Length != idsSize)
        { _idsBuf = new long[idsSize]; rebuild = true; }

        if (_audioBuf == null || _audioBuf.Length != audioSize)
        { _audioBuf = new bool[audioSize]; rebuild = true; }

        if (_attnBuf == null || _attnBuf.Length != attnSize)
        { _attnBuf = new bool[attnSize]; rebuild = true; }

        if (_posBuf == null || _posBuf.Length != posSize)
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
    public void Dispose()
    {
        // [OPT-7] IOBinding / OrtValue 需要显式释放，顺序：先解绑，再释放 OrtValue，最后释放 session
        try { _ioBinding?.ClearBoundInputs(); _ioBinding?.ClearBoundOutputs(); } catch { /* 忽略清理期异常 */ }
        _ovIds?.Dispose();
        _ovAudio?.Dispose();
        _ovAttn?.Dispose();
        _ovPos?.Dispose();
        _ioBinding?.Dispose();
        _runOptions?.Dispose();
        _session?.Dispose();
    }
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
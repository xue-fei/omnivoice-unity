using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 封装 audio_tokenizer_encoder + audio_tokenizer_decoder ONNX 会话
/// Encoder IO: audio [B,1,N] float32 → audio_codes [B,8,T] int64
/// Decoder IO: audio_codes [B,8,T] int64 → audio [B,1,N] float32
/// </summary>
public class AudioTokenizer : IDisposable
{
    const int HOP_LENGTH = 960;
    const int NUM_CODEBOOKS = 8;

    InferenceSession _encSession;
    InferenceSession _decSession;

    public AudioTokenizer(string encModelPath, string decModelPath,
                          ExecutionProviderType ep = ExecutionProviderType.CPU,
                          int deviceId = 0)
    {
        // Encoder 和 Decoder 共用同一份 SessionOptions 配置，
        // 但各自持有独立 InferenceSession，互不干扰。
        var opts = BuildSessionOptions(ep, deviceId);

        _encSession = new InferenceSession(encModelPath, opts);
        _decSession = new InferenceSession(decModelPath, opts);

        Debug.Log($"[AudioTokenizer] 已加载 (EP={ep}, device={deviceId})");
    }

    // ════════════════════════════════════════════════════════════════
    // BuildSessionOptions — 按 EP 类型配置，失败时自动回退 CPU
    // ════════════════════════════════════════════════════════════════
    static SessionOptions BuildSessionOptions(ExecutionProviderType ep, int deviceId)
    {
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        opts.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        opts.InterOpNumThreads = 1;

        // AudioTokenizer 模型（EnCodec）是纯卷积结构，shape 固定，
        // 开启 MemoryPattern 让 ORT 复用 GPU 内存分配，减少每次调用的开销。
        opts.EnableMemoryPattern = true;

        if (ep == ExecutionProviderType.CUDA)
        {
            try
            {
                var cudaOpts = new OrtCUDAProviderOptions();
                cudaOpts.UpdateOptions(new Dictionary<string, string>
                {
                    { "device_id",                deviceId.ToString() },
                    { "arena_extend_strategy",    "kSameAsRequested"  },
                    { "cudnn_conv_algo_search",   "HEURISTIC"         },
                    { "do_copy_in_default_stream","1"                 },
                    { "use_tf32",                 "1"                 },
                });
                opts.AppendExecutionProvider_CUDA(cudaOpts);
                Debug.Log($"[AudioTokenizer] CUDA EP (device={deviceId})");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[AudioTokenizer] CUDA EP 失败: {ex.Message}，回退 CPU");
                ApplyCpuFallback(opts);
            }
        }
        else if (ep == ExecutionProviderType.DML)
        {
            try
            {
                opts.AppendExecutionProvider_DML(deviceId);
                Debug.Log($"[AudioTokenizer] DML EP (device={deviceId})");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[AudioTokenizer] DML EP 失败: {ex.Message}，回退 CPU");
                ApplyCpuFallback(opts);
            }
        }
        else
        {
            ApplyCpuFallback(opts);
        }

        return opts;
    }

    static void ApplyCpuFallback(SessionOptions opts)
    {
        opts.IntraOpNumThreads = Math.Max(2, Environment.ProcessorCount / 2);
        Debug.Log($"[AudioTokenizer] CPU EP (threads={opts.IntraOpNumThreads})");
    }

    // ════════════════════════════════════════════════════════════════
    // Encode
    // ════════════════════════════════════════════════════════════════
    /// <summary>
    /// 将参考音频编码为 audio codes
    /// </summary>
    /// <param name="pcm">mono float32, 24kHz, 归一化到 [-1,1]</param>
    /// <returns>audio_codes [8, T]</returns>
    public long[,] Encode(float[] pcm)
    {
        int aligned = ((pcm.Length + HOP_LENGTH - 1) / HOP_LENGTH) * HOP_LENGTH;
        float[] padded = new float[aligned];
        Array.Copy(pcm, padded, pcm.Length);

        var tensor = new DenseTensor<float>(padded, new[] { 1, 1, aligned });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("audio", tensor)
        };

        using var results = _encSession.Run(inputs);
        var codesTensor = results[0].AsTensor<long>();
        int T = codesTensor.Dimensions[2];
        var codes = new long[NUM_CODEBOOKS, T];
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            for (int t = 0; t < T; t++)
                codes[cb, t] = codesTensor[0, cb, t];
        return codes;
    }

    // ════════════════════════════════════════════════════════════════
    // Decode
    // ════════════════════════════════════════════════════════════════
    /// <summary>
    /// 将 audio codes 解码为 PCM 波形
    /// </summary>
    /// <param name="codes">[8, T]</param>
    /// <returns>mono float32 PCM, 24kHz</returns>
    public float[] Decode(long[,] codes)
    {
        int T = codes.GetLength(1);
        var flat = new long[NUM_CODEBOOKS * T];
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            for (int t = 0; t < T; t++)
                flat[cb * T + t] = codes[cb, t];

        var tensor = new DenseTensor<long>(flat, new[] { 1, NUM_CODEBOOKS, T });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("audio_codes", tensor)
        };

        using var results = _decSession.Run(inputs);
        var audioTensor = results[0].AsTensor<float>();
        int N = audioTensor.Dimensions[2];
        float[] pcm = new float[N];
        for (int i = 0; i < N; i++)
            pcm[i] = audioTensor[0, 0, i];
        return pcm;
    }

    public void Dispose()
    {
        _encSession?.Dispose();
        _decSession?.Dispose();
    }
}
using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

/// <summary>
/// 封装 audio_tokenizer_encoder + audio_tokenizer_decoder ONNX 会话
/// Encoder IO: audio [B,1,N] float32 → audio_codes [B,8,T] int64
/// Decoder IO: audio_codes [B,8,T] int64 → audio [B,1,N] float32
/// </summary>
public class AudioTokenizer : IDisposable
{
    const int SAMPLE_RATE = 24000;
    const int HOP_LENGTH = 960;   // num_samples / num_frames
    const int NUM_CODEBOOKS = 8;

    InferenceSession _encSession;
    InferenceSession _decSession;

    public AudioTokenizer(string encModelPath, string decModelPath)
    {
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        opts.InterOpNumThreads = 1;

        _encSession = new InferenceSession(encModelPath, opts);
        _decSession = new InferenceSession(decModelPath, opts);
        Debug.Log("[AudioTokenizer] Sessions loaded.");
    }

    /// <summary>
    /// 将参考音频编码为 audio codes
    /// </summary>
    /// <param name="pcm">mono float32, 24kHz, 归一化到 [-1,1]</param>
    /// <returns>audio_codes [8, T]</returns>
    public long[,] Encode(float[] pcm)
    {
        // 对齐到 hop_length 的整数倍
        int aligned = ((pcm.Length + HOP_LENGTH - 1) / HOP_LENGTH) * HOP_LENGTH;
        float[] padded = new float[aligned];
        Array.Copy(pcm, padded, pcm.Length);

        // [B=1, C=1, N]
        var tensor = new DenseTensor<float>(padded, new[] { 1, 1, aligned });
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("audio", tensor)
        };

        using var results = _encSession.Run(inputs);
        var codesTensor = results[0].AsTensor<long>();
        // shape: [1, 8, T]
        int T = codesTensor.Dimensions[2];
        var codes = new long[NUM_CODEBOOKS, T];
        for (int cb = 0; cb < NUM_CODEBOOKS; cb++)
            for (int t = 0; t < T; t++)
                codes[cb, t] = codesTensor[0, cb, t];
        return codes;
    }

    /// <summary>
    /// 将 audio codes 解码为 PCM 波形
    /// </summary>
    /// <param name="codes">[8, T]</param>
    /// <returns>mono float32 PCM, 24kHz</returns>
    public float[] Decode(long[,] codes)
    {
        int T = codes.GetLength(1);
        // 展开为 [1, 8, T] 一维数组（C# row-major: B=0 固定）
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
        // shape: [1, 1, N]
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
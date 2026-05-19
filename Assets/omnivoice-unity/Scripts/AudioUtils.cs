using System;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public static class AudioUtils
{
    public const int MODEL_SAMPLE_RATE = 24000;

    /// <summary>
    /// 将 Unity AudioClip 转为 mono float32 PCM，并重采样到 24kHz
    /// </summary>
    public static float[] AudioClipToPCM(AudioClip clip)
    {
        float[] data = new float[clip.samples * clip.channels];
        clip.GetData(data, 0);

        // 若多声道，混合为单声道
        float[] mono;
        if (clip.channels > 1)
        {
            mono = new float[clip.samples];
            for (int i = 0; i < clip.samples; i++)
            {
                float sum = 0;
                for (int c = 0; c < clip.channels; c++)
                    sum += data[i * clip.channels + c];
                mono[i] = sum / clip.channels;
            }
        }
        else mono = data;

        // 重采样到 24kHz
        if (clip.frequency == MODEL_SAMPLE_RATE) return mono;
        return Resample(mono, clip.frequency, MODEL_SAMPLE_RATE);
    }

    /// <summary>
    /// 线性插值重采样
    /// </summary>
    public static float[] Resample(float[] input, int srcRate, int dstRate)
    {
        if (srcRate == dstRate) return input;
        double ratio = (double)srcRate / dstRate;
        int outLen = (int)(input.Length / ratio);
        var output = new float[outLen];
        for (int i = 0; i < outLen; i++)
        {
            double pos = i * ratio;
            int idx = (int)pos;
            float frac = (float)(pos - idx);
            if (idx + 1 < input.Length)
                output[i] = input[idx] * (1 - frac) + input[idx + 1] * frac;
            else
                output[i] = input[Math.Min(idx, input.Length - 1)];
        }
        return output;
    }

    /// <summary>
    /// PCM float32 → Unity AudioClip
    /// </summary>
    public static AudioClip PCMToAudioClip(float[] pcm, string name = "generated")
    {
        var clip = AudioClip.Create(name, pcm.Length, 1, MODEL_SAMPLE_RATE, false);
        clip.SetData(pcm, 0);
        return clip;
    }

    /// <summary>
    /// 保存 PCM 为 WAV 文件（16-bit）
    /// </summary>
    public static void SaveWav(string path, float[] pcm, int sampleRate = MODEL_SAMPLE_RATE)
    {
        using var fs = new FileStream(path, FileMode.Create);
        using var bw = new BinaryWriter(fs);
        int byteCount = pcm.Length * 2;
        bw.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
        bw.Write(36 + byteCount);
        bw.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));
        bw.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
        bw.Write(16); bw.Write((short)1); bw.Write((short)1);
        bw.Write(sampleRate); bw.Write(sampleRate * 2);
        bw.Write((short)2); bw.Write((short)16);
        bw.Write(System.Text.Encoding.ASCII.GetBytes("data"));
        bw.Write(byteCount);
        foreach (var s in pcm)
        {
            short v = (short)Mathf.Clamp(s * 32767f, -32768f, 32767f);
            bw.Write(v);
        }
    }

    /// <summary>
    /// RMS 归一化输出音频
    /// </summary>
    public static void NormalizeRMS(float[] pcm, float targetRMS = 0.1f)
    {
        float rms = 0;
        foreach (var s in pcm) rms += s * s;
        rms = Mathf.Sqrt(rms / pcm.Length);
        if (rms < 1e-8f) return;
        float scale = targetRMS / rms;
        for (int i = 0; i < pcm.Length; i++) pcm[i] *= scale;
    }

    /// <summary>
    /// 简单的淡入淡出（避免爆音）
    /// </summary>
    public static void ApplyFade(float[] pcm, int fadeSamples = 480)
    {
        int f = Math.Min(fadeSamples, pcm.Length / 4);
        for (int i = 0; i < f; i++)
        {
            float t = (float)i / f;
            pcm[i] *= t;
            pcm[pcm.Length - 1 - i] *= t;
        }
    }
}
using System;
using System.Diagnostics;

/// <summary>
/// RTF (Real-Time Factor) 统计类
/// RTF = 处理时间 / 音频时长
/// RTF < 1.0 表示实时，RTF > 1.0 表示非实时
/// </summary>
[Serializable]
public class RTFStatistics
{
    // 时间记录
    private Stopwatch _stopwatch = new Stopwatch();

    // 各阶段耗时（毫秒）
    private float _encodingTimeMs;
    private float _generationTimeMs;
    private float _decodingTimeMs;
    private float _totalTimeMs;

    // 音频时长（秒）
    private float _audioDuration;

    // RTF 值
    private float _rtf;

    // 是否有数据
    private bool _hasData;

    // 公共属性
    public float EncodingTimeMs => _encodingTimeMs;
    public float GenerationTimeMs => _generationTimeMs;
    public float DecodingTimeMs => _decodingTimeMs;
    public float TotalTimeMs => _totalTimeMs;
    public float AudioDuration => _audioDuration;
    public float RTF => _rtf;
    public bool HasData => _hasData;

    // 新增：上次各阶段耗时（用于日志输出）
    public float LastEncodingTimeMs => _encodingTimeMs;
    public float LastGenerationTimeMs => _generationTimeMs;
    public float LastDecodingTimeMs => _decodingTimeMs;
    public float LastTotalTimeMs => _totalTimeMs;

    public RTFStatistics()
    {
        Reset();
    }

    /// <summary>
    /// 重置所有统计数据
    /// </summary>
    public void Reset()
    {
        _encodingTimeMs = 0;
        _generationTimeMs = 0;
        _decodingTimeMs = 0;
        _totalTimeMs = 0;
        _audioDuration = 0;
        _rtf = 0;
        _hasData = false;
        _stopwatch.Reset();
    }

    #region 计时方法

    /// <summary>
    /// 开始编码计时
    /// </summary>
    public void StartEncoding()
    {
        _stopwatch.Restart();
    }

    /// <summary>
    /// 结束编码计时
    /// </summary>
    public void EndEncoding()
    {
        _stopwatch.Stop();
        _encodingTimeMs = (float)_stopwatch.Elapsed.TotalMilliseconds;
        _stopwatch.Reset();
    }

    /// <summary>
    /// 开始生成计时
    /// </summary>
    public void StartGeneration()
    {
        _stopwatch.Restart();
    }

    /// <summary>
    /// 结束生成计时
    /// </summary>
    public void EndGeneration()
    {
        _stopwatch.Stop();
        _generationTimeMs = (float)_stopwatch.Elapsed.TotalMilliseconds;
        _stopwatch.Reset();
    }

    /// <summary>
    /// 开始解码计时
    /// </summary>
    public void StartDecoding()
    {
        _stopwatch.Restart();
    }

    /// <summary>
    /// 结束解码计时
    /// </summary>
    public void EndDecoding()
    {
        _stopwatch.Stop();
        _decodingTimeMs = (float)_stopwatch.Elapsed.TotalMilliseconds;
        _stopwatch.Reset();
    }

    /// <summary>
    /// 设置音频时长
    /// </summary>
    public void SetAudioDuration(float duration)
    {
        _audioDuration = duration;
    }

    /// <summary>
    /// 计算总耗时和 RTF
    /// </summary>
    public void CalculateRTF()
    {
        _totalTimeMs = _encodingTimeMs + _generationTimeMs + _decodingTimeMs;

        if (_audioDuration > 0)
        {
            _rtf = (_totalTimeMs / 1000f) / _audioDuration;
            _hasData = true;
        }
        else
        {
            _rtf = 0;
            _hasData = false;
        }
    }

    #endregion

    #region 获取统计信息

    /// <summary>
    /// 获取格式化的统计字符串
    /// </summary>
    public string GetFormattedStats()
    {
        if (!_hasData)
            return "暂无数据";

        return $"RTF: {_rtf:F3}\n" +
               $"音频时长: {_audioDuration:F2}s\n" +
               $"编码耗时: {_encodingTimeMs:F1}ms\n" +
               $"生成耗时: {_generationTimeMs:F1}ms\n" +
               $"解码耗时: {_decodingTimeMs:F1}ms\n" +
               $"总耗时: {_totalTimeMs:F1}ms\n" +
               $"{(IsRealtime() ? "✅ 实时 (RTF < 1.0)" : "⏳ 非实时 (RTF > 1.0)")}";
    }

    /// <summary>
    /// 获取简化的统计字符串（单行）
    /// </summary>
    public string GetShortStats()
    {
        if (!_hasData)
            return "RTF: 暂无数据";

        return $"RTF: {_rtf:F3} | 音频: {_audioDuration:F2}s | 总耗时: {_totalTimeMs:F1}ms | {(IsRealtime() ? "✅实时" : "⏳非实时")}";
    }

    /// <summary>
    /// 判断是否满足实时条件
    /// </summary>
    public bool IsRealtime()
    {
        return _hasData && _rtf < 1.0f;
    }

    /// <summary>
    /// 获取性能等级
    /// </summary>
    public string GetPerformanceLevel()
    {
        if (!_hasData) return "未知";

        if (_rtf < 0.5f) return "极佳 (RTF < 0.5)";
        if (_rtf < 1.0f) return "良好 (RTF < 1.0)";
        if (_rtf < 2.0f) return "一般 (RTF < 2.0)";
        return "较差 (RTF >= 2.0)";
    }

    #endregion

    #region 日志输出

    /// <summary>
    /// 输出 RTF 日志
    /// </summary>
    public void LogRTF()
    {
        if (!_hasData)
        {
            UnityEngine.Debug.Log("[RTF] 暂无数据");
            return;
        }

        UnityEngine.Debug.Log($"[RTF] === 性能统计 ===\n" +
                             $"RTF: {_rtf:F3} ({GetPerformanceLevel()})\n" +
                             $"音频时长: {_audioDuration:F2}s\n" +
                             $"编码耗时: {_encodingTimeMs:F1}ms\n" +
                             $"生成耗时: {_generationTimeMs:F1}ms\n" +
                             $"解码耗时: {_decodingTimeMs:F1}ms\n" +
                             $"总耗时: {_totalTimeMs:F1}ms\n" +
                             $"{(IsRealtime() ? "✅ 达到实时性能" : "⏳ 未达到实时性能")}");
    }

    #endregion
}
using System;
using System.IO;
using System.Collections;
using UnityEngine;

/// <summary>
/// OmniVoiceRunner — 语音克隆与生成入口
/// 异步初始化和推理，不阻塞主线程。
/// </summary>
public class OmniVoiceRunner : MonoBehaviour
{
    [Header("音频设置")]
    public AudioClip referenceAudio;
    [Tooltip("参考音频的文字内容（语音克隆模式需要）。留空则不合并到文本段。")]
    public string referenceText = "";
    [TextArea] public string targetText = "你好，这是使用语音克隆生成的音频。";
    public string targetLanguage = "Chinese";
    public AudioSource outputAudioSource;

    [Header("模型路径（相对 StreamingAssets）")]
    public string lmModelRelPath = "OmniVoice/omnivoice_lm_int8_hq/model.onnx";
    public string encModelRelPath = "OmniVoice/audio_tokenizer_encoder_int8/model.onnx";
    public string decModelRelPath = "OmniVoice/audio_tokenizer_decoder_int8/model.onnx";
    public string tokenizerJsonRelPath = "OmniVoice/tokenizer.json";

    [Header("推理加速（EP 选择）")]
    [Tooltip("CUDA = NVIDIA GPU；DML = DirectML (Windows/AMD/Intel)；CPU = 纯 CPU 多线程")]
    public ExecutionProviderType executionProvider = ExecutionProviderType.CUDA;
    [Tooltip("GPU device index，多卡环境可指定")]
    public int deviceId = 0;

    [Header("生成参数（与原版 Python 对齐）")]
    [Tooltip("扩散步数，原版默认 32；速度优先可降至 16")]
    public int numStep = 32;
    [Tooltip("CFG 引导强度，原版默认 2.0；若输出异常可尝试 0（关闭 CFG）")]
    public float guidanceScale = 1.0f;
    [Tooltip("调度时移 τ，原版默认 0.1")]
    public float tShift = 0.1f;
    [Tooltip("position_temperature: 位置选择温度，原版默认 5.0")]
    public float positionTemperature = 5.0f;
    [Tooltip("class_temperature: token 采样温度，原版默认 0.0（greedy argmax）；>0 时使用 top-k ratio + Gumbel")]
    public float classTemperature = 0.0f;
    [Tooltip("层惩罚系数，原版默认 5.0；控制 codebook 从低到高逐层解 mask")]
    public float layerPenaltyFactor = 5.0f;
    [Tooltip("[OPT-11] 调度加速倍率：1.0=原版，2.0=2倍速，3.0=3倍速，4.0=4倍速。推荐3.0")]
    public float scheduleAcceleration = 3.0f;
    [Tooltip("目标生成时长（秒）。0 = 按文字长度自动估算")]
    public float targetDurSec = 0f;

    [Header("状态回调")]
    /// <summary>模型加载完成回调（主线程）</summary>
    public UnityEngine.Events.UnityEvent onModelReady;
    /// <summary>模型加载失败回调（主线程，参数为错误消息）</summary>
    public UnityEngine.Events.UnityEvent<string> onModelLoadFailed;
    /// <summary>生成完成回调（主线程，参数为 RTF 字符串）</summary>
    public UnityEngine.Events.UnityEvent<string> onGenerationComplete;

    OmniVoiceLM _lm;
    AudioTokenizer _tokenizer;
    Qwen2Tokenizer _textTok;
    bool _isGenerating;
    bool _modelReady;

    public bool IsReady => _modelReady;
    public bool IsGenerating => _isGenerating;

    void Start()
    {
        Application.targetFrameRate = 60;
        // 异步初始化模型，不阻塞主线程
        InitializeModelsAsync();
    }

    /// <summary>
    /// 在后台线程异步加载模型，完成后回调主线程。
    /// </summary>
    private void InitializeModelsAsync()
    {
        Debug.Log("[OmniVoiceRunner] 开始异步加载模型...");

        Loom.RunAsync(() =>
        {
            try
            {
                string lmPath = Path.Combine(Application.streamingAssetsPath, lmModelRelPath);
                string encPath = Path.Combine(Application.streamingAssetsPath, encModelRelPath);
                string decPath = Path.Combine(Application.streamingAssetsPath, decModelRelPath);
                string tokPath = Path.Combine(Application.streamingAssetsPath, tokenizerJsonRelPath);

                _lm = new OmniVoiceLM(lmPath, executionProvider, deviceId)
                {
                    NumStep = numStep,
                    GuidanceScale = guidanceScale,
                    TShift = tShift,
                    PositionTemperature = positionTemperature,
                    ClassTemperature = classTemperature,
                    LayerPenaltyFactor = layerPenaltyFactor,
                    ScheduleAcceleration = scheduleAcceleration,
                };

                _tokenizer = new AudioTokenizer(encPath, decPath, executionProvider, deviceId);

                if (File.Exists(tokPath))
                {
                    _textTok = Qwen2Tokenizer.Load(tokPath);
                }

                // 回到主线程标记完成
                Loom.QueueOnMainThread(() =>
                {
                    _modelReady = true;
                    Debug.Log($"[OmniVoiceRunner] 模型异步加载完成 (EP={executionProvider}, device={deviceId})");
                    onModelReady?.Invoke();
                });
            }
            catch (Exception e)
            {
                Debug.LogError($"[OmniVoiceRunner] 模型加载失败: {e.Message}");
                Loom.QueueOnMainThread(() =>
                {
                    _modelReady = false;
                    onModelLoadFailed?.Invoke(e.Message);
                });
            }
        });
    }

    /// <summary>
    /// 开始语音克隆。
    /// </summary>
    public void CloneVoice()
    {
        if (!_modelReady)
        {
            Debug.LogWarning("[OmniVoiceRunner] 模型尚未加载完成");
            return;
        }
        StartCoroutine(CloneVoiceCoroutine());
    }

    void OnDestroy()
    {
        _lm?.Dispose();
        _tokenizer?.Dispose();
    }

    IEnumerator CloneVoiceCoroutine()
    {
        if (_isGenerating) { Debug.LogWarning("上一次生成仍在进行"); yield break; }
        _isGenerating = true;
        float t0 = Time.realtimeSinceStartup;

        // 1. 编码参考音频
        long[,] refCodes = null;
        int T_ref = 0;
        float refRms = -1f;

        if (referenceAudio != null)
        {
            float[] refPCM = AudioUtils.AudioClipToPCM(referenceAudio);

            refRms = 0f;
            foreach (float s in refPCM) refRms += s * s;
            refRms = Mathf.Sqrt(refRms / refPCM.Length);
            if (refRms > 0f && refRms < 0.1f)
            {
                float scale = 0.1f / refRms;
                for (int i = 0; i < refPCM.Length; i++) refPCM[i] *= scale;
                Debug.Log($"[OmniVoiceRunner] 参考音频 RMS 归一化: {refRms:F4} → 0.1 (×{scale:F2})");
            }

            refCodes = _tokenizer.Encode(refPCM);
            T_ref = refCodes.GetLength(1);
            float refDur = T_ref * 960f / 24000f;
            Debug.Log($"[OmniVoiceRunner] 参考音频: {refDur:F1}s ({T_ref} 帧)  RMS={refRms:F4}");

            const int MAX_REF_FRAMES = 500;
            if (T_ref > MAX_REF_FRAMES)
            {
                Debug.LogWarning($"[OmniVoiceRunner] 参考音频过长，截断至 {MAX_REF_FRAMES} 帧 (20s)");
                var truncated = new long[OmniVoiceLM.NUM_CODEBOOKS, MAX_REF_FRAMES];
                for (int cb = 0; cb < OmniVoiceLM.NUM_CODEBOOKS; cb++)
                    for (int t = 0; t < MAX_REF_FRAMES; t++)
                        truncated[cb, t] = refCodes[cb, t];
                refCodes = truncated;
                T_ref = MAX_REF_FRAMES;
            }

            if (refDur < 2f) Debug.LogWarning("参考音频过短（< 2s），克隆质量可能较差");
        }

        // 2. 构建文本 prompt
        int[] textTokenIds;
        bool hasRefAudio = referenceAudio != null;
        string refTextStr = hasRefAudio && !string.IsNullOrEmpty(referenceText) ? referenceText : null;
        string normalizedTarget = TextNormalizer.Normalize(targetText);
        Debug.Log("normalizedTarget:" + normalizedTarget);
        if (_textTok != null && !string.IsNullOrEmpty(normalizedTarget))
        {
            textTokenIds = _textTok.BuildPrompt(normalizedTarget, targetLanguage, instruct: null,
                                                refText: refTextStr, hasRefAudio: hasRefAudio);
            Debug.Log($"[OmniVoiceRunner] 文本 prompt: {textTokenIds.Length} tokens " +
                      $"(hasRefAudio={hasRefAudio}, refText={refTextStr != null})");
        }
        else
        {
            textTokenIds = Array.Empty<int>();
        }

        // 3. 估算目标帧数
        int targetLen = EstimateTargetLen(normalizedTarget, targetLanguage, T_ref);
        Debug.Log($"[OmniVoiceRunner] 目标帧数: {targetLen} ({targetLen * 960f / 24000f:F1}s)");

        // 4. 异步推理（Loom 后台线程 + 主线程回调）
        long[,] generatedCodes = null;
        Exception err = null;

        // 每帧等待直到推理完成
        bool inferenceDone = false;
        Loom.RunAsync(() =>
        {
            try { generatedCodes = _lm.Generate(textTokenIds, refCodes, targetLen); }
            catch (Exception e) { err = e; }
            finally { inferenceDone = true; }
        });

        // 等待推理完成（不阻塞主线程，但会等推理结束）
        yield return new WaitUntil(() => inferenceDone);

        if (err != null)
        {
            Debug.LogError($"[OmniVoiceLM] 生成异常:\n{err}");
            _isGenerating = false;
            yield break;
        }

        if (generatedCodes == null || generatedCodes.GetLength(1) == 0)
        {
            Debug.LogError("[OmniVoiceRunner] 生成结果为空");
            _isGenerating = false;
            yield break;
        }

        // 5. 解码（后台线程）
        float[] pcm = null;
        bool decodeDone = false;
        Loom.RunAsync(() =>
        {
            try { pcm = _tokenizer.Decode(generatedCodes); }
            catch (Exception e) { Debug.LogError($"[OmniVoiceRunner] 解码异常: {e.Message}"); }
            finally { decodeDone = true; }
        });

        yield return new WaitUntil(() => decodeDone);

        if (pcm == null)
        {
            Debug.LogError("[OmniVoiceRunner] 解码失败");
            _isGenerating = false;
            yield break;
        }

        // 6. 后处理（对齐 Python _post_process_audio）
        if (refRms >= 0f && refRms < 0.1f)
        {
            float restoreScale = refRms / 0.1f;
            for (int i = 0; i < pcm.Length; i++) pcm[i] *= restoreScale;
        }
        else if (refRms < 0f)
        {
            float peak = 0f;
            foreach (float s in pcm) { float abs = Mathf.Abs(s); if (abs > peak) peak = abs; }
            if (peak > 1e-6f)
                for (int i = 0; i < pcm.Length; i++) pcm[i] = pcm[i] / peak * 0.5f;
        }
        AudioUtils.ApplyFadeOut(pcm);

        float elapsed = Time.realtimeSinceStartup - t0;
        float audioDur = pcm.Length / 24000f;
        string rtfMsg = $"RTF={elapsed / audioDur:F2} 音频={audioDur:F1}s 耗时={elapsed:F1}s";
        Debug.Log($"[OmniVoiceRunner] ✅ 完成: {rtfMsg}");

        var clip = AudioUtils.PCMToAudioClip(pcm, "omnivoice_output");
        if (outputAudioSource != null) { outputAudioSource.clip = clip; outputAudioSource.Play(); }

        string savePath = Path.Combine(Application.dataPath, "omnivoice_output.wav");
        AudioUtils.SaveWav(savePath, pcm);
        Debug.Log($"[OmniVoiceRunner] 已保存至: {savePath}");

        // 回调
        onGenerationComplete?.Invoke(rtfMsg);

        _isGenerating = false;
    }

    int EstimateTargetLen(string text, string language, int T_ref)
    {
        if (targetDurSec > 0f)
            return Mathf.RoundToInt(targetDurSec * 24000f / 960f);
        if (string.IsNullOrEmpty(text))
            return T_ref > 0 ? T_ref : 100;

        string resolvedLang = _textTok != null
            ? Qwen2Tokenizer.ResolveLang(language)
            : language;
        bool isChinese = resolvedLang.StartsWith("zh", StringComparison.OrdinalIgnoreCase)
                      || resolvedLang.StartsWith("yue", StringComparison.OrdinalIgnoreCase)
                      || resolvedLang.StartsWith("wuu", StringComparison.OrdinalIgnoreCase)
                      || resolvedLang.StartsWith("nan", StringComparison.OrdinalIgnoreCase);

        float durSec;
        if (isChinese)
        {
            int charCount = 0;
            foreach (char c in text)
                if (!char.IsPunctuation(c) && !char.IsWhiteSpace(c)) charCount++;
            durSec = charCount * 0.22f;
        }
        else
        {
            int wordCount = text.Split(new[] { ' ', '\t', '\n' },
                StringSplitOptions.RemoveEmptyEntries).Length;
            durSec = wordCount * 0.4f;
        }

        durSec = Mathf.Clamp(durSec, 1.0f, 30.0f);
        int frames = Mathf.RoundToInt(durSec * 24000f / 960f);
        return Mathf.Max(frames, 25);
    }
}

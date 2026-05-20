using System;
using System.IO;
using System.Collections;
using UnityEngine;

/// <summary>
/// OmniVoice Unity Runner — 正确的扩散 LM 流程
///
/// 完整流程：
///   1. 用 Qwen2Tokenizer 把文本构建成 prompt tokens
///   2. 用 AudioTokenizer.Encode 把参考音频转为 codes
///   3. 估算目标生成帧数（按文字长度 + RuleDurationEstimator 简化版）
///   4. 调用 OmniVoiceLM.Generate 做扩散迭代（在后台线程）
///   5. 用 AudioTokenizer.Decode 把 codes 转回 PCM
///   6. 后处理（音量归一化、淡入淡出）并播放/保存
/// </summary>
public class OmniVoiceRunner : MonoBehaviour
{
    [Header("音频设置")]
    public AudioClip referenceAudio;

    [TextArea]
    public string targetText = "你好，这是使用语音克隆生成的音频。";
    public string targetLanguage = "Chinese";   // "Chinese" 或 "English"

    public AudioSource outputAudioSource;

    [Header("模型路径（相对 StreamingAssets）")]
    public string lmModelRelPath = "OmniVoice/omnivoice_lm_int8_hq/model.onnx";
    public string encModelRelPath = "OmniVoice/audio_tokenizer_encoder_int8/model.onnx";
    public string decModelRelPath = "OmniVoice/audio_tokenizer_decoder_int8/model.onnx";
    public string tokenizerJsonRelPath = "OmniVoice/tokenizer.json";

    [Header("生成参数（与原版 Python 对齐）")]
    [Tooltip("扩散步数，原版默认 32；速度优先可降至 16")]
    public int numStep = 32;

    [Tooltip("CFG 引导强度，原版默认 2.0")]
    public float guidanceScale = 2.0f;

    [Tooltip("调度时移 τ，原版默认 0.1")]
    public float tShift = 0.1f;

    [Tooltip("mask 位置选择温度，原版默认 5.0")]
    public float maskTemperature = 5.0f;

    [Tooltip("层惩罚系数，原版默认 5.0")]
    public float layerPenaltyFactor = 5.0f;

    [Tooltip("目标生成时长（秒）。0 = 按文字长度自动估算")]
    public float targetDurSec = 0f;

    // ─── 内部字段 ────────────────────────────────────────────────────────────
    OmniVoiceLM _lm;
    AudioTokenizer _tokenizer;
    Qwen2Tokenizer _textTok;
    bool _isGenerating;

    // ════════════════════════════════════════════════════════════════════════
    // Unity 生命周期
    // ════════════════════════════════════════════════════════════════════════

    void Start()
    {
        string lmPath = Path.Combine(Application.streamingAssetsPath, lmModelRelPath);
        string encPath = Path.Combine(Application.streamingAssetsPath, encModelRelPath);
        string decPath = Path.Combine(Application.streamingAssetsPath, decModelRelPath);
        string tokPath = Path.Combine(Application.streamingAssetsPath, tokenizerJsonRelPath);

        _lm = new OmniVoiceLM(lmPath)
        {
            NumStep = numStep,
            GuidanceScale = guidanceScale,
            TShift = tShift,
            MaskTempature = maskTemperature,
            LayerPenaltyFactor = layerPenaltyFactor,
        };

        _tokenizer = new AudioTokenizer(encPath, decPath);

        if (File.Exists(tokPath))
        {
            _textTok = Qwen2Tokenizer.Load(tokPath);
            Debug.Log("[OmniVoiceRunner] 文本 Tokenizer 已加载");
        }
        else
        {
            Debug.LogWarning($"[OmniVoiceRunner] 未找到 tokenizer.json ({tokPath})，将以空文本 prompt 运行");
        }

        Debug.Log("[OmniVoiceRunner] 初始化完成");
    }

    public void CloneVoice() => StartCoroutine(CloneVoiceCoroutine());

    void OnDestroy()
    {
        _lm?.Dispose();
        _tokenizer?.Dispose();
    }

    // ════════════════════════════════════════════════════════════════════════
    // 生成主协程
    // ════════════════════════════════════════════════════════════════════════

    IEnumerator CloneVoiceCoroutine()
    {
        if (_isGenerating) { Debug.LogWarning("上一次生成仍在进行"); yield break; }
        _isGenerating = true;
        float t0 = Time.realtimeSinceStartup;
        Debug.Log("[OmniVoiceRunner] ▶ 开始...");

        // ── 步骤 1：编码参考音频 ─────────────────────────────────────────
        long[,] refCodes = null;
        int T_ref = 0;

        if (referenceAudio != null)
        {
            float[] refPCM = AudioUtils.AudioClipToPCM(referenceAudio);
            // 参考音频建议 3-10 秒（过长会降质）
            refCodes = _tokenizer.Encode(refPCM);
            T_ref = refCodes.GetLength(1);
            float refDur = T_ref * 960f / 24000f;
            Debug.Log($"[OmniVoiceRunner] 参考音频: {refDur:F1}s ({T_ref} 帧)");
            if (refDur < 2f) Debug.LogWarning("参考音频过短（< 2s），克隆质量可能较差");
            if (refDur > 15f) Debug.LogWarning("参考音频较长（> 15s），建议裁剪至 3-10s");
        }

        // ── 步骤 2：构建文本 prompt ──────────────────────────────────────
        int[] textTokenIds;
        if (_textTok != null && !string.IsNullOrEmpty(targetText))
        {
            // BuildPrompt 格式（来自 Qwen2Tokenizer.cs）：
            //   <|denoise|> <|lang_start|> {language} <|lang_end|>
            //   <|text_start|> {text} <|text_end|>
            textTokenIds = _textTok.BuildPrompt(targetText, targetLanguage);
            Debug.Log($"[OmniVoiceRunner] 文本 prompt: {textTokenIds.Length} tokens");
        }
        else
        {
            textTokenIds = Array.Empty<int>();
        }

        // ── 步骤 3：估算目标生成帧数 ────────────────────────────────────
        int targetLen = EstimateTargetLen(targetText, targetLanguage, T_ref);
        Debug.Log($"[OmniVoiceRunner] 目标帧数: {targetLen} ({targetLen * 960f / 24000f:F1}s)");

        // ── 步骤 4：扩散 LM 生成（后台线程） ────────────────────────────
        long[,] generatedCodes = null;
        bool done = false;
        Exception err = null;

        System.Threading.ThreadPool.QueueUserWorkItem(_ =>
        {
            try { generatedCodes = _lm.Generate(textTokenIds, refCodes, targetLen); }
            catch (Exception e) { err = e; }
            finally { done = true; }
        });

        while (!done) yield return null;

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

        // ── 步骤 5：解码 codes → PCM ────────────────────────────────────
        float[] pcm = _tokenizer.Decode(generatedCodes);

        // ── 步骤 6：后处理 ───────────────────────────────────────────────
        AudioUtils.NormalizeRMS(pcm);
        AudioUtils.ApplyFade(pcm);

        float elapsed = Time.realtimeSinceStartup - t0;
        float audioDur = pcm.Length / 24000f;
        Debug.Log($"[OmniVoiceRunner] ✅ 完成: 音频={audioDur:F1}s 耗时={elapsed:F1}s RTF={elapsed / audioDur:F2}");

        // 播放
        var clip = AudioUtils.PCMToAudioClip(pcm, "omnivoice_output");
        if (outputAudioSource != null) { outputAudioSource.clip = clip; outputAudioSource.Play(); }

        // 保存
        string savePath = Path.Combine(Application.dataPath, "omnivoice_output.wav");
        AudioUtils.SaveWav(savePath, pcm);
        Debug.Log($"[OmniVoiceRunner] 已保存至: {savePath}");

        _isGenerating = false;
    }

    // ════════════════════════════════════════════════════════════════════════
    // 目标帧数估算（简化版 RuleDurationEstimator）
    // ════════════════════════════════════════════════════════════════════════

    int EstimateTargetLen(string text, string language, int T_ref)
    {
        if (targetDurSec > 0f)
            return Mathf.RoundToInt(targetDurSec * 24000f / 960f);

        if (string.IsNullOrEmpty(text))
            return T_ref > 0 ? T_ref : 100;

        // 粗估：中文每字约 0.22s（~5.5帧），英文每词约 0.4s（~10帧）
        // （参考原版 RuleDurationEstimator，比率相近）
        bool isChinese = language.IndexOf("Chinese", StringComparison.OrdinalIgnoreCase) >= 0;
        float durSec;
        if (isChinese)
        {
            // 中文：计字符数（去掉标点），每字 ~0.22s
            int charCount = 0;
            foreach (char c in text)
                if (!char.IsPunctuation(c) && !char.IsWhiteSpace(c)) charCount++;
            durSec = charCount * 0.22f;
        }
        else
        {
            // 英文：按空格拆词，每词 ~0.4s
            int wordCount = text.Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries).Length;
            durSec = wordCount * 0.4f;
        }

        durSec = Mathf.Clamp(durSec, 1.0f, 30.0f);
        int frames = Mathf.RoundToInt(durSec * 24000f / 960f);

        // 若有参考音频，以参考帧数为上界（避免生成远超参考的奇怪音频）
        if (T_ref > 0)
            frames = Mathf.Min(frames, T_ref * 3);

        return Mathf.Max(frames, 25);
    }
}
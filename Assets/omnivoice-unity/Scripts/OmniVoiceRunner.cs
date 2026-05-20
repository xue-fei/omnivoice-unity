using System;
using System.IO;
using System.Collections;
using UnityEngine;

public class OmniVoiceRunner : MonoBehaviour
{
    [Header("音频设置")]
    public AudioClip referenceAudio;
    [TextArea] public string targetText = "你好，这是使用语音克隆生成的音频。";
    public string targetLanguage = "Chinese";
    public AudioSource outputAudioSource;

    [Header("模型路径（相对 StreamingAssets）")]
    public string lmModelRelPath = "OmniVoice/omnivoice_lm_int8_hq/model.onnx";
    public string encModelRelPath = "OmniVoice/audio_tokenizer_encoder_int8/model.onnx";
    public string decModelRelPath = "OmniVoice/audio_tokenizer_decoder_int8/model.onnx";
    public string tokenizerJsonRelPath = "OmniVoice/tokenizer.json";

    [Header("生成参数（与原版 Python 对齐）")]
    [Tooltip("扩散步数，原版默认 32；速度优先可降至 16")]
    public int numStep = 32;
    [Tooltip("CFG 引导强度，原版默认 2.0；若输出异常可尝试 0（关闭 CFG）")]
    public float guidanceScale = 2.0f;
    [Tooltip("调度时移 τ，原版默认 0.1")]
    public float tShift = 0.1f;
    [Tooltip("mask 位置选择温度，原版默认 5.0")]
    public float maskTemperature = 5.0f;
    [Tooltip("层惩罚系数，原版约 1.0；旧版 5.0 过强会导致高层 codebook 不解 mask")]
    public float layerPenaltyFactor = 1.0f;
    [Tooltip("目标生成时长（秒）。0 = 按文字长度自动估算")]
    public float targetDurSec = 0f;

    OmniVoiceLM _lm;
    AudioTokenizer _tokenizer;
    Qwen2Tokenizer _textTok;
    bool _isGenerating;

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
            MaskTemperature = maskTemperature,
            LayerPenaltyFactor = layerPenaltyFactor,
        };

        _tokenizer = new AudioTokenizer(encPath, decPath);

        if (File.Exists(tokPath))
        {
            _textTok = Qwen2Tokenizer.Load(tokPath);
            if (_textTok != null)
                Debug.Log("[OmniVoiceRunner] 文本 Tokenizer 已加载");
        }
        else
        {
            Debug.LogWarning($"[OmniVoiceRunner] 未找到 tokenizer.json ({tokPath})");
        }

        Debug.Log("[OmniVoiceRunner] 初始化完成");
    }

    public void CloneVoice() => StartCoroutine(CloneVoiceCoroutine());

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
        if (referenceAudio != null)
        {
            float[] refPCM = AudioUtils.AudioClipToPCM(referenceAudio);
            refCodes = _tokenizer.Encode(refPCM);
            T_ref = refCodes.GetLength(1);
            float refDur = T_ref * 960f / 24000f;
            Debug.Log($"[OmniVoiceRunner] 参考音频原始: {refDur:F1}s ({T_ref} 帧)");

            // ★ 修复：截断参考音频到 6 秒（约 150 帧），避免过长参考干扰开头生成
            const int MAX_REF_FRAMES = 150; // 6s @ 25fps
            if (T_ref > MAX_REF_FRAMES)
            {
                Debug.LogWarning($"[OmniVoiceRunner] 参考音频过长，截断至 {MAX_REF_FRAMES} 帧 (6s)");
                var truncated = new long[OmniVoiceLM.NUM_CODEBOOKS, MAX_REF_FRAMES];
                for (int cb = 0; cb < OmniVoiceLM.NUM_CODEBOOKS; cb++)
                    for (int t = 0; t < MAX_REF_FRAMES; t++)
                        truncated[cb, t] = refCodes[cb, t];
                refCodes = truncated;
                T_ref = MAX_REF_FRAMES;
            }

            if (refDur < 2f) Debug.LogWarning("参考音频过短（<< 2s），克隆质量可能较差");
        }

        // 2. 构建文本 prompt
        int[] textTokenIds;
        if (_textTok != null && !string.IsNullOrEmpty(targetText))
        {
            textTokenIds = _textTok.BuildPrompt(targetText, targetLanguage);
            Debug.Log($"[OmniVoiceRunner] 文本 prompt: {textTokenIds.Length} tokens");
        }
        else
        {
            textTokenIds = Array.Empty<int>();
        }

        // 3. 估算目标帧数
        int targetLen = EstimateTargetLen(targetText, targetLanguage, T_ref);
        Debug.Log($"[OmniVoiceRunner] 目标帧数: {targetLen} ({targetLen * 960f / 24000f:F1}s)");

        // 4. 扩散生成（后台线程）
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

        // 5. 解码
        float[] pcm = _tokenizer.Decode(generatedCodes);

        // 6. 后处理
        AudioUtils.NormalizeRMS(pcm);
        AudioUtils.ApplyFade(pcm);

        float elapsed = Time.realtimeSinceStartup - t0;
        float audioDur = pcm.Length / 24000f;
        Debug.Log($"[OmniVoiceRunner] ✅ 完成: 音频={audioDur:F1}s 耗时={elapsed:F1}s RTF={elapsed / audioDur:F2}");

        var clip = AudioUtils.PCMToAudioClip(pcm, "omnivoice_output");
        if (outputAudioSource != null) { outputAudioSource.clip = clip; outputAudioSource.Play(); }

        string savePath = Path.Combine(Application.dataPath, "omnivoice_output.wav");
        AudioUtils.SaveWav(savePath, pcm);
        Debug.Log($"[OmniVoiceRunner] 已保存至: {savePath}");

        _isGenerating = false;
    }

    int EstimateTargetLen(string text, string language, int T_ref)
    {
        if (targetDurSec > 0f)
            return Mathf.RoundToInt(targetDurSec * 24000f / 960f);
        if (string.IsNullOrEmpty(text))
            return T_ref > 0 ? T_ref : 100;

        bool isChinese = language.IndexOf("Chinese", StringComparison.OrdinalIgnoreCase) >= 0;
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
            int wordCount = text.Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries).Length;
            durSec = wordCount * 0.4f;
        }

        durSec = Mathf.Clamp(durSec, 1.0f, 30.0f);
        int frames = Mathf.RoundToInt(durSec * 24000f / 960f);
        if (T_ref > 0)
            frames = Mathf.Min(frames, T_ref * 3);
        return Mathf.Max(frames, 25);
    }
}
using System;
using System.IO;
using System.Collections;
using UnityEngine;

/// <summary>
/// OmniVoice Unity Runner（扩散 LM 正确流程）
///
/// ★ 修正要点（对比旧版）：
///   1. 接入 Qwen2Tokenizer 构建文本 prompt tokens（语言 + 文本标记）
///   2. 使用 OmniVoiceLM.NumStep 迭代扩散，而非逐 token 自回归
///   3. GuidanceScale 默认 2.0（Python 版默认值）
///   4. targetLen 由文本长度估算（不依赖参考音频长度）
///   5. 支持独立指定生成时长
/// </summary>
public class OmniVoiceRunner : MonoBehaviour
{
    [Header("音频设置")]
    public AudioClip referenceAudio;

    [TextArea]
    public string targetText = "你好，这是使用语音克隆生成的音频。";

    public string targetLanguage = "Chinese";   // "Chinese" or "English"

    public AudioSource outputAudioSource;

    [Header("模型路径（相对 StreamingAssets）")]
    public string lmModelRelPath = "OmniVoice/omnivoice_lm_int8_hq/model.onnx";
    public string encModelRelPath = "OmniVoice/audio_tokenizer_encoder_int8/model.onnx";
    public string decModelRelPath = "OmniVoice/audio_tokenizer_decoder_int8/model.onnx";
    public string tokenizerJsonRelPath = "OmniVoice/tokenizer.json";

    [Header("生成参数（与 Python 版对齐）")]
    [Tooltip("扩散迭代步数。32 是原版默认值，越小越快质量越低。")]
    public int numStep = 32;

    [Tooltip("CFG 引导强度。原版默认 2.0，设 0 关闭。")]
    public float guidanceScale = 2.0f;

    [Tooltip("扩散调度 t_shift，原版默认 0.1。")]
    public float tShift = 0.1f;

    [Tooltip("TopK 采样。原版默认 50。")]
    public int topK = 50;

    [Tooltip("每帧生成目标时长(秒)。0 = 按参考音频长度自动估算。")]
    public float targetDurSec = 0f;

    // ─── 内部字段 ────────────────────────────────────────────────────────────
    OmniVoiceLM _lm;
    AudioTokenizer _tokenizer;
    Qwen2Tokenizer _textTokenizer;
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

        // LM
        _lm = new OmniVoiceLM(lmPath)
        {
            NumStep = numStep,
            GuidanceScale = guidanceScale,
            TShift = tShift,
            TopK = topK,
            Temperature = 1.0f,
        };

        // 音频 codec
        _tokenizer = new AudioTokenizer(encPath, decPath);

        // 文本 tokenizer（如不存在则回退为空 prompt）
        if (File.Exists(tokPath))
        {
            _textTokenizer = Qwen2Tokenizer.Load(tokPath);
            Debug.Log("[OmniVoiceRunner] 文本 Tokenizer 已加载");
        }
        else
        {
            Debug.LogWarning($"[OmniVoiceRunner] 未找到 tokenizer.json: {tokPath}，将以空文本 prompt 运行（音质可能下降）");
        }

        Debug.Log("[OmniVoiceRunner] 所有模型加载完成");
    }

    // 外部调用入口（如 UI Button.OnClick）
    public void CloneVoice() => StartCoroutine(CloneVoiceCoroutine());

    void OnDestroy()
    {
        _lm?.Dispose();
        _tokenizer?.Dispose();
    }

    // ════════════════════════════════════════════════════════════════════════
    // 语音克隆主协程
    // ════════════════════════════════════════════════════════════════════════

    IEnumerator CloneVoiceCoroutine()
    {
        if (_isGenerating)
        {
            Debug.LogWarning("[OmniVoiceRunner] 上一次生成仍在进行，请等待");
            yield break;
        }
        _isGenerating = true;

        Debug.Log("[OmniVoiceRunner] ▶ 开始语音克隆...");
        float t0 = Time.realtimeSinceStartup;

        // ── 步骤 1：编码参考音频 ─────────────────────────────────────────
        long[,] refCodes = null;
        int T_ref = 0;

        if (referenceAudio != null)
        {
            float[] refPCM = AudioUtils.AudioClipToPCM(referenceAudio);
            refCodes = _tokenizer.Encode(refPCM);
            T_ref = refCodes.GetLength(1);
            Debug.Log($"[OmniVoiceRunner] 参考音频 codes: [8, {T_ref}] ({T_ref * 960f / 24000f:F1}s)");
        }

        // ── 步骤 2：构建文本 prompt tokens ──────────────────────────────
        int[] textTokenIds;
        if (_textTokenizer != null && !string.IsNullOrEmpty(targetText))
        {
            // BuildPrompt 输出格式：<|denoise|> <|lang_start|> Chinese <|lang_end|> <|text_start|> 文字 <|text_end|>
            textTokenIds = _textTokenizer.BuildPrompt(targetText, targetLanguage);
            Debug.Log($"[OmniVoiceRunner] 文本 prompt: {textTokenIds.Length} tokens");
        }
        else
        {
            // audio-only 模式（无文本条件）
            textTokenIds = Array.Empty<int>();
            Debug.LogWarning("[OmniVoiceRunner] 以 audio-only 模式运行（无文本 tokenizer）");
        }

        // ── 步骤 3：确定目标生成帧数 ────────────────────────────────────
        int targetLen;
        if (targetDurSec > 0f)
        {
            // 用户指定时长
            targetLen = Mathf.RoundToInt(targetDurSec * 24000f / 960f);
        }
        else if (T_ref > 0)
        {
            // 按参考音频长度（语音克隆常见策略）
            // 同时参考文本长度做比例缩放（文字越多，生成越长）
            float textLenFactor = textTokenIds.Length > 0
                ? Mathf.Clamp((float)targetText.Length / 10f, 0.5f, 5f)
                : 1f;
            targetLen = Mathf.RoundToInt(T_ref * textLenFactor);
            targetLen = Mathf.Clamp(targetLen, 25, 500);  // 约 1s ~ 20s
        }
        else
        {
            // 无参考音频时，按文字长度粗估（每个中文字约 10 帧）
            targetLen = Mathf.Max(50, targetText.Length * 10);
        }

        Debug.Log($"[OmniVoiceRunner] 目标生成: {targetLen} 帧 ({targetLen * 960f / 24000f:F1}s)");

        // ── 步骤 4：扩散 LM 生成（后台线程）────────────────────────────
        long[,] generatedCodes = null;
        bool done = false;
        Exception err = null;

        System.Threading.ThreadPool.QueueUserWorkItem(_ =>
        {
            try
            {
                generatedCodes = _lm.Generate(textTokenIds, refCodes, targetLen);
            }
            catch (Exception e) { err = e; }
            finally { done = true; }
        });

        while (!done) yield return null;

        if (err != null)
        {
            Debug.LogError($"[OmniVoiceLM] 生成失败:\n{err}");
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
        float[] outputPCM = _tokenizer.Decode(generatedCodes);

        // 音频后处理
        AudioUtils.NormalizeRMS(outputPCM);
        AudioUtils.ApplyFade(outputPCM);

        // ── 步骤 6：播放 + 保存 ─────────────────────────────────────────
        float elapsed = Time.realtimeSinceStartup - t0;
        float audioDur = outputPCM.Length / 24000f;
        Debug.Log($"[OmniVoiceRunner] ✅ 完成: 音频={audioDur:F1}s 耗时={elapsed:F1}s RTF={elapsed / audioDur:F2}");

        var clip = AudioUtils.PCMToAudioClip(outputPCM, "omnivoice_output");
        if (outputAudioSource != null)
        {
            outputAudioSource.clip = clip;
            outputAudioSource.Play();
        }

        string savePath = Path.Combine(Application.dataPath, "omnivoice_output.wav");
        AudioUtils.SaveWav(savePath, outputPCM);
        Debug.Log($"[OmniVoiceRunner] 已保存: {savePath}");

        _isGenerating = false;
    }
}
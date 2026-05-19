using System;
using System.IO;
using System.Collections;
using UnityEngine;

/// <summary>
/// 挂在任意 GameObject 上。
/// 在 Inspector 中指定参考音频 AudioClip 和目标文本，调用 CloneVoice()。
/// </summary>
public class OmniVoiceRunner : MonoBehaviour
{
    [Header("音频设置")]
    public AudioClip referenceAudio;   // 3-10 秒参考音频（声音提供者）
    [TextArea]
    public string targetText = "你好，这是使用语音克隆生成的音频。";
    public AudioSource outputAudioSource;

    [Header("模型路径（相对 StreamingAssets）")]
    public string lmModelRelPath = "OmniVoice/omnivoice_lm_int8_hq/model.onnx";
    public string encModelRelPath = "OmniVoice/audio_tokenizer_encoder_int8/model.onnx";
    public string decModelRelPath = "OmniVoice/audio_tokenizer_decoder_int8/model.onnx";

    [Header("生成参数")]
    public float guidanceScale = 2.0f;
    public int topK = 50;
    public int maxNewTokens = 800;

    OmniVoiceLM _lm;
    AudioTokenizer _tokenizer;
    bool _isGenerating = false;

    void Start()
    {
        // 模型路径
        string lmPath = Path.Combine(Application.streamingAssetsPath, lmModelRelPath);
        string encPath = Path.Combine(Application.streamingAssetsPath, encModelRelPath);
        string decPath = Path.Combine(Application.streamingAssetsPath, decModelRelPath);

        _lm = new OmniVoiceLM(lmPath)
        {
            GuidanceScale = guidanceScale,
            TopK = topK,
            MaxNewTokens = maxNewTokens,
        };
        _tokenizer = new AudioTokenizer(encPath, decPath);
        Debug.Log("[OmniVoiceRunner] 模型加载完成");
    }

    /// <summary>
    /// 可从 UI 按钮直接调用
    /// </summary>
    public void CloneVoice() => StartCoroutine(CloneVoiceCoroutine());

    IEnumerator CloneVoiceCoroutine()
    {
        if (_isGenerating) { Debug.LogWarning("正在生成中，请等待。"); yield break; }
        _isGenerating = true;

        Debug.Log("[OmniVoiceRunner] 开始语音克隆...");
        float t0 = Time.realtimeSinceStartup;

        // 1. 编码参考音频
        long[,] refCodes = null;
        if (referenceAudio != null)
        {
            float[] refPCM = AudioUtils.AudioClipToPCM(referenceAudio);
            Debug.Log($"[OmniVoiceRunner] 参考音频: {refPCM.Length / 24000f:F1}s");
            refCodes = _tokenizer.Encode(refPCM);
            Debug.Log($"[OmniVoiceRunner] 参考 codes: [8, {refCodes.GetLength(1)}]");
        }

        // 2. 文本 tokenize（简化版：直接按字符转 unicode 码点作为 token id）
        //    生产环境请替换为真实的 BPE tokenizer（读取 vocab.json）
        int[] textTokenIds = SimpleTokenize(targetText);
        Debug.Log($"[OmniVoiceRunner] 文本 token 数: {textTokenIds.Length}");

        // 3. LM 生成（在非主线程中执行，避免卡 UI）
        long[,] generatedCodes = null;
        bool done = false;
        System.Threading.ThreadPool.QueueUserWorkItem(_ =>
        {
            try { generatedCodes = _lm.Generate(textTokenIds, refCodes); }
            catch (Exception e) { Debug.LogError($"[OmniVoiceLM] 生成失败: {e}"); }
            finally { done = true; }
        });

        // 等待后台线程完成
        while (!done) yield return null;

        if (generatedCodes == null || generatedCodes.GetLength(1) == 0)
        {
            Debug.LogError("[OmniVoiceRunner] 生成结果为空");
            _isGenerating = false;
            yield break;
        }

        // 4. 解码为 PCM
        float[] outputPCM = _tokenizer.Decode(generatedCodes);
        AudioUtils.NormalizeRMS(outputPCM);
        AudioUtils.ApplyFade(outputPCM);

        float elapsed = Time.realtimeSinceStartup - t0;
        float audioDur = outputPCM.Length / 24000f;
        Debug.Log($"[OmniVoiceRunner] 完成: 音频={audioDur:F1}s 耗时={elapsed:F1}s RTF={elapsed / audioDur:F2}");

        // 5. 播放
        var clip = AudioUtils.PCMToAudioClip(outputPCM, "omnivoice_output");
        if (outputAudioSource != null)
        {
            outputAudioSource.clip = clip;
            outputAudioSource.Play();
        }

        // 6. 可选：保存到本地
        string savePath = Path.Combine(Application.dataPath, "omnivoice_output.wav");
        AudioUtils.SaveWav(savePath, outputPCM);
        Debug.Log($"[OmniVoiceRunner] 已保存: {savePath}");

        _isGenerating = false;
    }

    /// <summary>
    /// 极简 tokenizer：仅作占位符，生产时替换为 SentencePiece/BPE 实现
    /// 真实模型需要读取 OmniVoice 的 tokenizer.json / vocab.json
    /// </summary>
    static int[] SimpleTokenize(string text)
    {
        var ids = new System.Collections.Generic.List<int>();
        ids.Add(OmniVoiceLM.TEXT_BOS);
        foreach (char c in text)
            ids.Add(Math.Min((int)c, 50255)); // 截断到词表范围
        ids.Add(OmniVoiceLM.TEXT_EOS);
        return ids.ToArray();
    }

    void OnDestroy()
    {
        _lm?.Dispose();
        _tokenizer?.Dispose();
    }
}
using System;
using System.IO;
using System.Collections;
using UnityEngine;

public class OmniVoiceRunner : MonoBehaviour
{
    [Header("音频设置")]
    public AudioClip referenceAudio;
    [TextArea] public string targetText = "你好，这是使用语音克隆生成的音频。";
    public AudioSource outputAudioSource;

    [Header("模型路径（相对 StreamingAssets）")]
    public string lmModelRelPath = "OmniVoice/omnivoice_lm_int8_hq/model.onnx";
    public string encModelRelPath = "OmniVoice/audio_tokenizer_encoder_int8/model.onnx";
    public string decModelRelPath = "OmniVoice/audio_tokenizer_decoder_int8/model.onnx";

    [Header("生成参数")]
    public float guidanceScale = 0f;
    public int topK = 50;
    public int maxNewTokens = 200;

    OmniVoiceLM _lm;
    AudioTokenizer _tokenizer;
    bool _isGenerating = false;

    void Start()
    {
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

    public void CloneVoice() => StartCoroutine(CloneVoiceCoroutine());

    IEnumerator CloneVoiceCoroutine()
    {
        if (_isGenerating) { Debug.LogWarning("正在生成中，请等待。"); yield break; }
        _isGenerating = true;
        Debug.Log("[OmniVoiceRunner] 开始语音克隆...");
        float t0 = Time.realtimeSinceStartup;

        long[,] refCodes = null;
        if (referenceAudio != null)
        {
            float[] refPCM = AudioUtils.AudioClipToPCM(referenceAudio);
            refCodes = _tokenizer.Encode(refPCM);
            Debug.Log($"[OmniVoiceRunner] 参考 codes: [8, {refCodes.GetLength(1)}]");
        }

        // ⚠️ 当前 ONNX 为 Audio-Only 架构，彻底隔离文本流
        int[] promptIds = Array.Empty<int>();

        long[,] generatedCodes = null;
        bool done = false;
        Exception threadEx = null;

        System.Threading.ThreadPool.QueueUserWorkItem(_ =>
        {
            try { generatedCodes = _lm.Generate(promptIds, refCodes); }
            catch (Exception e) { threadEx = e; }
            finally { done = true; }
        });

        while (!done) yield return null;

        if (threadEx != null)
        {
            Debug.LogError($"[OmniVoiceLM] 生成失败: {threadEx}");
            _isGenerating = false; yield break;
        }
        if (generatedCodes == null || generatedCodes.GetLength(1) == 0)
        {
            Debug.LogError("[OmniVoiceRunner] 生成结果为空");
            _isGenerating = false; yield break;
        }

        float[] outputPCM = _tokenizer.Decode(generatedCodes);
        AudioUtils.NormalizeRMS(outputPCM);
        AudioUtils.ApplyFade(outputPCM);

        float elapsed = Time.realtimeSinceStartup - t0;
        float audioDur = outputPCM.Length / 24000f;
        Debug.Log($"[OmniVoiceRunner] 完成: 音频={audioDur:F1}s 耗时={elapsed:F1}s RTF={elapsed / audioDur:F2}");

        var clip = AudioUtils.PCMToAudioClip(outputPCM, "omnivoice_output");
        if (outputAudioSource != null) { outputAudioSource.clip = clip; outputAudioSource.Play(); }

        string savePath = Path.Combine(Application.dataPath, "omnivoice_output.wav");
        AudioUtils.SaveWav(savePath, outputPCM);
        Debug.Log($"[OmniVoiceRunner] 已保存: {savePath}");
        _isGenerating = false;
    }

    void OnDestroy()
    {
        _lm?.Dispose();
        _tokenizer?.Dispose();
    }
}
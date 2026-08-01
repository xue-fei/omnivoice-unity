using System;
using System.IO;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// OmniVoiceRunner 扩展 - 集成音色管理功能
/// 支持：克隆保存、加载已有音色、音色选择 UI
/// </summary>
public class OmniVoiceRunnerWithProfiles : MonoBehaviour
{
    [Header("音频设置")]
    public AudioClip referenceAudio;
    public string referenceText = "";
    [TextArea] public string targetText = "你好，这是使用语音克隆生成的音频。";
    public string targetLanguage = "Chinese";
    public AudioSource outputAudioSource;

    [Header("模型路径（相对 StreamingAssets）")]
    public string lmModelRelPath = "OmniVoice/omnivoice_lm_int8_hq/model.onnx";
    public string encModelRelPath = "OmniVoice/audio_tokenizer_encoder_int8/model.onnx";
    public string decModelRelPath = "OmniVoice/audio_tokenizer_decoder_int8/model.onnx";
    public string tokenizerJsonRelPath = "OmniVoice/tokenizer.json";

    [Header("推理加速")]
    public ExecutionProviderType executionProvider = ExecutionProviderType.CUDA;
    public int deviceId = 0;

    [Header("生成参数")]
    public int numStep = 32;
    public float guidanceScale = 2.0f;
    public float tShift = 0.1f;
    public float positionTemperature = 5.0f;
    public float classTemperature = 0.0f;
    public float layerPenaltyFactor = 5.0f;
    public float targetDurSec = 0f;

    [Header("音色管理 UI（可选）")]
    public Dropdown voiceProfileDropdown;
    public InputField newProfileNameInput;
    public Button saveProfileButton;
    public Button useProfileButton;
    public Button cloneFromAudioButton;

    OmniVoiceLM _lm;
    AudioTokenizer _tokenizer;
    Qwen2Tokenizer _textTok;
    bool _isGenerating;

    // 当前使用的音色 codes（来自音频克隆或加载的配置）
    private long[,] _currentRefCodes;
    private int _currentTRef;
    private float _currentRefRms;
    private string _currentProfileName; // 当前使用的音色名称（null 表示来自音频）

    void Start()
    {
        Application.targetFrameRate = 60;

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
        };

        _tokenizer = new AudioTokenizer(encPath, decPath, executionProvider, deviceId);

        if (File.Exists(tokPath))
        {
            _textTok = Qwen2Tokenizer.Load(tokPath);
        }

        // 初始化音色管理器
        if (VoiceProfileManager.Instance == null)
        {
            var go = new GameObject("VoiceProfileManager");
            go.AddComponent<VoiceProfileManager>();
        }

        // UI 绑定
        SetupUI();

        RefreshVoiceProfileList();
    }

    void SetupUI()
    {
        if (saveProfileButton != null)
            saveProfileButton.onClick.AddListener(OnSaveProfileClicked);
        if (useProfileButton != null)
            useProfileButton.onClick.AddListener(OnUseProfileClicked);
        if (cloneFromAudioButton != null)
            cloneFromAudioButton.onClick.AddListener(OnCloneFromAudioClicked);
    }

    #region UI 回调

    /// <summary>保存当前音色</summary>
    void OnSaveProfileClicked()
    {
        string name = newProfileNameInput != null ? newProfileNameInput.text : "";
        if (string.IsNullOrEmpty(name))
        {
            Debug.LogWarning("[OmniVoice] 请输入音色名称");
            return;
        }

        if (_currentRefCodes == null)
        {
            Debug.LogWarning("[OmniVoice] 没有可保存的音色，请先克隆或加载音色");
            return;
        }

        VoiceProfileManager.Instance.SaveVoiceProfile(name, _currentRefCodes, referenceText);
        RefreshVoiceProfileList();
    }

    /// <summary>使用选中的音色</summary>
    void OnUseProfileClicked()
    {
        if (voiceProfileDropdown == null || voiceProfileDropdown.options.Count == 0) return;

        string name = voiceProfileDropdown.options[voiceProfileDropdown.value].text;
        LoadVoiceProfile(name);
    }

    /// <summary>从参考音频克隆</summary>
    void OnCloneFromAudioClicked()
    {
        CloneVoiceFromAudio();
    }

    #endregion

    #region 音色操作

    /// <summary>加载音色配置</summary>
    public void LoadVoiceProfile(string name)
    {
        long[,] codes = VoiceProfileManager.Instance.LoadCodes(name);
        if (codes == null)
        {
            Debug.LogError($"[OmniVoice] 加载音色 '{name}' 失败");
            return;
        }

        _currentRefCodes = codes;
        _currentTRef = codes.GetLength(1);
        _currentRefRms = -1f; // 加载的音色没有 RMS 信息
        _currentProfileName = name;

        float dur = _currentTRef * 960f / 24000f;
        Debug.Log($"[OmniVoice] 已加载音色 '{name}': {dur:F1}s ({_currentTRef}帧)");
    }

    /// <summary>从参考音频克隆</summary>
    public void CloneVoiceFromAudio()
    {
        if (referenceAudio == null)
        {
            Debug.LogWarning("[OmniVoice] 请先设置参考音频");
            return;
        }
        StartCoroutine(CloneVoiceCoroutine());
    }

    /// <summary>使用已加载的音色直接生成（无需参考音频）</summary>
    public void GenerateWithProfile(string profileName, string text = null)
    {
        LoadVoiceProfile(profileName);
        if (text != null) targetText = text;
        StartCoroutine(GenerateCoroutine());
    }

    #endregion

    #region 生成协程

    /// <summary>从参考音频克隆并生成</summary>
    IEnumerator CloneVoiceCoroutine()
    {
        if (_isGenerating) yield break;
        _isGenerating = true;

        // 编码参考音频
        _currentRefCodes = null;
        _currentTRef = 0;
        _currentRefRms = -1f;
        _currentProfileName = null;

        if (referenceAudio != null)
        {
            float[] refPCM = AudioUtils.AudioClipToPCM(referenceAudio);

            // RMS 归一化
            _currentRefRms = 0f;
            foreach (float s in refPCM) _currentRefRms += s * s;
            _currentRefRms = Mathf.Sqrt(_currentRefRms / refPCM.Length);
            if (_currentRefRms > 0f && _currentRefRms < 0.1f)
            {
                float scale = 0.1f / _currentRefRms;
                for (int i = 0; i < refPCM.Length; i++) refPCM[i] *= scale;
            }

            _currentRefCodes = _tokenizer.Encode(refPCM);
            _currentTRef = _currentRefCodes.GetLength(1);

            // 截断过长音频
            const int MAX_REF = 500;
            if (_currentTRef > MAX_REF)
            {
                var truncated = new long[8, MAX_REF];
                for (int cb = 0; cb < 8; cb++)
                    for (int t = 0; t < MAX_REF; t++)
                        truncated[cb, t] = _currentRefCodes[cb, t];
                _currentRefCodes = truncated;
                _currentTRef = MAX_REF;
            }

            Debug.Log($"[OmniVoice] 参考音频编码完成: {_currentTRef}帧");
        }

        yield return StartCoroutine(GenerateCoroutine());
        _isGenerating = false;
    }

    /// <summary>使用当前 _currentRefCodes 生成音频</summary>
    IEnumerator GenerateCoroutine()
    {
        // 构建文本 prompt
        int[] textTokenIds;
        bool hasRefAudio = _currentRefCodes != null;
        string refTextStr = hasRefAudio && !string.IsNullOrEmpty(referenceText) ? referenceText : null;
        string normalizedTarget = TextNormalizer.Normalize(targetText);

        if (_textTok != null && !string.IsNullOrEmpty(normalizedTarget))
        {
            textTokenIds = _textTok.BuildPrompt(normalizedTarget, targetLanguage,
                instruct: null, refText: refTextStr, hasRefAudio: hasRefAudio);
        }
        else
        {
            textTokenIds = Array.Empty<int>();
        }

        int targetLen = EstimateTargetLen(normalizedTarget, targetLanguage, _currentTRef);

        // 后台推理
        long[,] generatedCodes = null;
        bool done = false;
        Exception err = null;

        System.Threading.ThreadPool.QueueUserWorkItem(_ =>
        {
            try
            {
                generatedCodes = _lm.Generate(textTokenIds, _currentRefCodes, targetLen);
            }
            catch (Exception e) { err = e; }
            finally { done = true; }
        });

        while (!done) yield return null;

        if (err != null)
        {
            Debug.LogError($"[OmniVoice] 生成异常: {err}");
            yield break;
        }

        // 解码
        float[] pcm = _tokenizer.Decode(generatedCodes);

        // 后处理
        if (_currentRefRms >= 0f && _currentRefRms < 0.1f)
        {
            float restoreScale = _currentRefRms / 0.1f;
            for (int i = 0; i < pcm.Length; i++) pcm[i] *= restoreScale;
        }
        else if (_currentRefRms < 0f)
        {
            float peak = 0f;
            foreach (float s in pcm) { float abs = Mathf.Abs(s); if (abs > peak) peak = abs; }
            if (peak > 1e-6f)
                for (int i = 0; i < pcm.Length; i++) pcm[i] = pcm[i] / peak * 0.5f;
        }

        AudioUtils.ApplyFadeOut(pcm);

        var clip = AudioUtils.PCMToAudioClip(pcm, "omnivoice_output");
        if (outputAudioSource != null) { outputAudioSource.clip = clip; outputAudioSource.Play(); }

        string savePath = Path.Combine(Application.dataPath, "omnivoice_output.wav");
        AudioUtils.SaveWav(savePath, pcm);
        Debug.Log($"[OmniVoice] ✅ 生成完成 → {savePath}");
    }

    #endregion

    #region UI 辅助

    void RefreshVoiceProfileList()
    {
        if (voiceProfileDropdown == null) return;

        voiceProfileDropdown.ClearOptions();
        var names = VoiceProfileManager.Instance.GetAllProfileNames();
        voiceProfileDropdown.AddOptions(names);

        if (names.Count == 0)
        {
            voiceProfileDropdown.options.Add(new Dropdown.OptionData("（无保存的音色）"));
        }
    }

    #endregion

    int EstimateTargetLen(string text, string language, int T_ref)
    {
        if (targetDurSec > 0f)
            return Mathf.RoundToInt(targetDurSec * 24000f / 960f);
        if (string.IsNullOrEmpty(text))
            return T_ref > 0 ? T_ref : 100;

        string resolvedLang = _textTok != null ? Qwen2Tokenizer.ResolveLang(language) : language;
        bool isChinese = resolvedLang.StartsWith("zh", StringComparison.OrdinalIgnoreCase);

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
        return Mathf.Max(Mathf.RoundToInt(durSec * 24000f / 960f), 25);
    }

    void OnDestroy()
    {
        _lm?.Dispose();
        _tokenizer?.Dispose();
    }
}

using System;
using System.IO;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using Debug = UnityEngine.Debug;

/// <summary>
/// OmniVoiceRunner 扩展 - 集成音色管理功能
/// 核心功能：从音色文件克隆并生成音频，支持 RTF 计算
/// 异步初始化 + 异步推理，不阻塞主线程
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
    [Tooltip("[OPT-11] 调度加速倍率：1.0=原版，2.0=2倍速，3.0=3倍速。推荐3.0")]
    public float scheduleAcceleration = 3.0f;
    public float targetDurSec = 0f;

    [Header("音色管理 UI")]
    public Dropdown voiceProfileDropdown;
    public InputField newProfileNameInput;
    public Button saveProfileButton;
    public Button loadProfileButton;
    public Button generateWithProfileButton;
    public Button cloneFromAudioButton;
    public Button deleteProfileButton;
    public Text selectedProfileInfo;
    public Text statusText;
    public Text rtfText;

    [Header("状态回调")]
    public UnityEngine.Events.UnityEvent onModelReady;
    public UnityEngine.Events.UnityEvent<string> onModelLoadFailed;

    // RTF 统计
    private RTFStatistics _rtfStats = new RTFStatistics();

    private OmniVoiceLM _lm;
    private AudioTokenizer _tokenizer;
    private Qwen2Tokenizer _textTok;
    private bool _isGenerating;
    private bool _modelReady;

    // 当前使用的音色 codes
    private long[,] _currentRefCodes;
    private int _currentTRef;
    private float _currentRefRms;
    private string _currentProfileName;

    public bool IsReady => _modelReady;
    public bool IsGenerating => _isGenerating;

    void Start()
    {
        Application.targetFrameRate = 60;
        InitializeModelsAsync();
        InitializeVoiceProfileManager();
        SetupUI();
        RefreshVoiceProfileList();
        UpdateStatus("就绪");
        UpdateRTFDisplay();
    }

    private void InitializeModelsAsync()
    {
        Debug.Log("[OmniVoice] 开始异步加载模型...");
        UpdateStatus("正在加载模型...");

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
                    Debug.Log("[OmniVoice] 模型异步加载成功");
                    UpdateStatus("模型已加载");
                    onModelReady?.Invoke();
                });
            }
            catch (Exception e)
            {
                Debug.LogError($"[OmniVoice] 模型加载失败: {e.Message}");
                Loom.QueueOnMainThread(() =>
                {
                    _modelReady = false;
                    UpdateStatus($"模型加载失败: {e.Message}");
                    onModelLoadFailed?.Invoke(e.Message);
                });
            }
        });
    }

    private void InitializeVoiceProfileManager()
    {
        if (VoiceProfileManager.Instance == null)
        {
            var go = new GameObject("VoiceProfileManager");
            go.AddComponent<VoiceProfileManager>();
        }
    }

    private void SetupUI()
    {
        if (saveProfileButton != null)
            saveProfileButton.onClick.AddListener(OnSaveProfileClicked);

        if (loadProfileButton != null)
            loadProfileButton.onClick.AddListener(OnLoadProfileClicked);

        if (generateWithProfileButton != null)
            generateWithProfileButton.onClick.AddListener(OnGenerateWithProfileClicked);

        if (cloneFromAudioButton != null)
            cloneFromAudioButton.onClick.AddListener(OnCloneFromAudioClicked);

        if (deleteProfileButton != null)
            deleteProfileButton.onClick.AddListener(OnDeleteProfileClicked);

        if (voiceProfileDropdown != null)
            voiceProfileDropdown.onValueChanged.AddListener(OnProfileSelected);
    }

    #region UI 回调

    private void OnSaveProfileClicked()
    {
        string name = newProfileNameInput != null ? newProfileNameInput.text : "";
        if (string.IsNullOrEmpty(name))
        {
            Debug.LogWarning("[OmniVoice] 请输入音色名称");
            UpdateStatus("请输入音色名称");
            return;
        }

        if (_currentRefCodes == null)
        {
            Debug.LogWarning("[OmniVoice] 没有可保存的音色");
            UpdateStatus("没有可保存的音色，请先克隆或加载");
            return;
        }

        VoiceProfileManager.Instance.SaveVoiceProfile(name, _currentRefCodes, referenceText);
        RefreshVoiceProfileList();
        UpdateStatus($"音色 '{name}' 已保存");

        if (newProfileNameInput != null)
            newProfileNameInput.text = "";
    }

    private void OnLoadProfileClicked()
    {
        if (voiceProfileDropdown == null || voiceProfileDropdown.options.Count == 0) return;

        string name = voiceProfileDropdown.options[voiceProfileDropdown.value].text;
        if (name == "（无保存的音色）")
        {
            UpdateStatus("没有可用的音色文件");
            return;
        }

        LoadVoiceProfileOnly(name);
    }

    private void OnGenerateWithProfileClicked()
    {
        if (voiceProfileDropdown == null || voiceProfileDropdown.options.Count == 0) return;

        string name = voiceProfileDropdown.options[voiceProfileDropdown.value].text;
        if (name == "（无保存的音色）")
        {
            UpdateStatus("没有可用的音色文件");
            return;
        }

        GenerateWithProfile(name);
    }

    private void OnCloneFromAudioClicked()
    {
        CloneVoiceFromAudio();
    }

    private void OnDeleteProfileClicked()
    {
        if (voiceProfileDropdown == null || voiceProfileDropdown.options.Count == 0) return;

        string name = voiceProfileDropdown.options[voiceProfileDropdown.value].text;
        if (name == "（无保存的音色）") return;

        if (VoiceProfileManager.Instance.DeleteProfile(name))
        {
            RefreshVoiceProfileList();

            if (_currentProfileName == name)
            {
                _currentRefCodes = null;
                _currentTRef = 0;
                _currentProfileName = null;
                UpdateProfileInfo(null);
            }
            UpdateStatus($"音色 '{name}' 已删除");
        }
    }

    private void OnProfileSelected(int index)
    {
        if (voiceProfileDropdown == null || voiceProfileDropdown.options.Count == 0) return;
        string name = voiceProfileDropdown.options[index].text;
        if (name != "（无保存的音色）")
        {
            UpdateProfileInfo(name);
        }
    }

    #endregion

    #region 音色操作

    public void LoadVoiceProfileOnly(string profileName)
    {
        if (string.IsNullOrEmpty(profileName))
        {
            Debug.LogError("[OmniVoice] 音色名称不能为空");
            return;
        }

        long[,] codes = VoiceProfileManager.Instance.LoadCodes(profileName);
        if (codes == null)
        {
            Debug.LogError($"[OmniVoice] 加载音色 '{profileName}' 失败");
            UpdateStatus($"加载音色 '{profileName}' 失败");
            return;
        }

        _currentRefCodes = codes;
        _currentTRef = codes.GetLength(1);
        _currentRefRms = -1f;
        _currentProfileName = profileName;

        float dur = _currentTRef * 960f / 24000f;
        Debug.Log($"[OmniVoice] ✅ 音色 '{profileName}' 已加载: {dur:F1}s ({_currentTRef}帧)");

        UpdateProfileInfo(profileName);
        UpdateStatus($"音色 '{profileName}' 已加载，点击生成按钮生成音频");

        if (voiceProfileDropdown != null)
        {
            int index = voiceProfileDropdown.options.FindIndex(opt => opt.text == profileName);
            if (index >= 0)
                voiceProfileDropdown.value = index;
        }
    }

    public void GenerateWithProfile(string profileName, string text = null)
    {
        if (string.IsNullOrEmpty(profileName))
        {
            Debug.LogError("[OmniVoice] 音色名称不能为空");
            return;
        }

        long[,] codes = VoiceProfileManager.Instance.LoadCodes(profileName);
        if (codes == null)
        {
            Debug.LogError($"[OmniVoice] 加载音色 '{profileName}' 失败");
            UpdateStatus($"加载音色 '{profileName}' 失败");
            return;
        }

        _currentRefCodes = codes;
        _currentTRef = codes.GetLength(1);
        _currentRefRms = -1f;
        _currentProfileName = profileName;

        float dur = _currentTRef * 960f / 24000f;
        Debug.Log($"[OmniVoice] ✅ 音色 '{profileName}' 已加载: {dur:F1}s ({_currentTRef}帧)");
        UpdateProfileInfo(profileName);
        UpdateStatus($"音色 '{profileName}' 已加载，正在生成音频...");

        if (text != null) targetText = text;

        StartCoroutine(GenerateCoroutine());
    }

    public void CloneVoiceFromAudio()
    {
        if (referenceAudio == null)
        {
            Debug.LogWarning("[OmniVoice] 请先设置参考音频");
            UpdateStatus("请先设置参考音频");
            return;
        }
        UpdateStatus("正在编码参考音频...");
        StartCoroutine(CloneVoiceCoroutine());
    }

    #endregion

    #region 生成协程

    private IEnumerator CloneVoiceCoroutine()
    {
        if (_isGenerating) yield break;
        if (_tokenizer == null)
        {
            Debug.LogError("[OmniVoice] 音频编码器未初始化");
            yield break;
        }

        _isGenerating = true;
        _rtfStats.StartEncoding();

        _currentRefCodes = null;
        _currentTRef = 0;
        _currentRefRms = -1f;
        _currentProfileName = null;

        if (referenceAudio != null)
        {
            float[] refPCM = AudioUtils.AudioClipToPCM(referenceAudio);

            _currentRefRms = 0f;
            foreach (float s in refPCM) _currentRefRms += s * s;
            _currentRefRms = Mathf.Sqrt(_currentRefRms / refPCM.Length);
            if (_currentRefRms > 0f && _currentRefRms < 0.1f)
            {
                float scale = 0.1f / _currentRefRms;
                for (int i = 0; i < refPCM.Length; i++) refPCM[i] *= scale;
            }

            // 异步编码
            long[,] encodedCodes = null;
            Exception encodeErr = null;
            bool encodeDone = false;

            Loom.RunAsync(() =>
            {
                try { encodedCodes = _tokenizer.Encode(refPCM); }
                catch (Exception e) { encodeErr = e; }
                finally { encodeDone = true; }
            });

            yield return new WaitUntil(() => encodeDone);

            if (encodeErr != null)
            {
                Debug.LogError($"[OmniVoice] 音频编码失败: {encodeErr.Message}");
                UpdateStatus($"音频编码失败: {encodeErr.Message}");
                _isGenerating = false;
                yield break;
            }

            _currentRefCodes = encodedCodes;
            _currentTRef = _currentRefCodes.GetLength(1);

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

            _rtfStats.EndEncoding();
            Debug.Log($"[OmniVoice] 参考音频编码完成: {_currentTRef}帧, 编码耗时: {_rtfStats.EncodingTimeMs:F1}ms");
            UpdateStatus($"音频编码完成，正在生成...");
            UpdateProfileInfo(null);
        }
        else
        {
            Debug.LogWarning("[OmniVoice] 参考音频为空");
            UpdateStatus("参考音频为空");
            _isGenerating = false;
            yield break;
        }

        yield return StartCoroutine(GenerateCoroutine());
        _isGenerating = false;
    }

    private IEnumerator GenerateCoroutine()
    {
        if (_lm == null || _tokenizer == null)
        {
            Debug.LogError("[OmniVoice] 模型未初始化");
            UpdateStatus("模型未初始化");
            yield break;
        }

        if (_currentRefCodes == null)
        {
            Debug.LogWarning("[OmniVoice] 没有参考音色");
            UpdateStatus("没有参考音色，请先加载或克隆");
            yield break;
        }

        // 开始计时
        _rtfStats.StartGeneration();

        UpdateStatus("正在构建文本...");

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

        UpdateStatus($"正在生成音频 (目标长度: {targetLen}帧)...");

        // 异步推理
        long[,] generatedCodes = null;
        Exception err = null;
        bool inferenceDone = false;

        Loom.RunAsync(() =>
        {
            try
            {
                generatedCodes = _lm.Generate(textTokenIds, _currentRefCodes, targetLen);
            }
            catch (Exception e) { err = e; }
            finally { inferenceDone = true; }
        });

        yield return new WaitUntil(() => inferenceDone);

        // 记录生成结束时间
        _rtfStats.EndGeneration();

        if (err != null)
        {
            Debug.LogError($"[OmniVoice] 生成异常: {err}");
            UpdateStatus($"生成失败: {err.Message}");
            yield break;
        }

        if (generatedCodes == null)
        {
            Debug.LogError("[OmniVoice] 生成结果为 null");
            UpdateStatus("生成结果为 null");
            yield break;
        }

        UpdateStatus("正在解码音频...");

        // 异步解码
        float[] pcm = null;
        bool decodeDone = false;
        Loom.RunAsync(() =>
        {
            try { pcm = _tokenizer.Decode(generatedCodes); }
            catch (Exception e) { Debug.LogError($"[OmniVoice] 解码异常: {e.Message}"); }
            finally { decodeDone = true; }
        });

        yield return new WaitUntil(() => decodeDone);

        _rtfStats.EndDecoding();

        if (pcm == null)
        {
            Debug.LogError("[OmniVoice] 解码失败");
            UpdateStatus("解码失败");
            yield break;
        }

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

        // 计算音频时长
        float audioDuration = (float)pcm.Length / 24000f;
        _rtfStats.SetAudioDuration(audioDuration);

        var clip = AudioUtils.PCMToAudioClip(pcm, "omnivoice_output");
        if (outputAudioSource != null)
        {
            outputAudioSource.clip = clip;
            outputAudioSource.Play();
        }

        string savePath = Path.Combine(Application.dataPath, "omnivoice_output.wav");
        AudioUtils.SaveWav(savePath, pcm);

        string profileInfo = string.IsNullOrEmpty(_currentProfileName) ? "实时克隆" : _currentProfileName;

        // 计算并显示 RTF
        _rtfStats.CalculateRTF();
        UpdateRTFDisplay();

        Debug.Log($"[OmniVoice] ✅ 生成完成 → {savePath} (音色: {profileInfo})");
        Debug.Log($"[OmniVoice] RTF: {_rtfStats.RTF:F3}, 音频时长: {_rtfStats.AudioDuration:F2}s, 总耗时: {_rtfStats.TotalTimeMs:F1}ms");

        UpdateStatus($"✅ 生成完成！RTF: {_rtfStats.RTF:F3} | 音色: {profileInfo}");
    }

    #endregion

    #region RTF 计算与显示

    /// <summary>
    /// 更新 RTF 显示
    /// </summary>
    private void UpdateRTFDisplay()
    {
        if (rtfText == null) return;

        if (_rtfStats.HasData)
        {
            rtfText.text = $"━━━ RTF 性能统计 ━━━\n" +
                          $"RTF: {_rtfStats.RTF:F3}\n" +
                          $"音频时长: {_rtfStats.AudioDuration:F2}s\n" +
                          $"生成耗时: {_rtfStats.GenerationTimeMs:F1}ms\n" +
                          $"解码耗时: {_rtfStats.DecodingTimeMs:F1}ms\n" +
                          $"总耗时: {_rtfStats.TotalTimeMs:F1}ms\n" +
                          $"━━━━━━━━━━━━━━━━━━━\n" +
                          $"{(IsRealtime() ? "✅ 实时 (RTF < 1.0)" : "⏳ 非实时 (RTF > 1.0)")}";
        }
        else
        {
            rtfText.text = "━━━ RTF 性能统计 ━━━\n" +
                          "等待首次生成...";
        }
    }

    /// <summary>
    /// 判断是否满足实时条件
    /// </summary>
    private bool IsRealtime()
    {
        return _rtfStats.HasData && _rtfStats.RTF < 1.0f;
    }

    #endregion

    #region UI 辅助

    private void RefreshVoiceProfileList()
    {
        if (voiceProfileDropdown == null) return;

        voiceProfileDropdown.ClearOptions();
        var names = VoiceProfileManager.Instance.GetAllProfileNames();

        if (names.Count > 0)
        {
            voiceProfileDropdown.AddOptions(names);
        }
        else
        {
            voiceProfileDropdown.options.Add(new Dropdown.OptionData("（无保存的音色）"));
        }

        if (!string.IsNullOrEmpty(_currentProfileName))
        {
            int index = names.IndexOf(_currentProfileName);
            if (index >= 0)
                voiceProfileDropdown.value = index;
        }
    }

    private void UpdateProfileInfo(string profileName)
    {
        if (selectedProfileInfo == null) return;

        if (string.IsNullOrEmpty(profileName))
        {
            if (_currentRefCodes != null && _currentProfileName == null)
            {
                float dur = _currentTRef * 960f / 24000f;
                selectedProfileInfo.text = $"当前音色: 实时克隆 (从音频)\n时长: {dur:F1}s | 帧数: {_currentTRef}";
            }
            else
            {
                selectedProfileInfo.text = "当前音色: 未加载";
            }
            return;
        }

        var profile = VoiceProfileManager.Instance.GetProfile(profileName);
        if (profile != null)
        {
            float dur = profile.frameCount * 960f / 24000f;
            selectedProfileInfo.text = $"音色: {profileName}\n" +
                                      $"时长: {dur:F1}s | 帧数: {profile.frameCount}\n" +
                                      $"创建: {profile.createdAt}\n" +
                                      $"参考: {profile.referenceText}";
        }
    }

    private void UpdateStatus(string message)
    {
        if (statusText != null)
        {
            statusText.text = message;
        }
        Debug.Log($"[OmniVoice] 状态: {message}");
    }

    #endregion

    #region 辅助方法

    private int EstimateTargetLen(string text, string language, int T_ref)
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

    #endregion

    void OnDestroy()
    {
        _lm?.Dispose();
        _tokenizer?.Dispose();
    }
}

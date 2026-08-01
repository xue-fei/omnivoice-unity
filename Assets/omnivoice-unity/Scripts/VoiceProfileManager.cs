using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// 音色数据 - 保存克隆后的音色信息
/// </summary>
[Serializable]
public class VoiceProfile
{
    public string profileName;          // 音色名称（用户自定义）
    public string speakerId;            // 唯一 ID
    public string referenceText;        // 参考音频的文字内容
    public string createdAt;            // 创建时间
    public int frameCount;              // 帧数 T
    public byte[] codesData;            // refCodes 序列化数据 (long[8, T] 扁平化)

    // 新增：存储音色元数据
    public float sampleRate = 24000f;    // 采样率
    public int codebookCount = 8;        // 码本数量
}

/// <summary>
/// 音色管理器 - 保存和加载克隆的音色
/// </summary>
public class VoiceProfileManager : MonoBehaviour
{
    [Header("保存路径")]
    [SerializeField] private string _saveFolder = "VoiceProfiles";

    private string _savePath;
    private Dictionary<string, VoiceProfile> _profiles = new Dictionary<string, VoiceProfile>();
    private List<string> _profileNames = new List<string>();
    private Dictionary<string, long[,]> _cachedCodes = new Dictionary<string, long[,]>(); // 缓存解码后的 codes

    // 单例
    private static VoiceProfileManager _instance;
    public static VoiceProfileManager Instance => _instance;

    void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(gameObject);
            return;
        }
        _instance = this;
        DontDestroyOnLoad(gameObject);

        _savePath = Path.Combine(Application.streamingAssetsPath, _saveFolder);
        Directory.CreateDirectory(_savePath);
        LoadProfileIndex();
    }

    #region 保存音色

    /// <summary>
    /// 保存音色（从 OmniVoiceRunner 克隆完成后调用）
    /// </summary>
    public string SaveVoiceProfile(string name, long[,] refCodes, string refText = "")
    {
        if (refCodes == null || refCodes.GetLength(1) < 10)
        {
            Debug.LogError("[VoiceProfile] refCodes 无效，无法保存");
            return null;
        }

        // 检查名称是否已存在
        if (_profiles.ContainsKey(name))
        {
            Debug.LogWarning($"[VoiceProfile] 音色 '{name}' 已存在，将覆盖");
            // 清除缓存
            if (_cachedCodes.ContainsKey(name))
                _cachedCodes.Remove(name);
        }

        string speakerId = Guid.NewGuid().ToString("N")[..8];
        int T = refCodes.GetLength(1);

        // 序列化 long[8, T] 为 byte[]
        byte[] data = SerializeCodes(refCodes);

        var profile = new VoiceProfile
        {
            profileName = name,
            speakerId = speakerId,
            referenceText = refText,
            createdAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
            frameCount = T,
            codesData = data,
            codebookCount = 8,
            sampleRate = 24000f
        };

        // 保存到文件
        string fileName = $"{speakerId}_{SanitizeFileName(name)}.json";
        string filePath = Path.Combine(_savePath, fileName);
        string json = JsonUtility.ToJson(profile, false);
        File.WriteAllText(filePath, json);

        // 更新索引和缓存
        _profiles[name] = profile;
        _cachedCodes[name] = refCodes; // 缓存解码后的数据
        if (!_profileNames.Contains(name)) _profileNames.Add(name);
        SaveProfileIndex();

        float dur = T * 960f / 24000f;
        Debug.Log($"[VoiceProfile] ✅ 音色 '{name}' 已保存: {dur:F1}s ({T}帧) {data.Length}字节 → {filePath}");
        return speakerId;
    }

    /// <summary>
    /// 从参考音频编码并保存（便捷方法）
    /// </summary>
    public string SaveFromAudio(string name, AudioClip audio, AudioTokenizer tokenizer, string refText = "")
    {
        float[] pcm = AudioUtils.AudioClipToPCM(audio);
        long[,] codes = tokenizer.Encode(pcm);
        return SaveVoiceProfile(name, codes, refText);
    }

    #endregion

    #region 加载音色（核心：从音色文件克隆）

    /// <summary>
    /// 【核心方法】从音色文件加载 refCodes（用于 Generate）
    /// 这就是"从音色文件克隆"的本质：直接加载预计算的 codes，无需重新编码
    /// </summary>
    public long[,] LoadCodes(string name, bool useCache = true)
    {
        // 优先从缓存获取
        if (useCache && _cachedCodes.TryGetValue(name, out long[,] cached))
        {
            Debug.Log($"[VoiceProfile] 从缓存加载音色 '{name}'");
            return cached;
        }

        VoiceProfile profile = GetProfile(name);
        if (profile == null) return null;

        try
        {
            long[,] codes = DeserializeCodes(profile.codesData, profile.frameCount);

            // 存入缓存
            if (useCache)
                _cachedCodes[name] = codes;

            float dur = profile.frameCount * 960f / 24000f;
            Debug.Log($"[VoiceProfile] ✅ 从文件加载音色 '{name}': {dur:F1}s ({profile.frameCount}帧)");

            return codes;
        }
        catch (Exception e)
        {
            Debug.LogError($"[VoiceProfile] 加载音色 '{name}' 失败: {e.Message}");
            return null;
        }
    }

    /// <summary>
    /// 获取音色的元数据（不加载 codes）
    /// </summary>
    public VoiceProfile GetProfile(string name)
    {
        if (_profiles.TryGetValue(name, out var profile))
            return profile;
        return null;
    }

    /// <summary>
    /// 获取所有音色名称
    /// </summary>
    public List<string> GetAllProfileNames() => new List<string>(_profileNames);

    /// <summary>
    /// 检查音色是否存在
    /// </summary>
    public bool HasProfile(string name) => _profiles.ContainsKey(name);

    /// <summary>
    /// 获取音色的帧数
    /// </summary>
    public int GetProfileFrameCount(string name)
    {
        var profile = GetProfile(name);
        return profile?.frameCount ?? 0;
    }

    /// <summary>
    /// 获取音色的时长（秒）
    /// </summary>
    public float GetProfileDuration(string name)
    {
        var profile = GetProfile(name);
        if (profile == null) return 0f;
        return profile.frameCount * 960f / 24000f;
    }

    #endregion

    #region 删除音色

    /// <summary>
    /// 删除音色
    /// </summary>
    public bool DeleteProfile(string name)
    {
        VoiceProfile profile = GetProfile(name);
        if (profile == null) return false;

        // 删除文件
        string fileName = $"{profile.speakerId}_{SanitizeFileName(name)}.json";
        string filePath = Path.Combine(_savePath, fileName);
        if (File.Exists(filePath)) File.Delete(filePath);

        // 更新索引和缓存
        _profiles.Remove(name);
        _profileNames.Remove(name);
        if (_cachedCodes.ContainsKey(name))
            _cachedCodes.Remove(name);

        SaveProfileIndex();

        Debug.Log($"[VoiceProfile] 音色 '{name}' 已删除");
        return true;
    }

    #endregion

    #region 序列化工具

    /// <summary>
    /// 将 long[8, T] 序列化为 byte[]
    /// </summary>
    private byte[] SerializeCodes(long[,] codes)
    {
        int cb = codes.GetLength(0); // 8
        int T = codes.GetLength(1);
        byte[] data = new byte[cb * T * sizeof(long)];
        Buffer.BlockCopy(codes, 0, data, 0, data.Length);
        return data;
    }

    /// <summary>
    /// 从 byte[] 反序列化出 long[8, T]
    /// </summary>
    private long[,] DeserializeCodes(byte[] data, int T)
    {
        int cb = 8;
        long[,] codes = new long[cb, T];
        Buffer.BlockCopy(data, 0, codes, 0, data.Length);
        return codes;
    }

    #endregion

    #region 索引管理

    [Serializable]
    private class ProfileIndex
    {
        public List<string> names = new List<string>();
    }

    private void LoadProfileIndex()
    {
        string indexPath = Path.Combine(_savePath, "index.json");
        _profiles.Clear();
        _profileNames.Clear();
        _cachedCodes.Clear();

        if (!File.Exists(indexPath)) return;

        try
        {
            string json = File.ReadAllText(indexPath);
            var index = JsonUtility.FromJson<ProfileIndex>(json);
            if (index?.names != null)
            {
                foreach (string name in index.names)
                {
                    VoiceProfile profile = LoadProfileFromFile(name);
                    if (profile != null)
                    {
                        _profiles[name] = profile;
                        _profileNames.Add(name);
                    }
                }
            }
            Debug.Log($"[VoiceProfile] 加载了 {_profileNames.Count} 个音色");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[VoiceProfile] 索引加载失败: {e.Message}");
        }
    }

    private VoiceProfile LoadProfileFromFile(string name)
    {
        // 在目录中搜索匹配的文件
        string[] files = Directory.GetFiles(_savePath, $"*_{SanitizeFileName(name)}.json");
        if (files.Length == 0) return null;

        try
        {
            string json = File.ReadAllText(files[0]);
            return JsonUtility.FromJson<VoiceProfile>(json);
        }
        catch
        {
            return null;
        }
    }

    private void SaveProfileIndex()
    {
        var index = new ProfileIndex { names = _profileNames };
        string json = JsonUtility.ToJson(index, false);
        File.WriteAllText(Path.Combine(_savePath, "index.json"), json);
    }

    #endregion

    private string SanitizeFileName(string name)
    {
        foreach (char c in Path.GetInvalidFileNameChars())
            name = name.Replace(c, '_');
        return name.Length > 50 ? name[..50] : name;
    }
}
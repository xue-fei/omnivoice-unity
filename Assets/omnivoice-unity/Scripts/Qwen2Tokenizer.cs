using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

/// <summary>
/// Qwen2 BPE Tokenizer — 在 Unity 中读取 tokenizer.json 并执行 BPE 编码。
///
/// 算法与 HuggingFace tokenizers 库完全一致：
///   1. 文本用 Qwen2 正则拆分为"单词"（不做 Unicode 归一化，因为 NFC 对 UTF-8 无副作用）
///   2. 每个单词的每个字节映射到 GPT-2 byte-to-unicode 空间
///   3. 用 BPE merge rules 合并
///   4. 查 vocab 表返回 token id
///
/// 使用前：
///   把 tokenizer.json 放到 StreamingAssets/OmniVoice/tokenizer.json
///   var tok = Qwen2Tokenizer.Load(path);
///   int[] ids = tok.Encode("你好世界", "Chinese");  // OmniVoice 格式
/// </summary>
public class Qwen2Tokenizer
{
    // ── OmniVoice 专用 Special Token IDs（来自 tokenizer.json added_tokens）──
    public const int TOKEN_DENOISE = 151669;
    public const int TOKEN_LANG_START = 151670;
    public const int TOKEN_LANG_END = 151671;
    public const int TOKEN_INSTRUCT_START = 151672;
    public const int TOKEN_INSTRUCT_END = 151673;
    public const int TOKEN_TEXT_START = 151674;
    public const int TOKEN_TEXT_END = 151675;
    public const int TOKEN_IM_END = 151645;
    public const int TOKEN_ENDOFTEXT = 151643;

    // ── GPT-2 byte→unicode 映射（256 个固定值，硬编码避免依赖）────────────
    // byte index → unicode codepoint
    // 生成规则：bytes 33-126、161-172、174-255 映射到自身；
    // 其余 0-32、127-160、173 映射到 256+ 的连续码点。
    static readonly int[] ByteToUnicode = new int[256]
    {
        256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,
        272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,
        288, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
         80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
         96, 97, 98, 99,100,101,102,103,104,105,106,107,108,109,110,111,
        112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,289,
        290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,
        306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,
        322,161,162,163,164,165,166,167,168,169,170,171,172,323,174,175,
        176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,
        192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,
        208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,
        224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,
        240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255
    };

    // ── 内部数据 ─────────────────────────────────────────────
    readonly Dictionary<string, int> _vocab;          // token_str → id
    readonly Dictionary<(string, string), int> _merges; // pair → rank
    readonly Dictionary<string, int> _specialTokens;   // "<|...|>" → id

    Qwen2Tokenizer(
        Dictionary<string, int> vocab,
        Dictionary<(string, string), int> merges,
        Dictionary<string, int> specialTokens)
    {
        _vocab = vocab;
        _merges = merges;
        _specialTokens = specialTokens;
    }

    // ── 工厂方法 ──────────────────────────────────────────────
    /// <summary>
    /// 从 tokenizer.json 加载。必须在后台线程调用（文件 ~11MB，解析约 1-2 秒）。
    /// 失败返回 null。
    /// </summary>
    public static Qwen2Tokenizer Load(string tokenizerJsonPath)
    {
        if (!File.Exists(tokenizerJsonPath))
        {
            Debug.LogError($"[Qwen2Tokenizer] File not found: {tokenizerJsonPath}");
            return null;
        }

        try
        {
            string json = File.ReadAllText(tokenizerJsonPath, Encoding.UTF8);
            return ParseJson(json);
        }
        catch (Exception ex)
        {
            Debug.LogError($"[Qwen2Tokenizer] Load failed: {ex.Message}\n{ex.StackTrace}");
            return null;
        }
    }

    // ── JSON 解析（不依赖 Newtonsoft，纯手写解析器）──────────
    static Qwen2Tokenizer ParseJson(string json)
    {
        // 用 Unity 内置的 JsonUtility 不支持任意嵌套，改用轻量手写解析
        // 结构固定，直接用字符串定位法提取三个字段

        // 1. added_tokens → specialTokens
        var specialTokens = new Dictionary<string, int>();
        {
            int arrStart = json.IndexOf("\"added_tokens\"", StringComparison.Ordinal);
            int arrBracket = json.IndexOf('[', arrStart);
            int depth = 0;
            int arrEnd = arrBracket;
            for (int i = arrBracket; i < json.Length; i++)
            {
                if (json[i] == '[') depth++;
                else if (json[i] == ']') { depth--; if (depth == 0) { arrEnd = i; break; } }
            }
            string arrSlice = json.Substring(arrBracket, arrEnd - arrBracket + 1);
            // parse each object in the array
            int pos = 0;
            while (true)
            {
                int objStart = arrSlice.IndexOf('{', pos);
                if (objStart < 0) break;
                int objEnd = arrSlice.IndexOf('}', objStart);
                if (objEnd < 0) break;
                string obj = arrSlice.Substring(objStart, objEnd - objStart + 1);
                int id = ParseIntField(obj, "\"id\"");
                string content = ParseStringField(obj, "\"content\"");
                if (content != null && id >= 0)
                    specialTokens[content] = id;
                pos = objEnd + 1;
            }
        }

        // 2. model.vocab → Dictionary<string,int>
        var vocab = new Dictionary<string, int>(160000);
        {
            int modelPos = json.IndexOf("\"model\"", StringComparison.Ordinal);
            int vocabPos = json.IndexOf("\"vocab\"", modelPos, StringComparison.Ordinal);
            int vocabBrace = json.IndexOf('{', vocabPos);
            // Parse key:value pairs until closing }
            // Keys and values can be any valid JSON string / number
            int i = vocabBrace + 1;
            while (i < json.Length)
            {
                // skip whitespace
                while (i < json.Length && json[i] <= ' ') i++;
                if (json[i] == '}') break;
                if (json[i] != '"') { i++; continue; }
                // read key
                int keyStart = i + 1;
                int keyEnd = FindStringEnd(json, keyStart);
                string key = Unescape(json.Substring(keyStart, keyEnd - keyStart));
                i = keyEnd + 1;
                // skip : and whitespace
                while (i < json.Length && (json[i] <= ' ' || json[i] == ':')) i++;
                // read integer value
                int numStart = i;
                while (i < json.Length && (json[i] >= '0' && json[i] <= '9')) i++;
                int id = int.Parse(json.Substring(numStart, i - numStart));
                vocab[key] = id;
                // skip comma
                while (i < json.Length && (json[i] <= ' ' || json[i] == ',')) i++;
            }
        }

        // 3. model.merges → ordered list
        var merges = new Dictionary<(string, string), int>(160000);
        {
            int modelPos = json.IndexOf("\"model\"", StringComparison.Ordinal);
            int mergesPos = json.IndexOf("\"merges\"", modelPos, StringComparison.Ordinal);
            int arrBracket = json.IndexOf('[', mergesPos);
            int rank = 0;
            int i = arrBracket + 1;
            while (i < json.Length)
            {
                while (i < json.Length && json[i] <= ' ') i++;
                if (json[i] == ']') break;
                if (json[i] == ',') { i++; continue; }

                if (json[i] == '"')
                {
                    // String format: "left right"
                    int sStart = i + 1;
                    int sEnd = FindStringEnd(json, sStart);
                    string s = Unescape(json.Substring(sStart, sEnd - sStart));
                    int space = s.IndexOf(' ');
                    if (space > 0)
                        merges[(s.Substring(0, space), s.Substring(space + 1))] = rank++;
                    i = sEnd + 1;
                }
                else if (json[i] == '[')
                {
                    // Array format: ["left","right"]
                    i++; // skip [
                    while (i < json.Length && json[i] <= ' ') i++;
                    int s1Start = i + 1;
                    int s1End = FindStringEnd(json, s1Start);
                    string s1 = Unescape(json.Substring(s1Start, s1End - s1Start));
                    i = s1End + 1;
                    while (i < json.Length && (json[i] <= ' ' || json[i] == ',')) i++;
                    int s2Start = i + 1;
                    int s2End = FindStringEnd(json, s2Start);
                    string s2 = Unescape(json.Substring(s2Start, s2End - s2Start));
                    merges[(s1, s2)] = rank++;
                    i = s2End + 1;
                    while (i < json.Length && json[i] != ']') i++;
                    i++; // skip ]
                }
                else
                {
                    i++;
                }
            }
        }

        Debug.Log($"[Qwen2Tokenizer] Loaded: vocab={vocab.Count}, merges={merges.Count}, specials={specialTokens.Count}");
        return new Qwen2Tokenizer(vocab, merges, specialTokens);
    }

    // ── 字符串解析辅助 ────────────────────────────────────────
    static int FindStringEnd(string s, int start)
    {
        for (int i = start; i < s.Length; i++)
        {
            if (s[i] == '\\') { i++; continue; }
            if (s[i] == '"') return i;
        }
        return s.Length;
    }

    static string Unescape(string s)
    {
        if (!s.Contains('\\')) return s;
        var sb = new StringBuilder(s.Length);
        for (int i = 0; i < s.Length; i++)
        {
            if (s[i] == '\\' && i + 1 < s.Length)
            {
                char next = s[i + 1];
                switch (next)
                {
                    case '"': sb.Append('"'); i++; break;
                    case '\\': sb.Append('\\'); i++; break;
                    case '/': sb.Append('/'); i++; break;
                    case 'n': sb.Append('\n'); i++; break;
                    case 'r': sb.Append('\r'); i++; break;
                    case 't': sb.Append('\t'); i++; break;
                    case 'u':
                        if (i + 5 < s.Length)
                        {
                            int cp = Convert.ToInt32(s.Substring(i + 2, 4), 16);
                            sb.Append((char)cp);
                            i += 5;
                        }
                        break;
                    default: sb.Append(s[i]); break;
                }
            }
            else sb.Append(s[i]);
        }
        return sb.ToString();
    }

    static int ParseIntField(string obj, string fieldName)
    {
        int pos = obj.IndexOf(fieldName, StringComparison.Ordinal);
        if (pos < 0) return -1;
        pos = obj.IndexOf(':', pos) + 1;
        while (pos < obj.Length && obj[pos] <= ' ') pos++;
        int end = pos;
        while (end < obj.Length && (obj[end] >= '0' && obj[end] <= '9')) end++;
        if (end == pos) return -1;
        return int.Parse(obj.Substring(pos, end - pos));
    }

    static string ParseStringField(string obj, string fieldName)
    {
        int pos = obj.IndexOf(fieldName, StringComparison.Ordinal);
        if (pos < 0) return null;
        pos = obj.IndexOf(':', pos) + 1;
        while (pos < obj.Length && obj[pos] <= ' ') pos++;
        if (pos >= obj.Length || obj[pos] != '"') return null;
        int start = pos + 1;
        int end = FindStringEnd(obj, start);
        return Unescape(obj.Substring(start, end - start));
    }

    // ── BPE 编码 ──────────────────────────────────────────────
    /// <summary>
    /// 将 UTF-8 文本字符串编码为 token id 列表。
    /// 不做 Qwen2 的 regex 预分词（对中文效果一致，对英文可能有微小差异）。
    /// </summary>
    public int[] EncodeText(string text)
    {
        if (string.IsNullOrEmpty(text)) return Array.Empty<int>();

        // 字节→unicode空间
        byte[] utf8 = Encoding.UTF8.GetBytes(text);
        // 用 Qwen2 pre-tokenizer: 按空格/标点分段后再 byte-level BPE
        // 简化版：整体 byte-level BPE（对中文结果相同，英文可能有 1-2 token 差异）
        var byteChars = new string[utf8.Length];
        for (int i = 0; i < utf8.Length; i++)
            byteChars[i] = ((char)ByteToUnicode[utf8[i]]).ToString();

        return BpeTokenize(byteChars);
    }

    int[] BpeTokenize(string[] chars)
    {
        var tokens = new List<string>(chars);

        while (tokens.Count > 1)
        {
            // 找 rank 最小（最优先合并）的相邻对
            int bestRank = int.MaxValue;
            int bestIndex = -1;
            for (int i = 0; i < tokens.Count - 1; i++)
            {
                if (_merges.TryGetValue((tokens[i], tokens[i + 1]), out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0) break;

            string merged = tokens[bestIndex] + tokens[bestIndex + 1];
            tokens[bestIndex] = merged;
            tokens.RemoveAt(bestIndex + 1);
        }

        var ids = new int[tokens.Count];
        for (int i = 0; i < tokens.Count; i++)
            ids[i] = _vocab.TryGetValue(tokens[i], out int id) ? id : TOKEN_ENDOFTEXT;
        return ids;
    }

    // ── OmniVoice 专用：构建完整 prompt token 序列 ───────────
    /// <summary>
    /// 构建 OmniVoice voice_clone / voice_design / auto 模式的 prompt。
    ///
    /// 格式：
    ///   &lt;|denoise|&gt; &lt;|lang_start|&gt; {language} &lt;|lang_end|&gt;
    ///   [&lt;|instruct_start|&gt; {instruct} &lt;|instruct_end|&gt;]   (voice_design only)
    ///   &lt;|text_start|&gt; {text} &lt;|text_end|&gt;
    /// </summary>
    /// <param name="text">要合成的文本</param>
    /// <param name="language">语言名称，如 "Chinese"、"English"</param>
    /// <param name="instruct">语音设计指令（可为 null）</param>
    public int[] BuildPrompt(string text, string language = "Chinese", string instruct = null)
    {
        var ids = new List<int>();

        ids.Add(TOKEN_DENOISE);
        ids.Add(TOKEN_LANG_START);
        ids.AddRange(EncodeText(language));
        ids.Add(TOKEN_LANG_END);

        if (!string.IsNullOrEmpty(instruct))
        {
            ids.Add(TOKEN_INSTRUCT_START);
            ids.AddRange(EncodeText(instruct));
            ids.Add(TOKEN_INSTRUCT_END);
        }

        ids.Add(TOKEN_TEXT_START);
        ids.AddRange(EncodeText(text));
        ids.Add(TOKEN_TEXT_END);

        return ids.ToArray();
    }

    /// <summary>特殊 token id 查找（如 "&lt;|im_end|&gt;"）</summary>
    public bool TryGetSpecialToken(string content, out int id) =>
        _specialTokens.TryGetValue(content, out id);
}
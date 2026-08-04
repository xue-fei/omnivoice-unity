using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

/// <summary>
/// 中文文本归一化（Text Normalization），对标 WeTextProcessing tn/chinese 规则集。
/// </summary>
public static class TextNormalizer
{
    // ──────────────────────────────────────────────
    //  公开入口
    // ──────────────────────────────────────────────

    public static string Normalize(string text,
                                   bool removeErhua = false,
                                   bool removePunct = false)
    {
        if (string.IsNullOrEmpty(text)) return text;

        // 儿化音需在全角转换之前处理（全角逗号转换后会破坏 lookahead）
        if (removeErhua) text = RemoveErhua(text);

        // 全角 → 半角（注意："，。！？、；：" 等常见中文标点保留）
        text = FullwidthToHalf(text);

        // 时间（需在 cardinal/measure 之前，避免 "3:30" 被当数字处理）
        text = NormalizeTime(text);

        // 日期（年月日）
        text = NormalizeDate(text);

        // 电话号码
        text = NormalizeTelephone(text);

        // 分数
        text = NormalizeFraction(text);

        // 百分比
        text = NormalizePercentage(text);

        // 金额（带货币符号）
        text = NormalizeMoney(text);

        // 带单位的量（measure）
        text = NormalizeMeasure(text);

        // 纯数字（cardinal / digit）—— 兜底
        text = NormalizeCardinalAndDigit(text);

        // 标点
        if (removePunct) text = RemovePunctuation(text);

        // 多余空格
        text = Regex.Replace(text, @"\s{2,}", " ").Trim();

        return text;
    }

    // ──────────────────────────────────────────────
    //  基础数字转换
    // ──────────────────────────────────────────────

    private static readonly string[] Digits = { "零", "一", "二", "三", "四", "五", "六", "七", "八", "九" };
    private static readonly string[] DigitsLiang = { "零", "一", "两", "三", "四", "五", "六", "七", "八", "九" };
    private static readonly string[] Units = { "", "十", "百", "千" };
    private static readonly string[] BigUnits = { "", "万", "亿", "万亿" };

    public static string IntegerToChinese(long n, bool useLiang = false)
    {
        if (n == 0) return "零";
        if (n < 0) return "负" + IntegerToChinese(-n, useLiang);

        var d = useLiang ? DigitsLiang : Digits;

        var groups = new List<int>();
        long tmp = n;
        while (tmp > 0) { groups.Add((int)(tmp % 10000)); tmp /= 10000; }

        var sb = new StringBuilder();
        for (int gi = groups.Count - 1; gi >= 0; gi--)
        {
            int g = groups[gi];
            if (g == 0)
            {
                if (sb.Length > 0 && HasNonZeroBelow(groups, gi))
                    sb.Append("零");
                continue;
            }

            if (sb.Length > 0 && g < 1000)
                sb.Append("零");

            if (g == 2 && gi >= 1)
                sb.Append("两");
            else
                sb.Append(GroupToChinese(g, d));
            sb.Append(BigUnits[gi]);
        }

        string result = sb.ToString();
        result = FixYiShi(result);
        return result;
    }

    private static string FixYiShi(string chineseNum)
    {
        // 将“一十”开头的“一”去掉，但“一十万”、“一十亿”中的“一”保留
        return Regex.Replace(chineseNum, @"^一十(?!万|亿)", "十");
    }

    private static bool HasNonZeroBelow(List<int> groups, int gi)
    {
        for (int i = gi - 1; i >= 0; i--)
            if (groups[i] != 0) return true;
        return false;
    }

    private static string GroupToChinese(int g, string[] d, bool useGroupLiang = true)
    {
        var sb = new StringBuilder();
        int[] parts = { g / 1000, (g / 100) % 10, (g / 10) % 10, g % 10 };
        bool prevZero = false;
        for (int i = 0; i < 4; i++)
        {
            int digit = parts[i];
            if (digit == 0) { prevZero = true; }
            else
            {
                if (prevZero && sb.Length > 0) sb.Append("零");
                string digitStr = (useGroupLiang && digit == 2 && i == 0) ? "两" : Digits[digit];
                sb.Append(digitStr);
                sb.Append(Units[3 - i]);
                prevZero = false;
            }
        }
        return sb.ToString();
    }

    private static string DecimalPartToChinese(string decStr)
    {
        var sb = new StringBuilder();
        foreach (char c in decStr)
            if (c >= '0' && c <= '9') sb.Append(Digits[c - '0']);
        return sb.ToString();
    }

    public static string NumberToChinese(string numStr, bool useLiang = false)
    {
        if (string.IsNullOrEmpty(numStr)) return numStr;
        // 移除千分位逗号
        numStr = numStr.Replace(",", "");
        numStr = numStr.Trim();
        bool negative = numStr.StartsWith("-");
        if (negative) numStr = numStr.Substring(1);
        string prefix = negative ? "负" : "";

        int dotIdx = numStr.IndexOf('.');
        if (dotIdx >= 0)
        {
            string intPart = numStr.Substring(0, dotIdx);
            string decPart = numStr.Substring(dotIdx + 1);
            string intCh = string.IsNullOrEmpty(intPart) || intPart == "0"
                ? "零"
                : IntegerToChinese(long.Parse(intPart), useLiang);
            return prefix + intCh + "点" + DecimalPartToChinese(decPart);
        }
        if (!long.TryParse(numStr, out long val)) return prefix + numStr;
        return prefix + IntegerToChinese(val, useLiang);
    }

    public static string NumberToDigitsChinese(string numStr)
    {
        var sb = new StringBuilder();
        foreach (char c in numStr)
            sb.Append(c >= '0' && c <= '9' ? Digits[c - '0'] : c.ToString());
        return sb.ToString();
    }

    // ──────────────────────────────────────────────
    //  1. Cardinal & Digit（兜底）
    // ──────────────────────────────────────────────

    private static readonly Regex _digitContextRe = new Regex(
        @"(?:编号|号码|序列号|工号|学号|房号|座位号)\s*(\d+)",
        RegexOptions.Compiled);

    private static readonly Regex _cardinalRe = new Regex(
        @"-?\d+(?:\.\d+)?",
        RegexOptions.Compiled);

    private static string NormalizeCardinalAndDigit(string text)
    {
        text = _digitContextRe.Replace(text, m =>
        {
            int numStart = m.Value.Length - m.Groups[1].Length;
            return m.Value.Substring(0, numStart) + NumberToDigitsChinese(m.Groups[1].Value);
        });
        text = _cardinalRe.Replace(text, m => NumberToChinese(m.Value));
        return text;
    }

    // ──────────────────────────────────────────────
    //  2. Date
    // ──────────────────────────────────────────────

    /// <summary>
    /// 年份转换：4位年份逐位读（2024→二零二四），2-3位年份常规读（105→一百零五）
    /// </summary>
    private static string ConvertYear(string yearStr)
    {
        if (yearStr.Length == 4)
            return YearToChinese(yearStr);   // 逐位读
        else
            return IntegerToChinese(long.Parse(yearStr)); // 常规读
    }

    private static string NormalizeDate(string text)
    {
        text = Regex.Replace(text,
            @"(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})(?!\d)",
            m => ConvertYear(m.Groups[1].Value) + "年"
               + IntegerToChinese(int.Parse(m.Groups[2].Value)) + "月"
               + IntegerToChinese(int.Parse(m.Groups[3].Value)) + "日");

        text = Regex.Replace(text,
            @"(\d{2,4})年(\d{1,2})月(\d{1,2})[日号]?",
            m => ConvertYear(m.Groups[1].Value) + "年"
               + IntegerToChinese(int.Parse(m.Groups[2].Value)) + "月"
               + IntegerToChinese(int.Parse(m.Groups[3].Value)) + "日");

        text = Regex.Replace(text,
            @"(\d{2,4})年(\d{1,2})月",
            m => ConvertYear(m.Groups[1].Value) + "年"
               + IntegerToChinese(int.Parse(m.Groups[2].Value)) + "月");

        text = Regex.Replace(text,
            @"(\d{2,4})年",
            m => ConvertYear(m.Groups[1].Value) + "年");

        text = Regex.Replace(text,
            @"(\d{1,2})月(\d{1,2})[日号]",
            m => IntegerToChinese(int.Parse(m.Groups[1].Value)) + "月"
               + IntegerToChinese(int.Parse(m.Groups[2].Value)) + "日");

        text = Regex.Replace(text,
            @"(\d{1,2})月(?!\d)",
            m => IntegerToChinese(int.Parse(m.Groups[1].Value)) + "月");

        text = Regex.Replace(text,
            @"(\d{1,2})[日号](?!\d)",
            m => IntegerToChinese(int.Parse(m.Groups[1].Value)) + "日");

        return text;
    }

    private static string YearToChinese(string yearStr)
    {
        var sb = new StringBuilder();
        foreach (char c in yearStr)
            if (c >= '0' && c <= '9') sb.Append(Digits[c - '0']);
        return sb.ToString();
    }

    // ──────────────────────────────────────────────
    //  3. Time
    // ──────────────────────────────────────────────

    private static readonly Regex _timeRe = new Regex(
        @"(?<!\d)([0-9]{1,2})[:：]([0-9]{1,2})(?:[:：]([0-9]{1,2})秒?)?(?!\d)",
        RegexOptions.Compiled);

    private static string NormalizeTime(string text)
    {
        return _timeRe.Replace(text, m =>
        {
            string h = IntegerToChinese(int.Parse(m.Groups[1].Value)) + "时";
            string min = IntegerToChinese(int.Parse(m.Groups[2].Value)) + "分";
            string sec = m.Groups[3].Success
                ? IntegerToChinese(int.Parse(m.Groups[3].Value)) + "秒"
                : "";
            return h + min + sec;
        });
    }

    // ──────────────────────────────────────────────
    //  4. Telephone
    // ──────────────────────────────────────────────

    private static readonly Regex _mobileRe = new Regex(
        @"(?<!\d)(1[3-9]\d{9})(?!\d)",
        RegexOptions.Compiled);

    private static readonly Regex _telRe = new Regex(
        @"(?<!\d)(\d{3,4})-(\d{7,8})(?!\d)",
        RegexOptions.Compiled);

    private static string NormalizeTelephone(string text)
    {
        text = _mobileRe.Replace(text, m => NumberToDigitsChinese(m.Groups[1].Value));
        text = _telRe.Replace(text, m =>
            NumberToDigitsChinese(m.Groups[1].Value) + "，" +
            NumberToDigitsChinese(m.Groups[2].Value));
        return text;
    }

    // ──────────────────────────────────────────────
    //  5. Fraction
    // ──────────────────────────────────────────────

    private static readonly Regex _fractionRe = new Regex(
        @"(\d+)/(\d+)", RegexOptions.Compiled);

    private static string NormalizeFraction(string text)
    {
        return _fractionRe.Replace(text, m =>
        {
            long denominator = long.Parse(m.Groups[2].Value);
            if (denominator == 0) return m.Value;
            return IntegerToChinese(denominator) + "分之" +
                   IntegerToChinese(long.Parse(m.Groups[1].Value));
        });
    }

    // ──────────────────────────────────────────────
    //  6. Percentage
    // ──────────────────────────────────────────────

    private static readonly Regex _percentRe = new Regex(
        @"(\d+(?:\.\d+)?)\s*[%％]", RegexOptions.Compiled);

    private static string NormalizePercentage(string text)
    {
        return _percentRe.Replace(text, m =>
            "百分之" + NumberToChinese(m.Groups[1].Value));
    }

    // ──────────────────────────────────────────────
    //  7. Money
    // ──────────────────────────────────────────────

    private static readonly Dictionary<string, string> _currencyMap =
        new Dictionary<string, string>
        {
            { "¥", "人民币" }, { "￥", "人民币" }, { "$", "美元" },
            { "€", "欧元" },  { "£", "英镑" },   { "₩", "韩元" },
            { "₹", "印度卢比" },
        };

    private static readonly Regex _moneyRe = new Regex(
        @"([¥￥$€£₩₹])\s*(\d+(?:\.\d+)?)\s*(万亿|万|亿|千|百|元|块|角|毛|分)?",
        RegexOptions.Compiled);

    private static readonly Regex _moneyZhRe = new Regex(
        @"(\d+(?:\.\d+)?)\s*(万亿元?|亿元?|万元?|千元?|百元?|元|块钱?|毛|角|分)",
        RegexOptions.Compiled);

    private static string NormalizeMoney(string text)
    {
        text = _moneyRe.Replace(text, m =>
        {
            string currency = _currencyMap.TryGetValue(m.Groups[1].Value, out var c) ? c : m.Groups[1].Value;
            string amount = NumberToChinese(m.Groups[2].Value, useLiang: true);
            string unit = m.Groups[3].Success ? m.Groups[3].Value : "元";
            return currency + amount + unit;
        });
        text = _moneyZhRe.Replace(text, m =>
            NumberToChinese(m.Groups[1].Value, useLiang: true) + m.Groups[2].Value);
        return text;
    }

    // ──────────────────────────────────────────────
    //  8. Measure
    // ──────────────────────────────────────────────

    private static readonly string[] _measureUnits =
    {
        "千米","公里","海里","米","分米","厘米","毫米","微米","纳米","英里","英尺","英寸","码",
        "平方千米","平方公里","平方米","平方分米","平方厘米","平方毫米","平方英尺","平方英寸",
        "公顷","亩","英亩",
        "立方米","立方分米","立方厘米","立方毫米","毫升","加仑",
        "吨","公斤","千克","克","毫克","微克","斤","两","磅","盎司",
        "摄氏度","华氏度","开尔文",
        "千米每小时","公里每小时","米每秒","节","马赫",
        "周","天","小时","分钟","毫秒","微秒","纳秒",
        "赫兹","千赫兹","兆赫兹","吉赫兹",
        "安培","毫安","伏特","千伏","毫伏","瓦特","千瓦","兆瓦","欧姆","法拉","亨利",
        "安时","毫安时",
        "比特","千字节","兆字节","吉字节","太字节",
        "帕斯卡","千帕","兆帕","大气压",
        "卡路里","千卡","焦耳","千焦","流明","勒克斯","分贝","像素",
        "km²","m²","cm²","km³","m³","kHz","MHz","GHz","kPa","MPa",
        "mAh","kWh","Kbps","Mbps","Gbps",
        "km","dm","cm","mm","mL","ml",
        "kg","mg","kW","MW","kV","mV","mA","Hz",
        "kJ","kcal","lx","lm","dB","px",
        "GB","TB","MB","kB","KB",
        "Pa","°C","°F","μm","μg","nm",
        "km","m","L","t","g","A","V","W","B","J","K","F","H",
    };

    private static readonly Regex _measureRe;

    static TextNormalizer()
    {
        var sorted = new List<string>(_measureUnits);
        sorted.Sort((a, b) => b.Length.CompareTo(a.Length));
        var seen = new HashSet<string>();
        var deduped = new List<string>();
        foreach (var u in sorted) if (seen.Add(u)) deduped.Add(u);

        string unitPattern = string.Join("|", deduped.ConvertAll(Regex.Escape));
        _measureRe = new Regex(
            @"(-?\d+(?:\.\d+)?)\s*(" + unitPattern + @")",
            RegexOptions.Compiled);
    }

    private static string NormalizeMeasure(string text)
    {
        return _measureRe.Replace(text, m =>
            NumberToChinese(m.Groups[1].Value, useLiang: true) + m.Groups[2].Value);
    }

    // ──────────────────────────────────────────────
    //  9. Erhua（儿化音）
    // ──────────────────────────────────────────────

    private static readonly Regex _erhuaRe = new Regex(
        @"([\u4e00-\u9fff])儿(?=[\u4e00-\u9fff\uff0c\u3002\uff01\uff1f\u3001\uff1b\uff1a\u201c\u201d\u2018\u2019\u2026\u2014\.\!\?\,\;\:\s\r\n\t]|$)",
        RegexOptions.Compiled);

    private static string RemoveErhua(string text)
    {
        return _erhuaRe.Replace(text, m => m.Groups[1].Value);
    }

    // ──────────────────────────────────────────────
    //  10. Fullwidth → Halfwidth
    // ──────────────────────────────────────────────

    private static readonly HashSet<char> _keepFullwidth = new HashSet<char>
    {
        '\uff0c','\u3002','\uff01','\uff1f','\u3001','\uff1b','\uff1a',
        '\u201c','\u201d','\u2018','\u2019','\u2026','\u2014','\u00b7',
        '\u300a','\u300b','\u3010','\u3011','\uff08','\uff09',
    };

    private static string FullwidthToHalf(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (char c in text)
        {
            if (c == '℃') { sb.Append("°C"); continue; }
            if (c == '℉') { sb.Append("°F"); continue; }
            if (_keepFullwidth.Contains(c)) { sb.Append(c); continue; }
            if (c >= '\uFF01' && c <= '\uFF5E') { sb.Append((char)(c - 0xFEE0)); continue; }
            if (c == '\u3000') { sb.Append(' '); continue; }
            sb.Append(c);
        }
        return sb.ToString();
    }

    // ──────────────────────────────────────────────
    //  11. 标点清洗
    // ──────────────────────────────────────────────

    private static string RemovePunctuation(string text)
    {
        text = Regex.Replace(text, @"[。！？\.\!\?]", " ");
        text = Regex.Replace(text, @"[，、；：,;:]", " ");
        text = Regex.Replace(text, @"[^\u4e00-\u9fff a-zA-Z0-9\n\r]", "");
        return text;
    }
}
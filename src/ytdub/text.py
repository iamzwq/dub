from __future__ import annotations

import logging
import math
import re
import unicodedata
import warnings

NO_SPACE_LANGS = {"zh", "ja", "th", "lo", "km", "my", "yue", "bo"}

LANG_NAMES = {
    "en": "英语", "ja": "日语", "ko": "韩语", "fr": "法语", "de": "德语", "es": "西班牙语",
    "pt": "葡萄牙语", "ru": "俄语", "it": "意大利语", "ar": "阿拉伯语", "hi": "印地语",
    "vi": "越南语", "th": "泰语", "id": "印尼语", "tr": "土耳其语", "nl": "荷兰语",
    "pl": "波兰语", "uk": "乌克兰语", "zh": "中文",
}

SENT_END = ".!?。！？…؟।"
CLAUSE_PUNCT = ",;:，；：、"
CLOSERS = "\"'”’」』）)]】"
# 参与层级跟踪的成对引号/括号；直引号 " 按上下文判断开合；单引号 ' ‘ ’ 常作撇号，不跟踪
QUOTE_OPEN = "“「『（(【"
QUOTE_CLOSE = "”」』）)】"
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "u.s", "u.k",
    "inc", "ltd", "co", "no", "fig", "approx", "dept", "est", "vol",
}
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_ASCII_RUN_RE = re.compile(r"[A-Za-z0-9]+(?:[,.'’\u202f\u2009][0-9]+)*")
_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def base_lang(code: str) -> str:
    return (code or "").split("-")[0].split("_")[0].lower()


def lang_name(code: str) -> str:
    return LANG_NAMES.get(base_lang(code), code)


def is_no_space(lang: str) -> bool:
    return base_lang(lang) in NO_SPACE_LANGS


def join_tokens(parts: list[str], lang: str) -> str:
    if is_no_space(lang):
        return "".join(p.strip() for p in parts).strip()
    return re.sub(r"\s+", " ", " ".join(p.strip() for p in parts)).strip()


def is_abbreviation_dot(text: str, i: int) -> bool:
    """text[i] 是 '.' 时，判断它是否属于缩写（Mr.、e.g.）或人名首字母（J.），而非句末。"""
    words = text[:i].split()
    if not words:
        return False
    word = words[-1].lstrip(QUOTE_OPEN + "\"'‘").lower()
    return word in ABBREVIATIONS or (len(word) == 1 and word.isalpha())


_NUMBER_SEPS = ",.'’\u202f\u2009"  # 千分位/小数分隔符：1,000  1.000,5  1'000  1 000（窄空格）


def in_number(text: str, i: int) -> bool:
    """text[i] 是否为数字内部的千分位/小数分隔符（两侧都是数字）。"""
    return (
        0 < i < len(text) - 1
        and text[i] in _NUMBER_SEPS
        and text[i - 1].isdigit()
        and text[i + 1].isdigit()
    )


def count_sentence_ends(text: str) -> int:
    """统计真正的句末标点：跳过数字分隔符（3.5、1.000）、网址/缩写中间的点（example.com、U.S.），连续标点（...、?!）只计一次。"""
    count = 0
    n = len(text)
    for i, ch in enumerate(text):
        if ch not in SENT_END:
            continue
        if i + 1 < n and text[i + 1] in SENT_END:
            continue
        if ch == "." and i + 1 < n and text[i + 1].isascii() and text[i + 1].isalnum():
            continue
        count += 1
    return count


def has_punctuation(text: str, lang: str) -> bool:
    """判断字幕是否已经带有足够的句末标点。"""
    ends = count_sentence_ends(text)
    if is_no_space(lang):
        units = len(_PUNCT_RE.sub("", text))
        limit = 80
    else:
        units = len(text.split())
        limit = 40
    if units == 0:
        return True
    if units < limit:
        return ends >= 1
    return ends > 0 and units / ends <= limit


def zh_len(text: str) -> float:
    """估算中文配音的"字数"：汉字计 1，连续英文/数字按每 2.5 个字符计 1（数字中的千分位/小数分隔符不计）。"""
    cjk = len(_CJK_RE.findall(text))
    ascii_units = sum(
        math.ceil(len(re.sub(r"[^A-Za-z0-9]", "", m)) / 2.5) for m in _ASCII_RUN_RE.findall(text)
    )
    return cjk + ascii_units


def normalize_unit(text: str) -> str:
    return _PUNCT_RE.sub("", text.lower())


def split_units(text: str, lang: str) -> list[str]:
    """把文本拆成对齐用的单元：有空格语言按词，无空格语言按字符。"""
    if is_no_space(lang):
        return [c for c in normalize_unit(text)]
    return [u for u in (normalize_unit(w) for w in text.split()) if u]


def clean_caption_text(text: str) -> str:
    text = text.replace("\u200b", "").replace("\xa0", " ")
    text = re.sub(r"\[[^\]]*\]", " ", text)  # [Music] [Applause]
    text = re.sub(r"[♪♫]+", " ", text)
    text = text.replace(">>", " ")
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"[ \t\r\n]+", " ", text).strip()


def _inseparable(s: str, k: int) -> bool:
    """在 s[k-1] 与 s[k] 之间断行是否会拆开英文单词或数字（含 1,234,567 这类分隔符）。"""
    if not 0 < k < len(s):
        return False
    a, b = s[k - 1], s[k]
    if a.isascii() and a.isalnum() and b.isascii() and b.isalnum():
        return True
    return in_number(s, k - 1) or in_number(s, k)


# ---------------------------------------------------------------- 字幕分屏 / 分行
_LINE_START_FORBIDDEN = SENT_END + CLAUSE_PUNCT + CLOSERS + "》〉"  # 不能出现在行首
_LINE_START_PARTICLES = set("的了着过里们吗呢吧啊呀么地得上中下内外")  # 依附于前一个词的单字，尽量不放在行首
_LINE_END_FORBIDDEN = "“‘「『（(【《〈"  # 不能留在行尾
_ZH_CONJUNCTIONS = {
    "所以", "但是", "但", "因为", "而且", "如果", "然后", "以及", "并且", "或者", "不过", "可是", "于是",
    "因此", "同时", "而", "还是", "就是", "比如", "例如", "甚至", "否则", "虽然", "即使", "只要", "只有",
    "除非", "无论", "不管", "尤其是", "也就是说", "换句话说",
}
_jieba = None


def display_width(s: str) -> float:
    """显示宽度（以一个汉字为 1）：全角字符 1，半角字符（英文、数字、空格）0.5。"""
    return sum(1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 0.5 for ch in s)


def _word_boundaries(text: str) -> tuple[set[int], set[int]]:
    """用 jieba 分词，返回 (词边界位置, 连词之前的位置)。"""
    global _jieba
    if _jieba is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # jieba 旧代码在新版 Python 中有 SyntaxWarning
            import jieba
        jieba.setLogLevel(logging.WARNING)
        _jieba = jieba
    bounds: set[int] = set()
    conj: set[int] = set()
    pos = 0
    for word in _jieba.cut(text):
        if word in _ZH_CONJUNCTIONS:
            conj.add(pos)
        pos += len(word)
        bounds.add(pos)
    return bounds, conj


def _cut_weight(s: str, k: int, bounds: set[int], conj: set[int]) -> float | None:
    """在 s[k-1] 与 s[k] 之间断开的优先级；None 表示不允许在此断开。"""
    if not 0 < k < len(s) or _inseparable(s, k):
        return None
    a, b = s[k - 1], s[k]
    if b in _LINE_START_FORBIDDEN or a in _LINE_END_FORBIDDEN:
        return None
    if a in SENT_END or (a in CLOSERS and k >= 2 and s[k - 2] in SENT_END):
        return 3.0
    if a in CLAUSE_PUNCT:
        return 2.0
    if a == " " or b == " ":
        return 1.2
    if k in conj:
        return 1.0
    if k in bounds:
        # 以单字助词/方位词开头的行（如“里更喜欢”）读起来别扭，降低优先级
        if b in _LINE_START_PARTICLES and (k + 1 in bounds or k + 1 == len(s)):
            return 0.0
        return 0.4
    return None  # 词中间


def balanced_split(text: str, cap: float) -> list[str]:
    """
    把文本切成若干段，每段显示宽度不超过 cap，段数尽量少、各段长度尽量均衡；
    断开位置优先级：句末标点 > 逗号等 > 空格 > 连词前 > 词边界，不在词、英文单词、数字中间断开。
    """
    text = text.strip()
    width = display_width(text)
    if not text or width <= cap or len(text) < 2:
        return [text] if text else []
    n = math.ceil(width / cap)
    k1 = (n + 1) // 2
    ideal = width * k1 / n
    bounds, conj = _word_boundaries(text)
    prefix = [0.0]
    for ch in text:
        prefix.append(prefix[-1] + display_width(ch))

    best: tuple[float, int] | None = None
    for strict in (True, False):
        for k in range(1, len(text)):
            weight = _cut_weight(text, k, bounds, conj)
            if weight is None:
                continue
            left = prefix[k]
            if strict and (left > k1 * cap or width - left > (n - k1) * cap):
                continue
            score = weight - 3 * abs(left - ideal) / cap
            if best is None or score > best[0]:
                best = (score, k)
        if best:
            break
    if best is None:  # 没有合法断点（如一个超长英文单词）：在最接近理想位置处硬切
        k = min(range(1, len(text)), key=lambda i: (_inseparable(text, i), abs(prefix[i] - ideal)))
    else:
        k = best[1]
    return balanced_split(text[:k], cap) + balanced_split(text[k:], cap)


def split_screens(text: str, max_chars: int, max_lines: int = 2) -> list[list[str]]:
    """把一句译文切成若干屏字幕，每屏最多 max_lines 行、每行最多 max_chars 个汉字宽。"""
    screens = balanced_split(text, max_chars * max(max_lines, 1))
    return [lines for lines in (balanced_split(s, max_chars) for s in screens) if lines]


def split_display_lines(text: str, max_chars: int) -> list[str]:
    """把文本切成若干不超过 max_chars 的字幕行（均衡切分）。"""
    return balanced_split(text, max_chars)


def strip_display_punct(line: str) -> str:
    # line = re.sub(r"[，、；：]", " ", line)
    line = re.sub(r"[。，、；：,.;: ]+([”’」』）)]*)$", r"\1", line)  # 行尾标点（含闭引号前的）
    return re.sub(r" {2,}", " ", line).strip()

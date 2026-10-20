from __future__ import annotations

import math
import re

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


def split_display_lines(text: str, max_chars: int) -> list[str]:
    """把一句中文译文切成若干不超过 max_chars 的字幕行，优先在标点处断开，不拆开英文单词和数字。"""
    text = text.strip()
    if zh_len(text) <= max_chars or len(text) <= max_chars:
        return [text] if text else []
    lines: list[str] = []
    rest = text
    while rest:
        if len(rest) <= max_chars:
            lines.append(rest)
            break
        window = rest[: max_chars + 1]
        cut = -1
        for i in range(len(window) - 1, max(len(window) // 3, 1) - 1, -1):
            if window[i] in SENT_END + CLAUSE_PUNCT + " " and not in_number(rest, i):
                cut = i + 1
                break
        if cut <= 0:
            cut = max_chars
            while _inseparable(rest, cut):
                cut -= 1
            if cut == 0:
                # 整行都是一个超长单词/数字：向后找到它的结尾
                cut = max_chars
                while _inseparable(rest, cut):
                    cut += 1
        lines.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    # 避免最后一行过短
    if len(lines) >= 2 and len(lines[-1]) <= 2:
        tail = lines.pop()
        lines[-1] = lines[-1] + tail
    return [ln for ln in lines if ln]


def strip_display_punct(line: str) -> str:
    line = re.sub(r"[，、；：]", " ", line)
    line = line.rstrip("。，、；：,.;: ")
    return re.sub(r" {2,}", " ", line).strip()

"""字幕解析（json3 / vtt）与按句末标点断句。"""
from __future__ import annotations

import re

from .models import Sentence, Token
from .text import (
    CLAUSE_PUNCT,
    CLOSERS,
    QUOTE_CLOSE,
    QUOTE_OPEN,
    SENT_END,
    ZH_CONJUNCTIONS,
    clean_caption_text,
    inseparable,
    is_abbreviation_dot,
    is_no_space,
    join_tokens,
)

_CONJUNCTIONS = {
    "and", "but", "so", "because", "which", "that", "or", "then", "when", "where", "while", "if",
    "although", "though", "since", "until", "after", "before", "unless", "whereas",
}


# ---------------------------------------------------------------- 解析
def parse_json3(data: dict) -> list[Token]:
    """解析 YouTube json3 字幕。自动字幕含逐词时间戳，手动字幕每个事件为一个 cue。"""
    tokens: list[Token] = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue
        ev_start = ev.get("tStartMs", 0) / 1000
        ev_end = ev_start + ev.get("dDurationMs", 0) / 1000
        word_level = any("tOffsetMs" in s for s in segs) or len(segs) > 1 and all(
            "acAsrConf" in s for s in segs
        )
        if word_level:
            items = [
                (s.get("utf8", ""), ev_start + s.get("tOffsetMs", 0) / 1000)
                for s in segs
                if s.get("utf8", "").strip()
            ]
            for i, (text, start) in enumerate(items):
                end = items[i + 1][1] if i + 1 < len(items) else ev_end
                tokens.append(Token(text, start, max(end, start)))
        else:
            text = "".join(s.get("utf8", "") for s in segs)
            if text.strip():
                tokens.append(Token(text, ev_start, ev_end))
    return _finalize(tokens)


_VTT_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})")


def _vtt_seconds(s: str) -> float:
    m = _VTT_TIME.match(s.strip())
    if not m:
        raise ValueError(f"无法解析时间: {s}")
    h, mi, se, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(se) + int(ms) / 1000


def parse_vtt(content: str) -> list[Token]:
    tokens: list[Token] = []
    blocks = re.split(r"\r?\n\r?\n", content.replace("\ufeff", ""))
    last_text = ""
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        a, b = lines[idx].split("-->")
        start, end = _vtt_seconds(a), _vtt_seconds(b.split()[0])
        text = " ".join(lines[idx + 1:])
        text = re.sub(r"<\d{2}:\d{2}[^>]*>", "", text)
        if text.strip() and text.strip() != last_text:
            tokens.append(Token(text, start, end))
            last_text = text.strip()
    return _finalize(tokens)


def _finalize(tokens: list[Token]) -> list[Token]:
    out: list[Token] = []
    for t in tokens:
        text = clean_caption_text(t.text)
        if text:
            out.append(Token(text, t.start, t.end))
    out.sort(key=lambda t: t.start)
    for i in range(len(out) - 1):
        nxt = out[i + 1].start
        if out[i].end > nxt:
            out[i].end = max(nxt, out[i].start + 0.01)
    return out


# ---------------------------------------------------------------- 按句末标点断句
QUOTE_MAX_SECONDS = 15.0  # 引号内的内容超过该时长仍未闭合时，允许在引号内断句（防止引号不配对）


def _straight_quote_kind(text: str, i: int) -> str | None:
    prev = text[i - 1] if i > 0 else " "
    nxt = text[i + 1] if i + 1 < len(text) else " "
    if (prev.isspace() or prev in QUOTE_OPEN or prev in "—–-") and not nxt.isspace():
        return "open"
    if not prev.isspace():
        return "close"
    return None


def _flatten(
    tokens: list[Token], lang: str, max_gap: float
) -> tuple[str, list[tuple[float, float]], set[int], set[int]]:
    """把 token 拼成一段文本，记录每个字符的时间（token 内按字符数线性分配）、长静音处的强制断点和各 token 的起始位置。"""
    sep = "" if is_no_space(lang) else " "
    chars: list[str] = []
    times: list[tuple[float, float]] = []
    hard: set[int] = set()
    starts: set[int] = set()
    for k, tok in enumerate(tokens):
        if k > 0:
            prev = tokens[k - 1]
            if max_gap > 0 and tok.start - prev.end >= max_gap:
                hard.add(len(chars))
            if sep:
                chars.append(sep)
                times.append((prev.end, prev.end))
        starts.add(len(chars))
        step = (tok.end - tok.start) / max(len(tok.text), 1)
        for j, ch in enumerate(tok.text):
            chars.append(ch)
            times.append((tok.start + step * j, tok.start + step * (j + 1)))
    return "".join(chars), times, hard, starts


def find_sentence_breaks(
    text: str, lang: str, times: list[tuple[float, float]] | None = None, hard: set[int] | None = None
) -> list[int]:
    """
    返回断句位置（每句的结束下标，不含）。规则：
    - 只在句末标点处断句；缩写（Mr.）、小数（3.5）、网址等不断；
    - 句末标点后紧跟的引号/括号归入本句：He said "yes." | Then…；
    - 引号内的句末标点不断句，引号内的多句话保持在一起，除非引号超过 QUOTE_MAX_SECONDS 仍未闭合；
    - 引号闭合后的内容若是句子的延续则不断：
      有大小写的语言看下一个词是否小写（"Why?" he asked.）；
      无空格语言（中/日）除非紧跟新的引号，否则视为延续（「行くよ。」と言った。）。
    """
    cased = not is_no_space(lang)
    hard = hard or set()
    n = len(text)
    breaks: list[int] = []
    depth = 0
    seg_start = 0
    i = 0
    while i < n:
        if i in hard and i > seg_start:
            breaks.append(i)
            seg_start = i
            depth = 0
        ch = text[i]
        if ch in QUOTE_OPEN:
            depth += 1
        elif ch in QUOTE_CLOSE:
            depth = max(depth - 1, 0)
        elif ch == '"':
            kind = _straight_quote_kind(text, i)
            depth = depth + 1 if kind == "open" else max(depth - 1, 0) if kind == "close" else depth

        if ch not in SENT_END or (ch == "." and is_abbreviation_dot(text, i)):
            i += 1
            continue

        j = i
        while j + 1 < n and text[j + 1] in SENT_END:
            j += 1
        inside = depth > 0
        m = j
        while m + 1 < n and text[m + 1] in CLOSERS:
            m += 1
            if text[m] in QUOTE_CLOSE or text[m] == '"':
                depth = max(depth - 1, 0)
        k = m + 1
        while k < n and text[k].isspace():
            k += 1
        follow = text[k] if k < n else ""

        if follow and ch == "." and k == m + 1 and (cased or (follow.isascii() and follow.isalnum())):
            split = False  # 3.5 / U.S / example.com
        elif cased and follow and k == m + 1 and not follow.isspace():
            split = False
        elif depth > 0:
            too_long = times is not None and times[j][1] - times[seg_start][0] > QUOTE_MAX_SECONDS
            split = too_long
            if too_long:
                depth = 0
        elif inside:  # 引号刚闭合
            if not follow or follow in QUOTE_OPEN or follow == '"':
                split = True
            elif cased:
                split = not follow.islower()
            else:
                split = False
        else:
            split = True

        if split:
            breaks.append(m + 1)
            seg_start = m + 1
        i = m + 1
    if not breaks or breaks[-1] < n:
        breaks.append(n)
    return breaks


_WORD_RE = re.compile(r"\S+")
_ZH_CONJ_MULTI = tuple(c for c in ZH_CONJUNCTIONS if len(c) > 1)  # 单字连词（而、但）易误中「而言」等词
MIN_PIECE_RATIO = 0.3  # 拆分后每段至少占 max_seconds 的比例，避免拆出过短的碎片


def _trim(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def _split_by_time(
    text: str, times: list[tuple[float, float]], starts: set[int], lang: str, a: int, b: int, max_seconds: float
) -> list[tuple[int, int]]:
    """
    句子超过 max_seconds 时，在停顿、从句边界（逗号/分号、连词前）处递归二分；
    优先选 token 边界（时间戳为原始值，token 内部是按字符数估算的），且每段不短于 MIN_PIECE_RATIO * max_seconds。
    """
    a, b = _trim(text, a, b)
    if b <= a:
        return []
    t0, t1 = times[a][0], times[b - 1][1]
    dur = t1 - t0
    if max_seconds <= 0 or dur <= max_seconds:
        return [(a, b)]
    spaced = not is_no_space(lang)
    mid = t0 + dur / 2
    min_piece = max_seconds * MIN_PIECE_RATIO
    best: tuple[float, int] | None = None
    for k in range(a + 1, b):
        if text[k].isspace() or inseparable(text, k):
            continue
        p = k - 1
        while p > a and text[p].isspace():
            p -= 1
        prev = text[p]
        if times[p][1] - t0 < min_piece or t1 - times[k][0] < min_piece:
            continue
        clause = prev in CLAUSE_PUNCT or prev in SENT_END
        at_token = k in starts
        if spaced:
            if not text[k - 1].isspace():
                continue
        elif not (clause or at_token):
            continue
        if prev in QUOTE_OPEN:
            continue
        score = 3.0 if clause else 0.0
        if at_token:
            score += 1.5
        if spaced:
            word = _WORD_RE.match(text, k).group().lower().strip(",.;:!?\"'")
            if word in _CONJUNCTIONS:
                score += 2
        elif text.startswith(_ZH_CONJ_MULTI, k):
            score += 2
        score += min(max(times[k][0] - times[p][1], 0.0), 1.5) * 4
        score -= abs(times[k][0] - mid) / dur * 6
        if best is None or score > best[0]:
            best = (score, k)
    if best is None:
        return [(a, b)]
    k = best[1]
    return _split_by_time(text, times, starts, lang, a, k, max_seconds) + _split_by_time(
        text, times, starts, lang, k, b, max_seconds
    )


def segment_punct(
    tokens: list[Token], lang: str, max_gap: float = 0.0, max_seconds: float = 0.0
) -> list[Sentence]:
    """
    有标点的字幕：标点优先，按句末标点断句（含引号内的完整引语）；
    单句超过 max_seconds（>0）时再按停顿和从句边界拆分，每段沿用原始字符时间戳。
    """
    text, times, hard, starts = _flatten(tokens, lang, max_gap)
    sentences: list[Sentence] = []
    prev = 0
    for b in find_sentence_breaks(text, lang, times, hard):
        for s, e in _split_by_time(text, times, starts, lang, prev, b, max_seconds):
            sentences.append(Sentence(len(sentences) + 1, round(times[s][0], 3), round(times[e - 1][1], 3), text[s:e]))
        prev = b
    return sentences


# ---------------------------------------------------------------- AI 断句辅助：按时长切分过长的句子
def _best_split(group: list[Token]) -> int:
    start, end = group[0].start, group[-1].end
    dur = max(end - start, 1e-3)
    mid = start + dur / 2
    best_j, best_score = len(group) // 2, float("-inf")
    for j in range(1, len(group)):
        prev, cur = group[j - 1], group[j]
        score = 0.0
        if prev.text.rstrip()[-1:] in CLAUSE_PUNCT:
            score += 3
        if prev.text.rstrip()[-1:] in SENT_END:
            score += 4
        if cur.text.strip().lower().strip(",.") in _CONJUNCTIONS:
            score += 2
        score += min(cur.start - prev.end, 1.5) * 4
        score -= abs(cur.start - mid) / dur * 6
        if score > best_score:
            best_j, best_score = j, score
    return best_j


def split_long(group: list[Token], max_seconds: float) -> list[list[Token]]:
    if len(group) <= 1 or group[-1].end - group[0].start <= max_seconds:
        return [group]
    j = _best_split(group)
    return split_long(group[:j], max_seconds) + split_long(group[j:], max_seconds)


def groups_to_sentences(groups: list[list[Token]], lang: str, texts: list[str] | None = None) -> list[Sentence]:
    sentences: list[Sentence] = []
    for i, g in enumerate(groups):
        if not g:
            continue
        text = texts[i] if texts else join_tokens([t.text for t in g], lang)
        text = text.strip()
        if not text:
            continue
        sentences.append(Sentence(id=len(sentences) + 1, start=round(g[0].start, 3), end=round(g[-1].end, 3), text=text))
    return sentences

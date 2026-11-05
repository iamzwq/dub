"""剪除片段（SponsorBlock 广告/赞助等）后的时间映射：下游步骤统一使用剪除后的时间轴。"""
from __future__ import annotations

from .models import Token


def merge_ranges(ranges: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted((max(0.0, a), min(duration, b)) for a, b in ranges):
        if b - a < 0.1:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def keep_ranges(cuts: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    keep, t = [], 0.0
    for a, b in cuts:
        if a > t:
            keep.append((t, a))
        t = max(t, b)
    if duration > t:
        keep.append((t, duration))
    return keep


def map_time(t: float, cuts: list[tuple[float, float]]) -> float:
    """原始时间 → 剪除后的时间；落在剪除区间内的时间映射到该区间起点。"""
    removed = 0.0
    for a, b in cuts:
        if t <= a:
            break
        if t < b:
            return a - removed
        removed += b - a
    return t - removed


def remap_tokens(tokens: list[Token], cuts: list[tuple[float, float]]) -> list[Token]:
    """丢弃中点落在剪除区间内的字幕片段，其余片段平移到剪除后的时间轴。"""
    if not cuts:
        return tokens
    out = []
    for t in tokens:
        mid = (t.start + t.end) / 2
        if any(a <= mid < b for a, b in cuts):
            continue
        out.append(Token(t.text, map_time(t.start, cuts), map_time(t.end, cuts)))
    return out

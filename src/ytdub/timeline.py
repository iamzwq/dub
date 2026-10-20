"""配音时间线编排：atempo 微调 + 有限顺延（纯函数，便于测试）。"""
from __future__ import annotations

from dataclasses import dataclass

MIN_AVAIL = 0.05


@dataclass
class Clip:
    id: int
    start: float  # 原字幕开始时间
    next_start: float  # 下一句开始时间（本句理想的最晚结束时间）
    duration: float  # 配音时长
    max_drift: float | None = None  # 覆盖全局顺延上限（最后一句为 0）


@dataclass
class Placement:
    id: int
    play_start: float
    tempo: float
    play_duration: float
    drift: float  # 相对原字幕开始时间的延后量

    @property
    def play_end(self) -> float:
        return self.play_start + self.play_duration


def schedule(
    clips: list[Clip],
    atempo_max: float = 1.15,
    max_drift: float = 1.5,
    drift_atempo_max: float = 1.25,
    tolerance: float = 0.1,
) -> list[Placement]:
    """
    依次放置每句配音：
    1. 上一句没说完则本句顺延（start = max(原始开始, 游标)）；
    2. 超出本句时段时，先用 atempo 压缩（≤ atempo_max）；
    3. 仍超出则允许延后结束（下一句跟着顺延），但累计延后不超过 max_drift，
       超过部分继续压缩（≤ drift_atempo_max）；
    4. 超出量 ≤ tolerance 时不压缩，直接顺延。
    """
    out: list[Placement] = []
    cursor = 0.0
    for c in clips:
        drift_cap = max_drift if c.max_drift is None else c.max_drift
        start = max(c.start, cursor)
        d = c.duration
        tempo = 1.0
        overflow = start + d - c.next_start
        if d > 0 and overflow > tolerance:
            avail = max(c.next_start - start, MIN_AVAIL)
            tempo = min(d / avail, atempo_max)
            if start + d / tempo > c.next_start + drift_cap:
                avail2 = max(c.next_start + drift_cap - start, MIN_AVAIL)
                tempo = max(tempo, min(d / avail2, drift_atempo_max))
        tempo = max(tempo, 1.0)
        dur = d / tempo
        out.append(Placement(c.id, round(start, 3), round(tempo, 4), round(dur, 3), round(start - c.start, 3)))
        cursor = start + dur
    return out

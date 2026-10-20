"""⑥ 拼接配音轨（dub.wav）并生成中文硬字幕（zh.ass）。"""
from __future__ import annotations

import numpy as np

from ..context import Context, StepError
from ..log import count_progress, log
from ..media import probe, read_audio, write_wav
from ..text import display_width, split_screens, strip_display_punct


def fingerprint(cfg) -> dict:
    return {"subtitle": cfg.subtitle.model_dump()}


def _ass_time(t: float) -> str:
    t = max(t, 0.0)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    """转义 ASS 特殊字符；文本中的换行转为 ASS 换行符 \\N。"""
    lines = text.split("\n")
    return r"\N".join(ln.replace("\\", "＼").replace("{", "｛").replace("}", "｝") for ln in lines)


def subtitle_events(
    timeline: list[dict], max_chars: int, strip_punct: bool, max_lines: int = 2
) -> list[tuple[float, float, str]]:
    """每句译文切成若干屏（每屏最多 max_lines 行，行间用 \\n 分隔），按显示宽度分配该句的显示时间。"""
    events: list[tuple[float, float, str]] = []
    for i, e in enumerate(timeline):
        text = e["zh"].strip()
        if not text:
            continue
        if e["duration"] > 0:
            start, end = e["play_start"], e["play_end"]
        else:
            start, end = e["start"], e["end"]
        nxt = timeline[i + 1]["play_start"] if i + 1 < len(timeline) else end + 0.5
        end = max(end, min(nxt, end + 0.5))
        screens = split_screens(text, max_chars, max_lines)
        if strip_punct:
            screens = [s for s in ([ln for ln in (strip_display_punct(x) for x in sc) if ln] for sc in screens) if s]
        if not screens:
            continue
        widths = [max(display_width("".join(sc)), 0.5) for sc in screens]
        total = sum(widths)
        t = start
        for sc, w in zip(screens, widths):
            d = (end - start) * w / total
            events.append((t, t + d, "\n".join(sc)))
            t += d
    return events


def ass_color(rgb: str, opacity: float) -> str:
    """'RRGGBB' + 不透明度 → ASS 颜色 &HAABBGGRR（ASS 的 alpha：00 不透明，FF 全透明）。"""
    rgb = rgb.lstrip("#")
    if len(rgb) != 6:
        raise ValueError(f"颜色格式应为 RRGGBB：{rgb}")
    alpha = round(255 * (1 - min(max(opacity, 0.0), 1.0)))
    return f"&H{alpha:02X}{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}".upper()


def build_ass(events: list[tuple[float, float, str]], width: int, height: int, cfg) -> str:
    font_size = max(16, round(min(width, height) * cfg.font_scale))
    margin_v = round(height * cfg.margin_scale)
    # BorderStyle=4（libass 扩展）：用 BackColour 画背景框，同时保留文字描边；=1 时只有描边
    if cfg.box_opacity > 0:
        border_style, back = 4, ass_color(cfg.box_color, cfg.box_opacity)
    else:
        border_style, back = 1, "&H64000000"
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{cfg.font},{font_size},&H00FFFFFF,&H000000FF,&H00000000,{back},0,0,0,0,100,100,0,0,{border_style},{cfg.outline},0,2,20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [
        f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Default,,0,0,0,,{_ass_escape(t)}" for s, e, t in events
    ]
    return header + "\n".join(lines) + "\n"


def max_chars_for(width: int, height: int, cfg) -> int:
    font_size = max(16, round(min(width, height) * cfg.font_scale))
    return max(8, min(cfg.max_chars_per_line, int(width * 0.9 / font_size)))


def run(ctx: Context) -> None:
    ws, cfg = ctx.ws, ctx.cfg
    timeline = ws.read_json("timeline.json")
    if not timeline:
        raise StepError("缺少 timeline.json，请先执行对齐步骤")
    meta = ws.read_json("meta.json") or {}
    video = ws.find_video()
    if video is None:
        raise StepError("未找到下载的视频文件")
    v = meta.get("video") or probe(video)
    width, height, duration = v.get("width", 1920), v.get("height", 1080), v["duration"]

    # 字幕
    max_chars = max_chars_for(width, height, cfg.subtitle)
    events = subtitle_events(timeline, max_chars, cfg.subtitle.strip_punctuation, cfg.subtitle.max_lines)
    ws.path("zh.ass").write_text(build_ass(events, width, height, cfg.subtitle), encoding="utf-8-sig")
    log.info("字幕：%d 屏（每屏最多 %d 行、每行最多 %d 字）→ zh.ass", len(events), cfg.subtitle.max_lines, max_chars)

    # 配音轨
    sr = None
    buf: np.ndarray | None = None
    with count_progress() as progress:
        task = progress.add_task("拼接配音", total=len(timeline), info="")
        for e in timeline:
            data, file_sr = read_audio(ws.tts_dir / e["file"])
            if sr is None:
                sr = file_sr
                total_len = max(duration, max(x["play_end"] for x in timeline)) + 0.5
                buf = np.zeros(int(total_len * sr), dtype=np.float32)
            elif file_sr != sr:
                raise StepError(f"采样率不一致：{e['file']}")
            if data.size:
                i = int(round(e["play_start"] * sr))
                j = min(i + data.size, buf.size)
                buf[i:j] += data[: j - i]
            progress.update(task, advance=1)
    assert buf is not None and sr is not None
    buf = buf[: int(duration * sr)]
    peak = float(np.max(np.abs(buf))) if buf.size else 0.0
    if peak > 1.0:
        buf /= peak
    write_wav(ws.path("dub.wav"), buf, sr)
    log.info("配音轨：%.1fs，%d Hz → dub.wav", buf.size / sr, sr)

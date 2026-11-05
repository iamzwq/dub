"""⑧ ffmpeg 合成：烧录中文硬字幕 + 替换为中文配音 + 嵌入中文封面。"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..context import Context, StepError
from ..cuts import keep_ranges
from ..log import log, percent_progress
from ..media import pick_encoder, probe

OUTPUT = "output.mp4"
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_filename(title: str, max_len: int = 80) -> str:
    """把标题转成跨平台安全的文件名（不含扩展名）；无法使用时返回空串。"""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", title)
    name = re.sub(r"\s+", " ", name).strip()[:max_len]
    # 去掉开头的 "-"（避免被 ffmpeg 当成参数）和结尾的点/空格（Windows 不允许）
    name = name.lstrip("-. ").rstrip(". ")
    if not name or name.upper() in _RESERVED_NAMES:
        return ""
    return name


def output_name(ws) -> str:
    titles = ws.read_json("titles.json") or {}
    name = safe_filename(str(titles.get("zh_title") or ""))
    # 不能与下载的原视频 video.* 同名
    return f"{name}.mp4" if name and name.lower() != "video" else OUTPUT


def fingerprint(cfg) -> dict:
    return {"compose": cfg.compose.model_dump()}


def _video_codec_args(encoder: str, cfg) -> list[str]:
    c = cfg.compose
    if encoder == "h264_nvenc":
        return ["-c:v:0", "h264_nvenc", "-preset", c.nvenc_preset, "-rc", "vbr", "-cq", str(c.nvenc_cq), "-b:v:0", "0"]
    return ["-c:v:0", "libx264", "-preset", c.x264_preset, "-crf", str(c.x264_crf)]


def build_command(
    video: str, encoder: str, cfg, cover: str | None, output: str = OUTPUT,
    keep: list[tuple[float, float]] | None = None,
) -> list[str]:
    c = cfg.compose
    args = ["ffmpeg", "-hide_banner", "-y", "-nostats", "-loglevel", "error", "-progress", "pipe:1",
            "-i", video, "-i", "dub.wav"]
    if cover:
        args += ["-i", cover]
    if keep:
        parts = [f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[k{i}]" for i, (a, b) in enumerate(keep)]
        labels = "".join(f"[k{i}]" for i in range(len(keep)))
        src = ";".join(parts) + f";{labels}concat=n={len(keep)}:v=1:a=0[cut];[cut]"
    else:
        src = "[0:v]"
    graph = f"{src}subtitles=zh.ass,format=yuv420p[v];[1:a]loudnorm={c.loudnorm},aresample=48000[a]"
    args += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]"]
    if cover:
        args += ["-map", "2:v"]
    args += _video_codec_args(encoder, cfg)
    if cover:
        args += ["-c:v:1", "mjpeg", "-disposition:v:1", "attached_pic"]
    args += ["-c:a", "aac", "-b:a", c.audio_bitrate, "-movflags", "+faststart", output]
    return args


def _run_ffmpeg(ctx: Context, args: list[str], duration: float, label: str) -> None:
    log.debug("执行：%s", " ".join(args))
    err_path = ctx.ws.path("compose.log")
    with open(err_path, "w", encoding="utf-8") as err, percent_progress() as progress:
        task = progress.add_task(label, total=duration, info="")
        proc = subprocess.Popen(
            args, cwd=ctx.ws.dir, stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8", errors="replace"
        )
        speed = ""
        assert proc.stdout is not None
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key in ("out_time_us", "out_time_ms") and value.isdigit():
                progress.update(task, completed=min(int(value) / 1_000_000, duration))
            elif key == "speed":
                speed = value
                progress.update(task, info=f"速度 {speed}")
            elif key == "progress" and value == "end":
                progress.update(task, completed=duration)
        code = proc.wait()
    if code != 0:
        tail = err_path.read_text(encoding="utf-8", errors="replace")[-1200:]
        raise RuntimeError(tail.strip() or f"ffmpeg 退出码 {code}")


def run(ctx: Context) -> None:
    ws, cfg = ctx.ws, ctx.cfg
    video = ws.require_video()
    for name in ("dub.wav", "zh.ass"):
        if not ws.path(name).exists():
            raise StepError(f"缺少 {name}，请先执行拼接步骤")
    duration = probe(video)["duration"]
    cuts = [tuple(c) for c in ws.meta().get("cuts") or []]
    keep = keep_ranges(cuts, duration) if cuts else None
    if keep:
        log.info("剪除 %d 个广告/赞助片段（共 %.1fs）", len(cuts), sum(b - a for a, b in cuts))
        duration = sum(b - a for a, b in keep)
    cover = "cover.zh.jpg" if ws.path("cover.zh.jpg").exists() else ("cover.jpg" if ws.path("cover.jpg").exists() else None)

    output = output_name(ws)
    encoder = pick_encoder(cfg.compose.encoder)
    log.info("编码器：%s；封面：%s；输出文件：%s", encoder, cover or "无", output)
    try:
        _run_ffmpeg(ctx, build_command(video.name, encoder, cfg, cover, output, keep), duration, f"合成视频（{encoder}）")
    except RuntimeError as e:
        if encoder == "libx264":
            raise StepError(f"ffmpeg 合成失败：{e}") from e
        log.warning("%s 编码失败，改用 libx264 重试：%s", encoder, str(e).splitlines()[-1] if str(e) else "")
        try:
            _run_ffmpeg(ctx, build_command(video.name, "libx264", cfg, cover, output, keep), duration, "合成视频（libx264）")
        except RuntimeError as e2:
            raise StepError(f"ffmpeg 合成失败：{e2}") from e2
    out = ws.path(output)
    # 标题变化后删除上次生成的旧成品，避免目录里残留多个视频
    prev = (ws.read_json("report.json", {}).get("compose") or {}).get("output")
    if prev and Path(prev).name != output and Path(prev).parent.resolve() == ws.dir.resolve():
        Path(prev).unlink(missing_ok=True)
    info = probe(out)
    log.info("输出：%s（%.1fs，%.1f MB）", out, info["duration"], out.stat().st_size / 1024 / 1024)
    ws.update_report("compose", {"encoder": encoder, "output": str(out)})

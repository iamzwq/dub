"""ffmpeg / ffprobe / 音频读写等工具函数。"""
from __future__ import annotations

import functools
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .log import log


def require_tools() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise RuntimeError(f"未找到 {', '.join(missing)}，请先安装 ffmpeg 并加入 PATH")


def probe(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path),
        ],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    data = json.loads(out)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    duration = float(data.get("format", {}).get("duration") or 0)
    info = {"duration": duration}
    if video:
        info["width"] = int(video.get("width", 0))
        info["height"] = int(video.get("height", 0))
        rot = 0
        for sd in video.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(sd["rotation"])
        if abs(rot) in (90, 270):
            info["width"], info["height"] = info["height"], info["width"]
    return info


def run_ffmpeg(args: list[str], cwd: Path | None = None) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    log.debug("ffmpeg %s", " ".join(args))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败：{res.stderr.strip()[-800:]}")


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """读取音频为单声道 float32。soundfile 失败时回退到 ffmpeg 解码。"""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001
        tmp = path.with_suffix(".decoded.wav")
        run_ffmpeg(["-i", str(path), "-ac", "1", str(tmp)])
        data, sr = sf.read(str(tmp), dtype="float32", always_2d=False)
        tmp.unlink(missing_ok=True)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    sf.write(str(path), data, sr, subtype="PCM_16")


def trim_silence(data: np.ndarray, sr: int, threshold_db: float = -45.0, pad: float = 0.03) -> np.ndarray:
    if data.size == 0:
        return data
    threshold = 10 ** (threshold_db / 20)
    frame = max(int(sr * 0.01), 1)
    n = data.size // frame
    if n == 0:
        return data
    rms = np.sqrt(np.mean(data[: n * frame].reshape(n, frame) ** 2, axis=1))
    loud = np.nonzero(rms > threshold)[0]
    if loud.size == 0:
        return data[:0]
    p = int(sr * pad)
    start = max(loud[0] * frame - p, 0)
    end = min((loud[-1] + 1) * frame + p, data.size)
    return data[start:end]


def atempo_filter(tempo: float) -> str:
    filters = []
    t = tempo
    while t > 2.0:
        filters.append("atempo=2.0")
        t /= 2.0
    while t < 0.5:
        filters.append("atempo=0.5")
        t /= 0.5
    filters.append(f"atempo={t:.4f}")
    return ",".join(filters)


def atempo(src: Path, dst: Path, tempo: float) -> None:
    """不变调地加速音频（tempo>1 变快）。"""
    run_ffmpeg(["-i", str(src), "-filter:a", atempo_filter(tempo), str(dst)])


@functools.lru_cache(maxsize=4)
def pick_encoder(preference: str) -> str:
    if preference == "libx264":
        return "libx264"
    if _encoder_works("h264_nvenc"):
        return "h264_nvenc"
    if preference == "h264_nvenc":
        log.warning("h264_nvenc 不可用，回退到 libx264")
    return "libx264"


def _encoder_works(name: str) -> bool:
    res = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
            "-c:v", name, "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    return res.returncode == 0

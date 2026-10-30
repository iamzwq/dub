"""配音合成工具（edge-tts / MOSS-TTS-Nano，带重试、缓存、静音裁剪）。"""
from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import re
from pathlib import Path

import edge_tts
import httpx
import numpy as np
import soundfile as sf

from .config import TTSConfig
from .log import log
from .media import atempo_filter, read_audio, run_ffmpeg, trim_silence, write_wav
from .workspace import short_hash

SR = 24000


def rate_str(rate: int) -> str:
    return f"{rate:+d}%"


def tts_filename(sid: int, text: str, voice: str, rate: int) -> str:
    return f"{sid:04d}_{short_hash(text, voice, rate)}.wav"


@functools.lru_cache(maxsize=8)
def _file_hash(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"参考音频不存在: {path}")
    return hashlib.sha1(p.read_bytes()).hexdigest()[:10]


def voice_key(t: TTSConfig) -> str:
    """音色标识，用于缓存文件名与步骤指纹。"""
    if t.engine == "edge":
        return t.voice
    m = t.moss
    ref = f"file:{_file_hash(m.prompt_audio)}" if m.prompt_audio else f"demo:{m.demo_id}"
    return f"moss_nano|{ref}|seed={m.seed}"


def concurrency(t: TTSConfig) -> int:
    return t.moss.concurrency if t.engine == "moss_nano" else t.concurrency


def speakable(text: str) -> bool:
    return bool(re.search(r"\w", text))


def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


async def synthesize(t: TTSConfig, text: str, rate: int, out: Path) -> float:
    """合成一句并保存为裁剪掉首尾静音的 wav，返回时长（秒）。已存在则直接复用。"""
    if out.exists():
        return wav_duration(out)
    if not speakable(text):
        write_wav(out, np.zeros(0, dtype=np.float32), SR)
        return 0.0
    if t.engine == "moss_nano":
        return await _synthesize_moss_nano(t, text, rate, out)
    return await _synthesize_edge(text, t.voice, rate, out, t.retries, t.trim_silence_db, t.proxy)


async def _synthesize_moss_nano(t: TTSConfig, text: str, rate: int, out: Path) -> float:
    m = t.moss
    form = {"text": text, "seed": str(m.seed), "max_new_frames": str(m.max_new_frames)}
    files = None
    if m.prompt_audio:
        p = Path(m.prompt_audio)
        files = {"prompt_audio": (p.name, p.read_bytes(), "application/octet-stream")}
    else:
        form["demo_id"] = m.demo_id
    url = m.url.rstrip("/") + "/api/generate"
    last: Exception | None = None
    for attempt in range(1, t.retries + 1):
        try:
            # 不走系统代理：服务通常在本机或局域网
            async with httpx.AsyncClient(timeout=m.timeout, trust_env=False) as client:
                r = await client.post(url, data=form, files=files)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            wav = base64.b64decode(r.json()["audio_base64"])
            break
        except Exception as e:  # noqa: BLE001
            last = e
            log.debug("MOSS-TTS-Nano 失败（第 %d 次）：%s", attempt, e)
        await asyncio.sleep(min(2 ** attempt, 15))
    else:
        raise RuntimeError(f"MOSS-TTS-Nano 合成失败（{url}）：{last}")

    def _post() -> float:
        raw = out.with_suffix(".raw.wav")
        conv = out.with_suffix(".conv.wav")
        raw.write_bytes(wav)
        # 48kHz 立体声 → 与 edge-tts 一致的 24kHz 单声道；模型无语速参数，用 atempo 实现 rate
        af = ["-filter:a", atempo_filter(1 + rate / 100)] if rate else []
        run_ffmpeg(["-i", str(raw), *af, "-ac", "1", "-ar", str(SR), str(conv)])
        data, sr = read_audio(conv)
        data = trim_silence(data, sr, t.trim_silence_db)
        write_wav(out, data, sr)
        raw.unlink(missing_ok=True)
        conv.unlink(missing_ok=True)
        return data.size / sr

    return await asyncio.to_thread(_post)


async def _synthesize_edge(
    text: str,
    voice: str,
    rate: int,
    out: Path,
    retries: int,
    trim_db: float,
    proxy: str | None,
) -> float:
    tmp = out.with_suffix(".mp3")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            await edge_tts.Communicate(text, voice, rate=rate_str(rate), proxy=proxy or None).save(str(tmp))
            if tmp.exists() and tmp.stat().st_size > 0:
                break
            raise RuntimeError("未收到音频")
        except edge_tts.exceptions.NoAudioReceived as e:
            last = e
            if attempt == retries:
                log.warning("edge-tts 无音频输出，按静音处理：%s", text)
                write_wav(out, np.zeros(0, dtype=np.float32), SR)
                return 0.0
        except Exception as e:  # noqa: BLE001
            last = e
            log.debug("edge-tts 失败（第 %d 次）：%s", attempt, e)
        await asyncio.sleep(min(2 ** attempt, 15))
    else:
        raise RuntimeError(f"edge-tts 合成失败：{last}")

    def _post() -> float:
        data, sr = read_audio(tmp)
        data = trim_silence(data, sr, trim_db)
        write_wav(out, data, sr)
        tmp.unlink(missing_ok=True)
        return data.size / sr

    return await asyncio.to_thread(_post)

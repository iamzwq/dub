"""edge-tts 合成工具（带重试、缓存、静音裁剪）。"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import edge_tts
import numpy as np
import soundfile as sf

from .log import log
from .media import read_audio, trim_silence, write_wav
from .workspace import short_hash

SR = 24000


def rate_str(rate: int) -> str:
    return f"{rate:+d}%"


def tts_filename(sid: int, text: str, voice: str, rate: int) -> str:
    return f"{sid:04d}_{short_hash(text, voice, rate)}.wav"


def speakable(text: str) -> bool:
    return bool(re.search(r"\w", text))


def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


async def synthesize(
    text: str,
    voice: str,
    rate: int,
    out: Path,
    retries: int = 3,
    trim_db: float = -45.0,
    proxy: str | None = None,
) -> float:
    """合成一句并保存为裁剪掉首尾静音的 wav，返回时长（秒）。已存在则直接复用。"""
    if out.exists():
        return wav_duration(out)
    if not speakable(text):
        write_wav(out, np.zeros(0, dtype=np.float32), SR)
        return 0.0
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

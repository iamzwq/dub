"""配音合成工具（edge-tts / MOSS-TTS-Nano / DashScope，带重试、缓存、静音裁剪）。"""
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


def tts_filename(sid: int, text: str, t: TTSConfig, rate: int) -> str:
    return f"{sid:04d}_{short_hash(text, cache_key(t), rate)}.wav"


@functools.lru_cache(maxsize=8)
def _file_hash(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"参考音频不存在: {path}")
    return hashlib.sha1(p.read_bytes()).hexdigest()[:10]


def voice_key(t: TTSConfig) -> str:
    """音色标识（用于日志显示）。"""
    if t.engine == "edge":
        return t.voice
    if t.engine == "dashscope":
        return f"dashscope|{t.dashscope.model}|{t.dashscope.voice}"
    m = t.moss
    ref = f"file:{_file_hash(m.prompt_audio)}" if m.prompt_audio else f"demo:{m.demo_id}"
    return f"moss_nano|{ref}|seed={m.seed}|frames={m.max_new_frames}"


def cache_key(t: TTSConfig) -> str:
    """影响合成结果的全部参数（不含语速），用于缓存文件名与步骤指纹。"""
    return f"{voice_key(t)}|trim={t.trim_silence_db:g}"


def concurrency(t: TTSConfig) -> int:
    if t.engine == "moss_nano":
        return t.moss.concurrency
    if t.engine == "dashscope":
        return t.dashscope.concurrency
    return t.concurrency


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
    if t.engine == "dashscope":
        return await _synthesize_dashscope(t, text, rate, out)
    return await _synthesize_edge(t, text, rate, out)


async def _retry(name: str, retries: int, call):
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return await call()
        except Exception as e:  # noqa: BLE001
            last = e
            log.debug("%s 失败（第 %d 次）：%s", name, attempt, e)
            if attempt < retries:
                await asyncio.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"{name} 合成失败：{last}") from last


def _save_trimmed(src: Path, out: Path, trim_db: float, *temps: Path) -> float:
    """读取音频、裁掉首尾静音后写入 out，并删除临时文件，返回时长。"""
    data, sr = read_audio(src)
    data = trim_silence(data, sr, trim_db)
    write_wav(out, data, sr)
    for p in (src, *temps):
        p.unlink(missing_ok=True)
    return data.size / sr


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

    async def call() -> bytes:
        # 不走系统代理：服务通常在本机或局域网
        async with httpx.AsyncClient(timeout=m.timeout, trust_env=False) as client:
            r = await client.post(url, data=form, files=files)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return base64.b64decode(r.json()["audio_base64"])

    wav = await _retry(f"MOSS-TTS-Nano（{url}）", t.retries, call)

    def _post() -> float:
        raw = out.with_suffix(".raw.wav")
        conv = out.with_suffix(".conv.wav")
        raw.write_bytes(wav)
        # 48kHz 立体声 → 与 edge-tts 一致的 24kHz 单声道；模型无语速参数，用 atempo 实现 rate
        af = ["-filter:a", atempo_filter(1 + rate / 100)] if rate else []
        run_ffmpeg(["-i", str(raw), *af, "-ac", "1", "-ar", str(SR), str(conv)])
        return _save_trimmed(conv, out, t.trim_silence_db, raw)

    return await asyncio.to_thread(_post)


async def _synthesize_dashscope(t: TTSConfig, text: str, rate: int, out: Path) -> float:
    import dashscope
    from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

    d = t.dashscope
    api_key = d.resolved_api_key()
    if not api_key:
        raise RuntimeError("DashScope 未配置 API Key：设置 tts.dashscope.api_key 或环境变量 DASHSCOPE_API_KEY")
    dashscope.api_key = api_key

    def _call() -> bytes | None:
        synth = SpeechSynthesizer(
            model=d.model,
            voice=d.voice,
            format=AudioFormat.WAV_24000HZ_MONO_16BIT,
            speech_rate=max(0.5, min(2.0, 1 + rate / 100)),
            url=d.url or None,
        )
        return synth.call(text, timeout_millis=int(d.timeout * 1000))

    async def call() -> bytes:
        wav = await asyncio.to_thread(_call)
        if not wav:
            raise RuntimeError("未收到音频")
        return wav

    wav = await _retry("DashScope", t.retries, call)

    def _post() -> float:
        raw = out.with_suffix(".raw.wav")
        raw.write_bytes(wav)
        return _save_trimmed(raw, out, t.trim_silence_db)

    return await asyncio.to_thread(_post)


async def _synthesize_edge(t: TTSConfig, text: str, rate: int, out: Path) -> float:
    tmp = out.with_suffix(".mp3")

    async def call() -> None:
        await edge_tts.Communicate(text, t.voice, rate=rate_str(rate), proxy=t.proxy or None).save(str(tmp))
        if not (tmp.exists() and tmp.stat().st_size > 0):
            raise RuntimeError("未收到音频")

    try:
        await _retry("edge-tts", t.retries, call)
    except RuntimeError as e:
        if not isinstance(e.__cause__, edge_tts.exceptions.NoAudioReceived):
            raise
        log.warning("edge-tts 无音频输出，按静音处理：%s", text)
        write_wav(out, np.zeros(0, dtype=np.float32), SR)
        return 0.0
    return await asyncio.to_thread(_save_trimmed, tmp, out, t.trim_silence_db)

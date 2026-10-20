"""⑤ 时长对齐：提语速 → AI 精简译文 → atempo 微调 → 有限顺延。画面保持原速。"""
from __future__ import annotations

import asyncio
import json
import math

from ..context import Context, StepError
from ..llm import LLM, LLMError
from ..log import count_progress, log
from ..media import atempo
from ..models import Sentence
from ..prompts import SHORTEN_SYSTEM
from ..text import zh_len
from ..timeline import Clip, schedule
from ..tts import synthesize, tts_filename, wav_duration
from .translate import load_sentences, slot_seconds


def fingerprint(cfg) -> dict:
    return {
        "align": cfg.align.model_dump(),
        "voice": cfg.tts.voice,
        "rate": cfg.tts.rate,
        "max_rate": cfg.tts.max_rate,
        "llm": cfg.llm.identity() if cfg.align.shorten_rounds else "",
    }


def run(ctx: Context) -> None:
    ctx.run_async(_run(ctx))


async def _shorten(llm: LLM, src: str, zh: str, max_chars: int) -> str:
    def validate(data):
        text = str((data or {}).get("zh", "")).strip()
        if not text:
            raise ValueError("缺少 zh")
        return text

    user = json.dumps({"原文": src, "当前译文": zh, "max_chars": max_chars}, ensure_ascii=False)
    return await llm.chat_json(SHORTEN_SYSTEM.format(max_chars=max_chars), user, validate=validate, label="精简")


async def _run(ctx: Context) -> None:
    ws, cfg = ctx.ws, ctx.cfg
    t, a = cfg.tts, cfg.align
    sentences: list[Sentence] = load_sentences(ctx)
    tts_items = {d["id"]: d for d in ws.read_json("tts.json", []) or []}
    if len(tts_items) != len(sentences):
        raise StepError("tts.json 与 sentences.json 不一致，请重跑配音步骤")
    meta = ws.read_json("meta.json") or {}
    video_duration = (meta.get("video") or {}).get("duration") or meta.get("duration")

    entries: list[dict] = []
    for i, s in enumerate(sentences):
        it = tts_items[s.id]
        entries.append({
            "id": s.id, "start": s.start, "end": s.end, "src": s.text, "zh": it["zh"],
            "slot": round(slot_seconds(sentences, i, video_duration), 3),
            "rate": it["rate"], "file": it["file"], "duration": it["duration"],
            "rate_adjusted": False, "shortened": 0,
        })

    over = [e for e in entries if e["duration"] > e["slot"] + a.overflow_tolerance]
    log.info("%d / %d 句配音超出时段，开始调整", len(over), len(entries))

    llm: LLM | None = None
    if over and a.shorten_rounds > 0:
        try:
            llm = ctx.llm()
        except LLMError as e:
            log.warning("未配置 LLM，跳过精简译文：%s", e)

    sem = asyncio.Semaphore(t.concurrency)

    async def synth(e: dict, text: str, rate: int) -> tuple[str, float]:
        name = tts_filename(e["id"], text, t.voice, rate)
        async with sem:
            dur = await synthesize(text, t.voice, rate, ws.tts_dir / name, t.retries, t.trim_silence_db, t.proxy)
        return name, dur

    async def fix(e: dict) -> None:
        text, rate, dur, name = e["zh"], e["rate"], e["duration"], e["file"]
        slot = e["slot"]
        for rnd in range(a.shorten_rounds + 1):
            if dur <= slot + a.overflow_tolerance:
                break
            factor = (1 + rate / 100) * dur / slot * 1.03
            new_rate = min(t.max_rate, math.ceil((factor - 1) * 100))
            if new_rate > rate:
                name, dur = await synth(e, text, new_rate)
                rate = new_rate
                e["rate_adjusted"] = True
                if dur <= slot + a.overflow_tolerance:
                    break
            if rnd == a.shorten_rounds or llm is None:
                break
            target = max(2, math.floor(zh_len(text) * slot / dur * 0.95))
            try:
                new_text = await _shorten(llm, e["src"], text, target)
            except LLMError as err:
                log.debug("精简失败 #%d：%s", e["id"], err)
                break
            if zh_len(new_text) >= zh_len(text):
                break
            log.debug("精简 #%d：%s → %s", e["id"], text, new_text)
            text = new_text
            e["shortened"] += 1
            rate = t.rate
            name, dur = await synth(e, text, rate)
        e.update(zh=text, rate=rate, duration=round(dur, 3), file=name)
        progress.update(task, advance=1)

    with count_progress() as progress:
        task = progress.add_task("调整语速/精简", total=len(over), info="")
        results = await asyncio.gather(*(fix(e) for e in over), return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        raise StepError(f"{len(errors)} 句调整失败：{errors[0]}")

    clips = [
        Clip(e["id"], e["start"], e["start"] + e["slot"], e["duration"], 0.0 if k == len(entries) - 1 else None)
        for k, e in enumerate(entries)
    ]
    placements = schedule(clips, a.atempo_max, a.max_drift, a.drift_atempo_max, a.overflow_tolerance)

    need_tempo = [(e, p) for e, p in zip(entries, placements) if p.tempo > 1.001]

    async def apply(e: dict, p) -> None:
        src = ws.tts_dir / e["file"]
        dst_name = f"{src.stem}_t{p.tempo:.3f}.wav"
        dst = ws.tts_dir / dst_name
        if not dst.exists():
            async with sem:
                await asyncio.to_thread(atempo, src, dst, p.tempo)
        e["file"] = dst_name
        e["duration"] = round(wav_duration(dst), 3)
        progress.update(task2, advance=1)

    if need_tempo:
        with count_progress() as progress:
            task2 = progress.add_task("atempo 压缩", total=len(need_tempo), info="")
            await asyncio.gather(*(apply(e, p) for e, p in need_tempo))

    timeline = []
    for e, p in zip(entries, placements):
        timeline.append({
            **e,
            "tempo": p.tempo,
            "drift": p.drift,
            "play_start": p.play_start,
            "play_end": round(p.play_start + e["duration"], 3),
        })
    ws.write_json("timeline.json", timeline)

    max_drift = max((x["drift"] for x in timeline), default=0.0)
    report = {
        "sentences": len(timeline),
        "overflow_before": len(over),
        "rate_adjusted": sum(1 for x in timeline if x["rate_adjusted"]),
        "shortened": sum(1 for x in timeline if x["shortened"]),
        "atempo": len(need_tempo),
        "max_tempo": max((x["tempo"] for x in timeline), default=1.0),
        "drifted": sum(1 for x in timeline if x["drift"] > 0.05),
        "max_drift": max_drift,
        "drift_over_cap": sum(1 for x in timeline if x["drift"] > a.max_drift + 0.01),
    }
    ws.update_report("align", report)
    log.info(
        "对齐完成：提速 %d 句，精简 %d 句，atempo %d 句（最大 %.2fx），顺延 %d 句（最大 %.2fs）",
        report["rate_adjusted"], report["shortened"], report["atempo"], report["max_tempo"],
        report["drifted"], max_drift,
    )
    if report["drift_over_cap"]:
        log.warning("%d 句的顺延超过上限 %.1fs（配音过长且已压缩到上限）", report["drift_over_cap"], a.max_drift)

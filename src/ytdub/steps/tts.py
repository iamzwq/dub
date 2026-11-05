"""④ 并发配音（基准语速）。"""
from __future__ import annotations

import asyncio

from ..context import Context, StepError
from ..log import count_progress, log
from ..tts import cache_key, concurrency, synthesize, tts_filename, voice_key


def fingerprint(cfg) -> dict:
    return {"tts": cache_key(cfg.tts), "rate": cfg.tts.rate}


def load_translation(ctx: Context) -> list[dict]:
    return ctx.ws.require_json("translation.json", "translate")


def run(ctx: Context) -> None:
    ctx.run_async(_run(ctx))


async def _run(ctx: Context) -> None:
    ws, t = ctx.ws, ctx.cfg.tts
    items = load_translation(ctx)
    workers = concurrency(t)
    sem = asyncio.Semaphore(workers)
    results: dict[int, dict] = {}
    failures: list[str] = []
    log.info("引擎 %s，音色 %s，语速 %+d%%，并发 %d，共 %d 句", t.engine, voice_key(t), t.rate, workers, len(items))

    with count_progress() as progress:
        task = progress.add_task("配音", total=len(items), info="")

        async def one(item: dict) -> None:
            name = tts_filename(item["id"], item["zh"], t, t.rate)
            async with sem:
                try:
                    dur = await synthesize(t, item["zh"], t.rate, ws.tts_dir / name)
                except Exception as e:  # noqa: BLE001
                    failures.append(f"#{item['id']}: {e}")
                    progress.update(task, advance=1, info=f"失败 {len(failures)}")
                    return
            results[item["id"]] = {"id": item["id"], "zh": item["zh"], "rate": t.rate, "file": name, "duration": round(dur, 3)}
            progress.update(task, advance=1, info=f"失败 {len(failures)}" if failures else "")

        await asyncio.gather(*(one(it) for it in items))

    if failures:
        raise StepError(f"{len(failures)} 句配音失败（已完成部分已缓存，可重跑）：{failures[0]}")
    ws.write_json("tts.json", [results[it["id"]] for it in items])
    total = sum(r["duration"] for r in results.values())
    log.info("配音完成：%d 句，语音总时长 %.1fs", len(results), total)

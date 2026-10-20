"""③ AI 翻译：主题摘要 + 自动术语表 + 用户术语表，带上下文批量翻译。"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import yaml

from ..context import Context, StepError
from ..llm import LLM, LLMError, LLMFatalError
from ..log import count_progress, log
from ..models import Sentence
from ..prompts import SUMMARY_SYSTEM, TRANSLATE_SYSTEM
from ..text import lang_name
from ..workspace import fingerprint as fp_hash

_SIMPLIFIED = {"zh", "zh-hans", "zh-cn", "zh-sg"}


def _glossary_path(cfg) -> Path:
    return Path(cfg.translate.glossary)


def load_user_glossary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise StepError(f"术语表格式错误（应为 原文: 译文 的映射）：{path}")
    return {str(k): str(v) for k, v in data.items() if k is not None and v is not None}


def fingerprint(cfg) -> dict:
    t = cfg.translate
    g = _glossary_path(cfg)
    return {
        "llm": cfg.llm.identity(),
        "cps": t.chars_per_second,
        "batch": t.batch_size,
        "ctx": t.context_size,
        "glossary": fp_hash(load_user_glossary(g)) if g.exists() else "",
    }


def load_sentences(ctx: Context) -> list[Sentence]:
    data = ctx.ws.read_json("sentences.json")
    if not data:
        raise StepError("缺少 sentences.json，请先执行断句步骤")
    return [Sentence.from_dict(d) for d in data]


def slot_seconds(sentences: list[Sentence], i: int, video_duration: float | None = None) -> float:
    s = sentences[i]
    if i + 1 < len(sentences):
        return max(sentences[i + 1].start - s.start, 0.3)
    end = video_duration if video_duration and video_duration > s.end else s.end + 1.0
    return max(end - s.start, 0.3)


def run(ctx: Context) -> None:
    ws, cfg = ctx.ws, ctx.cfg
    sentences = load_sentences(ctx)
    meta = ws.read_json("meta.json") or {}
    lang = meta.get("language", "en")
    duration = (meta.get("video") or {}).get("duration") or meta.get("duration")

    if lang.lower() in _SIMPLIFIED:
        log.info("源语言为简体中文，跳过翻译")
        ws.write_json("translation.json", [{"id": s.id, "src": s.text, "zh": s.text} for s in sentences])
        title = meta.get("title", "")
        ws.write_json("titles.json", {"zh_title": title, "cover_title": title, "summary": "", "terms": []})
        return

    try:
        llm = ctx.llm()
    except LLMError as e:
        raise StepError(str(e)) from e
    ctx.run_async(_translate_all(ctx, llm, sentences, lang, meta, duration))


async def _summarize(llm: LLM, sentences: list[Sentence], meta: dict) -> dict:
    transcript = " ".join(s.text for s in sentences)
    if len(transcript) > 15000:
        transcript = transcript[:15000] + " ..."
    user = json.dumps(
        {
            "title": meta.get("title", ""),
            "description": (meta.get("description") or "")[:2000],
            "transcript": transcript,
        },
        ensure_ascii=False,
    )

    def validate(data):
        if not isinstance(data, dict) or "summary" not in data:
            raise ValueError("缺少 summary")
        terms = [
            {"src": str(t["src"]).strip(), "zh": str(t["zh"]).strip()}
            for t in data.get("terms") or []
            if isinstance(t, dict) and t.get("src") and t.get("zh")
        ]
        return {
            "summary": str(data.get("summary", "")),
            "terms": terms,
            "zh_title": str(data.get("zh_title") or meta.get("title", "")),
            "cover_title": str(data.get("cover_title") or data.get("zh_title") or meta.get("title", "")),
        }

    return await llm.chat_json(SUMMARY_SYSTEM, user, validate=validate, label="摘要/术语")


CACHE = "translate.cache.json"


async def _translate_all(ctx: Context, llm: LLM, sentences: list[Sentence], lang: str, meta: dict, duration) -> None:
    ws, cfg = ctx.ws, ctx.cfg
    tcfg = cfg.translate

    # 续跑缓存：只有模型、术语表、翻译参数都没变时才复用（中断后重跑可接着翻译）
    cache_key = fp_hash(fingerprint(cfg))
    cache = ws.read_json(CACHE) or {}
    if cache.get("fp") != cache_key:
        cache = {"fp": cache_key, "titles": None, "items": {}}

    titles = cache.get("titles")
    if not titles:
        log.info("生成视频摘要与术语表…")
        try:
            titles = await _summarize(llm, sentences, meta)
        except LLMFatalError:
            raise
        except LLMError as e:
            log.warning("摘要/术语生成失败，继续翻译：%s", e)
            title = meta.get("title", "")
            titles = {"summary": "", "terms": [], "zh_title": title, "cover_title": title}
        cache["titles"] = titles
        ws.write_json(CACHE, cache)
    ws.write_json("titles.json", titles)
    log.info("中文标题：%s；封面标题：%s", titles["zh_title"], titles["cover_title"])

    glossary = {t["src"]: t["zh"] for t in titles.get("terms", [])}
    user_glossary = load_user_glossary(_glossary_path(cfg))
    glossary.update(user_glossary)
    ws.write_json("glossary.json", glossary)
    log.info("术语表：自动 %d 条，用户 %d 条", len(titles.get("terms", [])), len(user_glossary))

    previous = cache["items"]
    results: dict[int, str] = {
        s.id: previous[str(s.id)]["zh"]
        for s in sentences
        if str(s.id) in previous and previous[str(s.id)].get("src") == s.text and previous[str(s.id)].get("zh")
    }

    system = TRANSLATE_SYSTEM.format(
        lang=lang_name(lang), title=meta.get("title", ""), summary=titles.get("summary") or "（无）"
    )
    index = {s.id: i for i, s in enumerate(sentences)}
    max_chars = {
        s.id: max(4, math.floor(slot_seconds(sentences, i, duration) * tcfg.chars_per_second))
        for i, s in enumerate(sentences)
    }

    def save() -> None:
        ws.write_json(
            "translation.json",
            [{"id": s.id, "src": s.text, "zh": results[s.id]} for s in sentences if s.id in results],
        )
        cache["items"] = {str(s.id): {"src": s.text, "zh": results[s.id]} for s in sentences if s.id in results}
        ws.write_json(CACHE, cache)

    async def translate_batch(batch: list[Sentence]) -> None:
        first, last = index[batch[0].id], index[batch[-1].id]
        n = tcfg.context_size
        before = [s.text for s in sentences[max(0, first - n): first]]
        after = [s.text for s in sentences[last + 1: last + 1 + n]]
        text_blob = " ".join([*before, *(s.text for s in batch), *after]).lower()
        terms = {k: v for k, v in glossary.items() if k.lower() in text_blob}
        payload = {
            "context_before": before,
            "sentences": [{"id": s.id, "text": s.text, "max_chars": max_chars[s.id]} for s in batch],
            "context_after": after,
            "glossary": terms,
        }
        expected = {s.id for s in batch}

        def validate(data):
            items = data.get("translations") if isinstance(data, dict) else data
            if not isinstance(items, list):
                raise ValueError("缺少 translations 数组")
            out = {}
            for it in items:
                if isinstance(it, dict) and "id" in it and str(it.get("zh", "")).strip():
                    out[int(it["id"])] = str(it["zh"]).strip()
            if set(out) != expected:
                raise ValueError(f"id 不匹配：缺少 {sorted(expected - set(out))}，多出 {sorted(set(out) - expected)}")
            return out

        try:
            out = await llm.chat_json(system, json.dumps(payload, ensure_ascii=False), validate=validate,
                                      label=f"翻译 #{batch[0].id}-{batch[-1].id}")
        except LLMFatalError:
            raise
        except LLMError:
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            log.warning("批次 #%d-%d 多次失败，拆成两半重试", batch[0].id, batch[-1].id)
            await translate_batch(batch[:mid])
            await translate_batch(batch[mid:])
            return
        results.update(out)
        save()
        progress.update(task, advance=len(batch))

    todo = [s for s in sentences if s.id not in results]
    if not todo:
        log.info("翻译已全部完成（使用缓存）")
        return
    # 按原顺序把待翻译句子切成批次（批次内保持连续）
    batches: list[list[Sentence]] = []
    cur: list[Sentence] = []
    for s in todo:
        if cur and (len(cur) >= tcfg.batch_size or index[s.id] != index[cur[-1].id] + 1):
            batches.append(cur)
            cur = []
        cur.append(s)
    if cur:
        batches.append(cur)

    log.info("翻译 %d 句（缓存命中 %d），共 %d 批，模型 %s", len(todo), len(results), len(batches), llm.model)
    with count_progress() as progress:
        task = progress.add_task("翻译", total=len(sentences), completed=len(results), info="")
        coros = [translate_batch(b) for b in batches]
        outcomes = await asyncio.gather(*coros, return_exceptions=True)
    errors = [o for o in outcomes if isinstance(o, Exception)]
    save()
    if errors:
        raise StepError(f"{len(errors)} 个批次翻译失败（已完成部分已保存，可重跑续译）：{errors[0]}")
    log.info("翻译完成：%d 句，tokens 输入 %d / 输出 %d", len(results), llm.prompt_tokens, llm.completion_tokens)

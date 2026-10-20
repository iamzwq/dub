"""② 字幕解析与断句：有标点 → 按句末标点断句；无标点的自动字幕 → AI 断句。"""
from __future__ import annotations

import json
from difflib import SequenceMatcher

from ..context import Context, StepError
from ..llm import LLM, LLMError, LLMFatalError
from ..log import count_progress, log
from ..models import Sentence, Token
from ..prompts import SEGMENT_SYSTEM
from ..subtitles import groups_to_sentences, parse_json3, parse_vtt, segment_punct, split_long
from ..text import has_punctuation, is_no_space, join_tokens, lang_name, split_units


def fingerprint(cfg) -> dict:
    s = cfg.segment
    fp = {"v": 2, "mode": s.mode, "gap": s.max_gap_seconds}
    if s.mode != "punct":
        fp.update({"llm": cfg.llm.identity(), "chunk": s.ai_chunk_words, "max": s.max_sentence_seconds})
    return fp


def load_tokens(ctx: Context) -> tuple[list[Token], str, dict]:
    meta = ctx.ws.read_json("meta.json")
    if not meta:
        raise StepError("缺少 meta.json，请先执行下载步骤")
    sub = meta["subtitle"]
    content = ctx.ws.path(sub["file"]).read_text(encoding="utf-8", errors="replace")
    tokens = parse_json3(json.loads(content)) if sub["ext"] == "json3" else parse_vtt(content)
    if not tokens:
        raise StepError("字幕内容为空")
    return tokens, meta["language"], meta


def run(ctx: Context) -> None:
    cfg = ctx.cfg.segment
    tokens, lang, meta = load_tokens(ctx)
    full_text = join_tokens([t.text for t in tokens], lang)
    punct = has_punctuation(full_text, lang)
    mode = cfg.mode
    if mode == "auto":
        mode = "punct" if punct else "ai"
    log.info(
        "字幕：%s，%d 个片段，%s标点 → %s",
        "手动上传" if meta["subtitle"]["kind"] == "manual" else "自动生成",
        len(tokens), "有" if punct else "无", "按句末标点断句" if mode == "punct" else "AI 断句",
    )

    if mode == "punct":
        sentences = segment_punct(tokens, lang, cfg.max_gap_seconds)
    else:
        try:
            llm = ctx.llm()
        except LLMError as e:
            raise StepError(f"AI 断句需要 LLM：{e}（或使用 --segment punct）") from e
        sentences = ctx.run_async(_segment_ai(llm, tokens, lang, ctx))

    if not sentences:
        raise StepError("断句结果为空")
    ctx.ws.write_json("sentences.json", [s.to_dict() for s in sentences])
    durs = [s.end - s.start for s in sentences]
    log.info("断句完成：%d 句，平均 %.1fs，最长 %.1fs", len(sentences), sum(durs) / len(durs), max(durs))
    ctx.ws.update_report("segment", {"mode": mode, "has_punctuation": punct, "sentences": len(sentences)})


# ---------------------------------------------------------------- AI 断句
def _map_output_to_orig(orig: list[str], flat: list[str]) -> tuple[list[int], float]:
    sm = SequenceMatcher(None, orig, flat, autojunk=False)
    out2orig = [0] * (len(flat) + 1)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        for j in range(j1, j2):
            if tag == "equal":
                out2orig[j] = i1 + (j - j1)
            elif tag == "replace":
                out2orig[j] = i1 + min(int((j - j1) * (i2 - i1) / (j2 - j1)), i2 - i1 - 1)
            elif tag == "insert":
                out2orig[j] = i1
    out2orig[len(flat)] = len(orig)
    return out2orig, sm.ratio()


def align_sentences(chunk_units: list[list[str]], sentences: list[str], lang: str) -> list[tuple[int, str]]:
    """把 LLM 输出的句子映射回 token 下标，返回 [(起始 token 下标, 句子文本)]。"""
    orig: list[str] = []
    owner: list[int] = []
    for ti, units in enumerate(chunk_units):
        for u in units:
            orig.append(u)
            owner.append(ti)
    sent_units = [split_units(s, lang) for s in sentences]
    flat = [u for us in sent_units for u in us]
    out2orig, ratio = _map_output_to_orig(orig, flat)
    if ratio < 0.85:
        raise ValueError(f"LLM 输出与原文差异过大（相似度 {ratio:.2f}）")
    result: list[tuple[int, str]] = []
    pos = 0
    for text, units in zip(sentences, sent_units):
        if not units:
            if result:
                result[-1] = (result[-1][0], result[-1][1] + text)
            continue
        oi = out2orig[pos]
        ti = owner[oi] if oi < len(owner) else len(chunk_units)
        pos += len(units)
        if result and (ti <= result[-1][0] or ti >= len(chunk_units)):
            sep = "" if is_no_space(lang) else " "
            result[-1] = (result[-1][0], result[-1][1] + sep + text)
        else:
            result.append((0 if not result else ti, text))
    return result


async def _segment_chunk(llm: LLM, chunk: list[Token], units: list[list[str]], lang: str, max_words: int):
    system = SEGMENT_SYSTEM.format(lang=lang_name(lang), max_words=max_words)
    user = join_tokens([t.text for t in chunk], lang)

    def validate(data):
        sents = data.get("sentences") if isinstance(data, dict) else data
        if not isinstance(sents, list):
            raise ValueError("缺少 sentences 数组")
        sents = [s.strip() for s in sents if isinstance(s, str) and s.strip()]
        if not sents:
            raise ValueError("sentences 为空")
        return align_sentences(units, sents, lang)

    return await llm.chat_json(system, user, validate=validate, label="断句")


async def _segment_ai(llm: LLM, tokens: list[Token], lang: str, ctx: Context) -> list[Sentence]:
    cfg = ctx.cfg.segment
    tok_units = [split_units(t.text, lang) for t in tokens]
    chunk_size = cfg.ai_chunk_words * (2 if is_no_space(lang) else 1)
    # 按正常语速估算单句上限：英文等约 2.5 词/秒，中日文约 4 字/秒
    max_words = round(cfg.max_sentence_seconds * (4 if is_no_space(lang) else 2.5))
    groups: list[list[Token]] = []
    texts: list[str] = []
    n = len(tokens)
    i = 0
    with count_progress() as progress:
        task = progress.add_task("AI 断句", total=n, info="")
        while i < n:
            j, count = i, 0
            while j < n and count < chunk_size:
                count += len(tok_units[j])
                j += 1
            chunk = tokens[i:j]
            is_last = j >= n
            try:
                result = await _segment_chunk(llm, chunk, tok_units[i:j], lang, max_words)
            except LLMFatalError as e:
                raise StepError(str(e)) from e
            except LLMError as e:
                log.warning("AI 断句失败，该段回退为按停顿断句：%s", e)
                for g in split_long(chunk, cfg.max_sentence_seconds):
                    groups.append(g)
                    texts.append(join_tokens([t.text for t in g], lang))
                i = j
                progress.update(task, completed=i)
                continue

            if not is_last and len(result) > 1:
                keep, next_i = result[:-1], i + result[-1][0]
            else:
                keep, next_i = result, j
            for k, (rel, text) in enumerate(keep):
                rel_end = keep[k + 1][0] if k + 1 < len(keep) else next_i - i
                g = chunk[rel:rel_end]
                if not g:
                    continue
                if g[-1].end - g[0].start > cfg.max_sentence_seconds and len(g) > 1:
                    for sub in split_long(g, cfg.max_sentence_seconds):
                        groups.append(sub)
                        texts.append(join_tokens([t.text for t in sub], lang))
                else:
                    groups.append(g)
                    texts.append(text)
            i = next_i
            progress.update(task, completed=i, info=f"{len(groups)} 句")
    return groups_to_sentences(groups, lang, texts)

"""① 下载视频、封面、字幕与元数据（yt-dlp）。"""
from __future__ import annotations

import copy
import io
from pathlib import Path

from PIL import Image

from ..context import Context, StepError
from ..log import bytes_progress, log
from ..media import probe
from ..subtitles import parse_json3, parse_vtt
from ..text import base_lang

SUB_EXTS = ("json3", "vtt")


def fingerprint(cfg) -> dict:
    return {"max_height": cfg.download.max_height, "lang": cfg.source_lang}


class _YdlLogger:
    def __init__(self) -> None:
        self._js_warned = False

    def debug(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)

    def info(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)

    def warning(self, msg: str) -> None:
        low = msg.lower()
        if "javascript" in low or "js runtime" in low or "challenge" in low or "deno" in low:
            log.debug("yt-dlp: %s", msg)
            if not self._js_warned:
                self._js_warned = True
                log.warning("yt-dlp 未检测到 JS 运行时（未安装 deno），部分格式可能缺失，将下载可用的最佳格式")
            return
        log.warning("yt-dlp: %s", msg)

    def error(self, msg: str) -> None:
        log.error("yt-dlp: %s", msg)


def _base_opts(ctx: Context) -> dict:
    d = ctx.cfg.download
    opts: dict = {
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
        "logger": _YdlLogger(),
        "retries": d.retries,
        "fragment_retries": d.retries,
        "concurrent_fragment_downloads": 4,
    }
    if ctx.cookies:
        opts["cookiefile"] = str(ctx.cookies)
    if d.sleep_requests:
        opts["sleep_interval_requests"] = d.sleep_requests
    if d.proxy:
        opts["proxy"] = d.proxy
    if d.ratelimit:
        from yt_dlp.utils import parse_bytes

        opts["ratelimit"] = parse_bytes(d.ratelimit)
    return opts


def _match_key(keys: list[str], lang: str) -> str | None:
    if lang in keys:
        return lang
    b = base_lang(lang)
    for k in keys:
        if k.lower() == b:
            return k
    for k in keys:
        if base_lang(k) == b:
            return k
    return None


def choose_subtitle(info: dict, lang_override: str = "") -> tuple[str, str, str, dict]:
    """返回 (源语言, 类型 manual/auto, 轨道 key, 格式条目)。"""
    subs = {k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat" and v}
    autos = {k: v for k, v in (info.get("automatic_captions") or {}).items() if v}
    orig_keys = [k for k in autos if k.endswith("-orig")]

    src = lang_override or info.get("language") or (orig_keys[0][: -len("-orig")] if orig_keys else "")
    if not src and len(subs) == 1:
        src = next(iter(subs))
    if not src:
        raise StepError("无法确定视频的源语言，请使用 --lang 指定（例如 --lang en）")

    key = _match_key(list(subs), src)
    if key:
        return src, "manual", key, _pick_format(subs[key], key)

    orig = [k for k in orig_keys if base_lang(k) == base_lang(src)]
    key = orig[0] if orig else _match_key(list(autos), src)
    if key:
        return src, "auto", key, _pick_format(autos[key], key)

    raise StepError(
        f"该视频没有 {src} 语言的字幕（手动字幕：{', '.join(subs) or '无'}；"
        f"自动字幕原语言：{', '.join(orig_keys) or '无'}），已中断"
    )


def _pick_format(formats: list[dict], key: str) -> dict:
    for ext in SUB_EXTS:
        for f in formats:
            if f.get("ext") == ext and f.get("url"):
                return f
    raise StepError(f"字幕轨道 {key} 没有可解析的格式（需要 json3 或 vtt）")


def _download_subtitle(ydl, ctx: Context, fmt: dict) -> Path:
    with ydl.urlopen(fmt["url"]) as resp:
        raw = resp.read()
    if not raw.strip():
        raise StepError("字幕下载结果为空（可能被 YouTube 限流或需要 PO Token），请更新 yt-dlp 或更换 cookies 后重试")
    path = ctx.ws.path(f"subs.src.{fmt['ext']}")
    path.write_bytes(raw)
    content = raw.decode("utf-8", errors="replace")
    if fmt["ext"] == "json3":
        import json

        tokens = parse_json3(json.loads(content))
    else:
        tokens = parse_vtt(content)
    if not tokens:
        raise StepError("字幕内容为空，已中断")
    log.info("字幕已下载：%s（%d 个片段）", path.name, len(tokens))
    return path


def _download_thumbnail(ydl, ctx: Context, info: dict) -> None:
    thumbs = sorted(
        (t for t in info.get("thumbnails") or [] if t.get("url")),
        key=lambda t: (t["preference"] if t.get("preference") is not None else -100, t.get("width") or 0),
        reverse=True,
    )
    if info.get("thumbnail"):
        thumbs.insert(0, {"url": info["thumbnail"]})
    for t in thumbs[:8]:
        try:
            with ydl.urlopen(t["url"]) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            if img.width < 320:
                continue
            img.save(ctx.ws.path("cover.jpg"), quality=95)
            log.info("封面已下载：%dx%d", img.width, img.height)
            return
        except Exception as e:  # noqa: BLE001
            log.debug("封面下载失败 %s：%s", t["url"], e)
    log.warning("未能下载封面图")


def run(ctx: Context) -> None:
    import yt_dlp

    ws, cfg = ctx.ws, ctx.cfg
    h = cfg.download.max_height
    opts = _base_opts(ctx)
    with yt_dlp.YoutubeDL(opts) as ydl:
        log.info("获取视频信息…")
        info = ydl.extract_info(ctx.url, download=False, process=False)
        if info.get("_type") not in (None, "video"):
            raise StepError("仅支持单个视频 URL")
        if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
            raise StepError("不支持直播或尚未开始的视频")
        log.info("标题：%s（%ss）", info.get("title"), info.get("duration"))

        src_lang, kind, key, fmt = choose_subtitle(info, cfg.source_lang)
        log.info("源语言：%s；字幕：%s（%s，%s）", src_lang, "手动上传" if kind == "manual" else "自动生成", key, fmt["ext"])
        sub_path = _download_subtitle(ydl, ctx, fmt)
        _download_thumbnail(ydl, ctx, info)

        meta = {
            "id": info.get("id"),
            "title": info.get("title"),
            "description": info.get("description") or "",
            "duration": info.get("duration"),
            "uploader": info.get("uploader"),
            "channel": info.get("channel"),
            "upload_date": info.get("upload_date"),
            "webpage_url": info.get("webpage_url") or ctx.url,
            "chapters": info.get("chapters") or [],
            "language": src_lang,
            "subtitle": {"kind": kind, "key": key, "ext": fmt["ext"], "file": sub_path.name},
        }
        ws.write_json("meta.json", meta)

    with bytes_progress() as progress:
        tasks: dict[str, int] = {}

        def hook(d: dict) -> None:
            fn = d.get("filename") or d.get("tmpfilename") or "?"
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if fn not in tasks:
                vcodec = (d.get("info_dict") or {}).get("vcodec")
                desc = "下载音频" if vcodec == "none" else "下载视频"
                tasks[fn] = progress.add_task(desc, total=total)
            if d["status"] == "downloading":
                progress.update(tasks[fn], completed=d.get("downloaded_bytes") or 0, total=total)
            elif d["status"] == "finished":
                done = d.get("downloaded_bytes") or total or 0
                progress.update(tasks[fn], completed=done, total=total or done)

        def pp_hook(d: dict) -> None:
            if d["status"] == "started" and d.get("postprocessor") == "Merger":
                log.info("合并音视频流…")

        dl_opts = {
            **opts,
            "format": f"bv*[height<={h}]+ba/b[height<={h}]/bv*+ba/b",
            "merge_output_format": "mp4",
            "outtmpl": {"default": str(ws.path("video.%(ext)s"))},
            "progress_hooks": [hook],
            "postprocessor_hooks": [pp_hook],
        }
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.process_ie_result(copy.deepcopy(info), download=True)

    video = ws.find_video()
    if video is None:
        raise StepError("视频下载失败：未找到输出文件")
    v = probe(video)
    meta["video"] = {"file": video.name, **v}
    ws.write_json("meta.json", meta)
    log.info("视频已下载：%s（%dx%d，%.1fs）", video.name, v.get("width", 0), v.get("height", 0), v["duration"])

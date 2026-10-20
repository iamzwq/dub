"""⑦ 生成带中文标题的封面（cover.zh.jpg）。"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..context import Context
from ..log import log

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
]


def fingerprint(cfg) -> dict:
    return {"font": cfg.cover.font_path}


def find_font(custom: str) -> str | None:
    for p in [custom, *_FONT_CANDIDATES]:
        if p and Path(p).exists():
            return p
    return None


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> list[str]:
    # 以"英文单词/单个汉字"为单位折行
    units: list[str] = []
    buf = ""
    for ch in text:
        if ch.isascii() and ch.isalnum():
            buf += ch
            continue
        if buf:
            units.append(buf)
            buf = ""
        units.append(ch)
    if buf:
        units.append(buf)
    lines: list[str] = []
    cur = ""
    for u in units:
        if cur and font.getlength(cur + u) > max_width:
            lines.append(cur.rstrip())
            cur = u.lstrip()
        else:
            cur += u
    if cur.strip():
        lines.append(cur.strip())
    return lines


def render_cover(src: Path, dst: Path, title: str, font_path: str, max_lines: int = 2) -> None:
    img = Image.open(src).convert("RGB")
    w, h = img.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    top = int(h * 0.40)
    for y in range(top, h):
        alpha = int(210 * ((y - top) / (h - top)) ** 1.2)
        od.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), overlay)

    max_width = w * 0.9
    size = int(h * 0.16)
    while size > 12:
        font = ImageFont.truetype(font_path, size)
        lines = wrap_text(title, font, max_width)
        if len(lines) <= max_lines:
            break
        size = int(size * 0.92)
    if len(lines) > 1:
        # 均衡各行长度：在不增加行数的前提下尽量收窄行宽
        width = font.getlength(title) / len(lines)
        while width < max_width:
            candidate = wrap_text(title, font, width)
            if len(candidate) == len(lines):
                lines = candidate
                break
            width += size * 0.5
    line_h = int(size * 1.2)
    stroke = max(2, size // 14)
    draw = ImageDraw.Draw(img)
    y = h - int(h * 0.07) - line_h * len(lines)
    for ln in lines:
        lw = font.getlength(ln)
        draw.text(((w - lw) / 2, y), ln, font=font, fill=(255, 255, 255), stroke_width=stroke, stroke_fill=(0, 0, 0))
        y += line_h
    img.convert("RGB").save(dst, quality=95)


def run(ctx: Context) -> None:
    ws, cfg = ctx.ws, ctx.cfg
    src = ws.path("cover.jpg")
    if not src.exists():
        log.warning("没有原始封面，跳过生成中文封面")
        return
    titles = ws.read_json("titles.json") or {}
    title = (titles.get("cover_title") or titles.get("zh_title") or "").strip()
    if not title:
        log.warning("没有中文标题，跳过生成中文封面")
        return
    font = find_font(cfg.cover.font_path)
    if not font:
        log.warning("未找到中文字体，请在 config.yaml 的 cover.font_path 中指定，跳过生成中文封面")
        return
    render_cover(src, ws.path("cover.zh.jpg"), title, font)
    log.info("中文封面：%s → cover.zh.jpg", title)

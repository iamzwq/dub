"""生成便于复制的投稿信息（原链接、发布日期、作者、简介、B 站 Topic）。"""
from __future__ import annotations

from .log import log
from .workspace import Workspace

FILENAME = "投稿信息.txt"


def format_date(raw: str | None) -> str:
    raw = raw or ""
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}" if len(raw) == 8 and raw.isdigit() else raw


def _section(name: str, body: str) -> str:
    return f"{name}：\n{body.strip() or '（无）'}"


def build_upload_info(meta: dict, titles: dict) -> str:
    sections = [
        "\n".join([
            f"原视频链接：{meta.get('webpage_url') or ''}",
            f"发布日期：{format_date(meta.get('upload_date'))}",
            f"原作者：{meta.get('uploader') or meta.get('channel') or ''}",
            f"原标题：{meta.get('title') or ''}",
        ]),
        _section("中文简介", titles.get("zh_description") or ""),
        _section("Topic", ", ".join(titles.get("topics") or [])),
    ]
    return "\n\n".join(sections) + "\n"


def write_upload_info(ws: Workspace) -> None:
    meta = ws.read_json("meta.json")
    titles = ws.read_json("titles.json")
    if not meta or not titles:
        return
    ws.path(FILENAME).write_text(build_upload_info(meta, titles), encoding="utf-8")
    if "topics" not in titles:
        log.warning("%s 中缺少 Topic 和中文简介（旧版本生成的摘要），需要时可用 --from translate 重新翻译生成", FILENAME)

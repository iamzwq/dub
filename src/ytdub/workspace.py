from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

STEPS = ["download", "segment", "translate", "tts", "align", "mix", "cover", "compose"]

STEP_TITLES = {
    "download": "下载视频/封面/字幕",
    "segment": "字幕解析与断句",
    "translate": "AI 翻译",
    "tts": "edge-tts 配音",
    "align": "时长对齐",
    "mix": "拼接配音轨与字幕",
    "cover": "生成中文封面",
    "compose": "ffmpeg 合成视频",
}

# 步骤的续跑缓存文件；--from 强制重跑时删除，避免复用旧结果
STEP_CACHES = {"translate": ["translate.cache.json"]}

_VIDEO_ID_PATTERNS = [
    r"(?:v=|/shorts/|/embed/|/live/|/v/|youtu\.be/)([A-Za-z0-9_-]{11})",
    r"^([A-Za-z0-9_-]{11})$",
]


def parse_video_id(url: str) -> str | None:
    for pattern in _VIDEO_ID_PATTERNS:
        m = re.search(pattern, url.strip())
        if m:
            return m.group(1)
    return None


def fingerprint(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def short_hash(*parts: Any) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:10]


class Workspace:
    def __init__(self, root: Path, video_id: str):
        self.video_id = video_id
        self.dir = root / video_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tts_dir.mkdir(exist_ok=True)

    # ---- 路径 ----
    def path(self, name: str) -> Path:
        return self.dir / name

    @property
    def tts_dir(self) -> Path:
        return self.dir / "tts"

    @property
    def state_file(self) -> Path:
        return self.dir / "state.json"

    # ---- JSON 读写 ----
    def read_json(self, name: str, default: Any = None) -> Any:
        p = self.path(name)
        if not p.exists():
            return default
        return json.loads(p.read_text(encoding="utf-8"))

    def write_json(self, name: str, data: Any) -> None:
        p = self.path(name)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)

    # ---- 步骤状态 ----
    def load_state(self) -> dict:
        return self.read_json("state.json", {"steps": {}})

    def is_done(self, step: str, fp: str) -> bool:
        info = self.load_state()["steps"].get(step)
        return bool(info and info.get("done") and info.get("fingerprint") == fp)

    def mark_done(self, step: str, fp: str, seconds: float) -> None:
        state = self.load_state()
        state["steps"][step] = {
            "done": True,
            "fingerprint": fp,
            "seconds": round(seconds, 1),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.write_json("state.json", state)

    def invalidate_from(self, step: str) -> None:
        state = self.load_state()
        for s in STEPS[STEPS.index(step):]:
            state["steps"].pop(s, None)
            for name in STEP_CACHES.get(s, []):
                self.path(name).unlink(missing_ok=True)
        self.write_json("state.json", state)

    # ---- 报告 ----
    def update_report(self, section: str, data: dict) -> None:
        report = self.read_json("report.json", {})
        report[section] = data
        self.write_json("report.json", report)

    def find_video(self) -> Path | None:
        for p in sorted(self.dir.glob("video.*")):
            if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
                return p
        return None

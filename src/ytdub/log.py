from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

console = Console()
log = logging.getLogger("ytdub")

_file_handler: logging.Handler | None = None


def _fix_stdio_encoding() -> None:
    """Windows 下输出被重定向时默认是 GBK 等本地编码，遇到 ✓ 等字符会报错。"""
    for stream in (sys.stdout, sys.stderr):
        if not hasattr(stream, "reconfigure"):
            continue
        try:
            if stream.isatty():
                stream.reconfigure(errors="replace")
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def setup_logging(verbose: bool = False) -> None:
    _fix_stdio_encoding()
    log.setLevel(logging.DEBUG)
    log.propagate = False
    handler = RichHandler(console=console, show_path=False, markup=False, rich_tracebacks=True)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    log.handlers = [handler]


def attach_file_log(path: Path) -> None:
    global _file_handler
    if _file_handler is not None:
        log.removeHandler(_file_handler)
    path.parent.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.FileHandler(path, encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    log.addHandler(_file_handler)


def count_progress() -> Progress:
    """按数量计数的进度条（句子、批次等）。"""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[info]}"),
        console=console,
    )


def bytes_progress() -> Progress:
    """按字节计数的下载进度条。"""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def percent_progress() -> Progress:
    """按百分比/时间的进度条（视频合成）。"""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[info]}"),
        console=console,
    )

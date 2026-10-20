from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from .config import load_config, parse_rate
from .context import StepError
from .llm import LLMError
from .log import console, log, setup_logging
from .workspace import STEPS

app = typer.Typer(add_completion=False, help="YouTube 视频 → AI 中文翻译 → edge-tts 配音 → 合成新视频")


def _check_step(value: Optional[str]) -> Optional[str]:
    if value and value not in STEPS:
        raise typer.BadParameter(f"可选步骤：{', '.join(STEPS)}")
    return value


@app.command()
def run(
    url: str = typer.Argument(..., help="YouTube 视频 URL 或 11 位视频 ID"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件，默认 ./config.yaml"),
    voice: Optional[str] = typer.Option(None, "--voice", help="edge-tts 音色，默认 zh-CN-YunyangNeural"),
    rate: Optional[str] = typer.Option(None, "--rate", help="基准语速，如 +5%"),
    segment: Optional[str] = typer.Option(None, "--segment", help="断句方式：auto | punct（按句末标点）| ai"),
    lang: Optional[str] = typer.Option(None, "--lang", help="手动指定源语言代码，如 en / ja"),
    cookies: Optional[str] = typer.Option(None, "--cookies", help="cookies.txt 路径"),
    from_step: Optional[str] = typer.Option(None, "--from", callback=_check_step, help=f"从指定步骤重跑：{'/'.join(STEPS)}"),
    to_step: Optional[str] = typer.Option(None, "--to", callback=_check_step, help="执行到指定步骤为止"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="终端显示调试日志"),
) -> None:
    """下载 → 断句 → 翻译 → 配音 → 对齐 → 拼接 → 封面 → 合成。"""
    setup_logging(verbose)
    try:
        cfg = load_config(config)
        if voice:
            cfg.tts.voice = voice
        if rate:
            cfg.tts.rate = parse_rate(rate)
        if segment:
            if segment not in ("auto", "punct", "ai"):
                raise typer.BadParameter("--segment 只能是 auto / punct / ai")
            cfg.segment.mode = segment
        if lang:
            cfg.source_lang = lang

        from .pipeline import run_pipeline

        code = run_pipeline(url, cfg, from_step, to_step, cookies)
    except (StepError, LLMError, ValueError, FileNotFoundError) as e:
        log.error("%s", e)
        raise typer.Exit(1)
    except KeyboardInterrupt:
        log.error("已中断（已完成的步骤和缓存会保留，重跑可继续）")
        raise typer.Exit(130)
    raise typer.Exit(code)


@app.command()
def voices(locale: str = typer.Option("zh-CN", "--locale", help="语言区域前缀，如 zh-CN / zh")) -> None:
    """列出 edge-tts 可用音色。"""
    import edge_tts

    setup_logging()
    items = asyncio.run(edge_tts.list_voices())
    table = Table(show_header=True, header_style="bold")
    for col in ("ShortName", "Gender", "Locale", "风格/个性"):
        table.add_column(col)
    for v in sorted(items, key=lambda x: x["ShortName"]):
        if not v["Locale"].startswith(locale):
            continue
        tag = v.get("VoiceTag") or {}
        traits = ", ".join((tag.get("VoicePersonalities") or []) + (tag.get("ContentCategories") or []))
        table.add_row(v["ShortName"], v["Gender"], v["Locale"], traits)
    console.print(table)


def main() -> None:  # pragma: no cover
    sys.exit(app())

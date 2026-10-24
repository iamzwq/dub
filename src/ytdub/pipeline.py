from __future__ import annotations

import time
from pathlib import Path

from rich.table import Table

from .config import Config
from .context import Context, StepError
from .log import attach_file_log, console, log
from .media import require_tools
from .steps import MODULES
from .workspace import STEP_TITLES, STEPS, Workspace, fingerprint, parse_video_id


def resolve_cookies(cfg: Config, override: str | None) -> Path | None:
    path = Path(override or cfg.download.cookies) if (override or cfg.download.cookies) else None
    if path and path.exists():
        return path
    if path:
        log.warning("cookies 文件不存在：%s（将不带 cookies 下载，可能被限流）", path)
    return None


def run_pipeline(url: str, cfg: Config, from_step: str | None, to_step: str | None, cookies: str | None) -> int:
    require_tools()
    video_id = parse_video_id(url)
    if not video_id:
        raise StepError(f"无法从 URL 中识别 YouTube 视频 ID：{url}")
    ws = Workspace(Path(cfg.workdir), video_id)
    attach_file_log(ws.path("run.log"))
    log.debug("==== 开始运行 %s ====", url)
    if from_step:
        ws.invalidate_from(from_step)

    last = STEPS.index(to_step) if to_step else len(STEPS) - 1
    ctx = Context(cfg=cfg, ws=ws, url=url)
    if not ws.is_done("download", fingerprint(MODULES["download"].fingerprint(cfg))):
        ctx.cookies = resolve_cookies(cfg, cookies)

    overview = Table(title=f"视频 {video_id} → {ws.dir}", show_header=True, header_style="bold")
    overview.add_column("#", justify="right")
    overview.add_column("步骤")
    overview.add_column("状态")
    plan: list[tuple[str, str, bool]] = []
    force = False
    for i, step in enumerate(STEPS):
        fp = fingerprint(MODULES[step].fingerprint(cfg))
        done = ws.is_done(step, fp) and not force
        if not done:
            force = True
        plan.append((step, fp, done))
        status = "[green]✓ 已完成（跳过）" if done else ("[yellow]待执行" if i <= last else "[dim]本次不执行")
        overview.add_row(str(i + 1), STEP_TITLES[step], status)
    console.print(overview)

    t_all = time.time()
    try:
        for i, (step, fp, done) in enumerate(plan):
            if i > last:
                break
            if done:
                continue
            console.rule(f"[bold cyan][{i + 1}/{len(STEPS)}] {STEP_TITLES[step]}")
            t0 = time.time()
            log.debug("步骤开始：%s", step)
            MODULES[step].run(ctx)
            dt = time.time() - t0
            ws.mark_done(step, fp, dt)
            log.info("✓ %s 完成，用时 %.1fs", STEP_TITLES[step], dt)
    finally:
        if ctx._llm is not None:
            ws.update_report("llm", ctx._llm.usage())
        ctx.close()

    print_summary(ws, time.time() - t_all, last)
    return 0


def print_summary(ws: Workspace, elapsed: float, last: int) -> None:
    state = ws.load_state()["steps"]
    report = ws.read_json("report.json", {})
    table = Table(title="运行汇总", show_header=True, header_style="bold")
    table.add_column("步骤")
    table.add_column("用时", justify="right")
    for step in STEPS[: last + 1]:
        info = state.get(step)
        table.add_row(STEP_TITLES[step], f"{info['seconds']}s" if info else "-")
    console.print(table)

    lines = [f"本次总耗时：{elapsed:.1f}s"]
    if seg := report.get("segment"):
        lines.append(f"断句：{seg['sentences']} 句（{'按句末标点' if seg['mode'] == 'punct' else 'AI'}）")
    if tr := report.get("llm"):
        lines.append(
            f"LLM（最近一次运行）：{tr['model']}，调用 {tr['calls']} 次，tokens 输入 {tr['prompt_tokens']} / 输出 {tr['completion_tokens']}"
        )
    if al := report.get("align"):
        lines.append(
            f"对齐：超时 {al['overflow_before']} 句 → 提速 {al['rate_adjusted']}、精简 {al['shortened']}、"
            f"atempo {al['atempo']}（最大 {al['max_tempo']:.2f}x）、顺延 {al['drifted']}（最大 {al['max_drift']:.2f}s）"
        )
    out = Path((report.get("compose") or {}).get("output") or ws.path("output.mp4"))
    if out.exists() and last == len(STEPS) - 1:
        lines.append(f"[bold green]输出视频：{out}")
    for ln in lines:
        console.print(ln)

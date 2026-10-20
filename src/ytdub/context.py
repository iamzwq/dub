from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .llm import LLM
from .workspace import Workspace


class StepError(RuntimeError):
    """可预期的流程错误：打印提示后中断。"""


@dataclass
class Context:
    cfg: Config
    ws: Workspace
    url: str
    cookies: Path | None = None
    _llm: LLM | None = field(default=None, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)

    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM(self.cfg)
        return self._llm

    def run_async(self, coro):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def close(self) -> None:
        if self._loop is not None:
            # Ctrl+C 中断时可能还有未完成的请求任务：先全部取消，避免它们在客户端关闭后继续重试
            pending = [t for t in asyncio.all_tasks(self._loop) if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            if self._llm is not None:
                self._loop.run_until_complete(self._llm.close())
            self._loop.close()
            self._loop = None

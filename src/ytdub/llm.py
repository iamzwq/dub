from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable
from urllib.parse import urlparse

from openai import AsyncOpenAI

from .config import Config
from .log import log


class LLMError(RuntimeError):
    pass


def extract_json(content: str) -> Any:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.S).strip()
    # 去掉部分推理模型输出的 <think> 块
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        i, j = content.find(open_ch), content.rfind(close_ch)
        if i != -1 and j > i:
            try:
                return json.loads(content[i: j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("返回内容不是合法 JSON")


def _is_local(base_url: str) -> bool:
    host = urlparse(base_url).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


class LLM:
    def __init__(self, cfg: Config):
        c = cfg.llm
        base_url, key, model = c.resolved_base_url(), c.resolved_api_key(), c.resolved_model()
        missing = [name for name, value in (("base_url", base_url), ("model", model)) if not value]
        if not key and not _is_local(base_url):
            missing.append("api_key")
        if missing:
            raise LLMError(
                f"LLM 未配置：缺少 {', '.join(missing)}。请在 config.yaml 的 llm 段，"
                f"或 .env 的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 中设置"
            )
        self.base_url = base_url
        self.model = model
        self.json_mode = c.json_mode
        self.extra_body = c.extra_body or None
        self.temperature = c.temperature
        self.max_retries = c.max_retries
        self.client = AsyncOpenAI(base_url=base_url, api_key=key or "none", timeout=c.timeout, max_retries=0)
        self.semaphore = asyncio.Semaphore(c.concurrency)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    async def chat_json(
        self,
        system: str,
        user: str,
        validate: Callable[[Any], Any] | None = None,
        label: str = "",
    ) -> Any:
        """调用 LLM 并解析 JSON；validate 抛出异常时视为失败并重试。"""
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.semaphore:
                    kwargs: dict[str, Any] = dict(
                        model=self.model,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        temperature=self.temperature,
                    )
                    if self.json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    if self.extra_body:
                        kwargs["extra_body"] = self.extra_body
                    resp = await self.client.chat.completions.create(**kwargs)
                self.calls += 1
                if resp.usage:
                    self.prompt_tokens += resp.usage.prompt_tokens or 0
                    self.completion_tokens += resp.usage.completion_tokens or 0
                content = resp.choices[0].message.content or ""
                data = extract_json(content)
                return validate(data) if validate else data
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.debug("LLM 调用失败 %s（第 %d 次）：%s", label, attempt, e)
                if attempt < self.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 20))
        raise LLMError(f"LLM 调用失败 {label}：{last_err}")

    def usage(self) -> dict:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    async def close(self) -> None:
        await self.client.close()

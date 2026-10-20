from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable
from urllib.parse import urlparse

import httpx
from openai import (
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
)

from .config import Config
from .log import log


class LLMError(RuntimeError):
    pass


class LLMFatalError(LLMError):
    """认证失败、模型不存在、参数不被支持等，重试或拆分批次都无济于事的错误。"""


_CONTENT_FILTER_HINTS = ("sensitive", "content_filter", "risk", "审核", "敏感")


def _short(e: Exception) -> str:
    text = str(e).replace("\n", " ")
    return text if len(text) <= 300 else text[:300] + "…"


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
        self.reasoning_effort = c.reasoning_effort.strip()
        self.extra_body = c.extra_body or None
        self.temperature = c.temperature
        self.max_retries = c.max_retries
        # 流式请求下 read 超时是"连续多久没收到数据"，而不是整个请求的总时长
        timeout = httpx.Timeout(c.timeout, connect=30.0)
        self.client = AsyncOpenAI(base_url=base_url, api_key=key or "none", timeout=timeout, max_retries=0)
        self.semaphore = asyncio.Semaphore(c.concurrency)
        self.stream_usage = True
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    async def _complete(self, kwargs: dict[str, Any]) -> str:
        """流式请求并拼接正文；推理模型的思考内容（reasoning_content）只用于保持连接活跃，不计入结果。"""
        extra: dict[str, Any] = {"stream": True}
        if self.stream_usage:
            extra["stream_options"] = {"include_usage": True}
        try:
            stream = await self.client.chat.completions.create(**kwargs, **extra)
        except BadRequestError as e:
            if self.stream_usage and "stream_options" in str(e):
                self.stream_usage = False
                return await self._complete(kwargs)
            raise
        parts: list[str] = []
        try:
            async for chunk in stream:
                if chunk.usage:
                    self.prompt_tokens += chunk.usage.prompt_tokens or 0
                    self.completion_tokens += chunk.usage.completion_tokens or 0
                for choice in chunk.choices or []:
                    if choice.delta and choice.delta.content:
                        parts.append(choice.delta.content)
        finally:
            await stream.close()
        return "".join(parts)

    async def chat_json(
        self,
        system: str,
        user: str,
        validate: Callable[[Any], Any] | None = None,
        label: str = "",
    ) -> Any:
        """调用 LLM 并解析 JSON；validate 抛出异常时视为失败并重试。"""
        last_err: Exception | None = None
        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            try:
                async with self.semaphore:
                    content = await self._complete(self._request_kwargs(system, user))
                self.calls += 1
                data = extract_json(content)
                return validate(data) if validate else data
            except (AuthenticationError, PermissionDeniedError, NotFoundError) as e:
                raise LLMFatalError(f"LLM 请求被拒绝（请检查 base_url / api_key / model）：{_short(e)}") from e
            except BadRequestError as e:
                if self._drop_unsupported(e):
                    attempt -= 1
                    continue
                if any(h in str(e).lower() for h in _CONTENT_FILTER_HINTS):
                    raise LLMError(f"LLM 拒绝了该内容 {label}：{_short(e)}") from e
                raise LLMFatalError(
                    f"LLM 请求参数错误（检查 model、json_mode、reasoning_effort、extra_body 配置）：{_short(e)}"
                ) from e
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 20)
                    log.warning("LLM 请求失败（%s，第 %d/%d 次）：%s，%ds 后重试",
                                label, attempt, self.max_retries, _short(e), wait)
                    await asyncio.sleep(wait)
        raise LLMError(f"LLM 调用失败 {label}：{_short(last_err) if last_err else ''}")

    def _request_kwargs(self, system: str, user: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
        )
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        return kwargs

    def _drop_unsupported(self, e: BadRequestError) -> bool:
        """接口明确拒绝 reasoning_effort 或 extra_body 中某个参数时，去掉该参数（本次运行内不再发送），返回是否需要重试。"""
        msg = str(e)
        dropped: list[str] = []
        if self.reasoning_effort and "reasoning_effort" in msg:
            self.reasoning_effort = ""
            dropped.append("reasoning_effort")
        for key in list(self.extra_body or {}):
            if key in msg and key != "reasoning_effort":
                self.extra_body = {k: v for k, v in self.extra_body.items() if k != key}
                dropped.append(f"extra_body.{key}")
        if dropped:
            log.info("该模型不支持参数 %s，已自动去掉后重试", "、".join(dropped))
        return bool(dropped)

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

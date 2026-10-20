import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from ytdub.config import Config
from ytdub.llm import LLM, LLMError, LLMFatalError


def _sse(chunks: list[dict]) -> bytes:
    lines = [f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode("utf-8")


def _chunk(delta: dict | None = None, usage: dict | None = None) -> dict:
    return {
        "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
        "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": None}],
        "usage": usage,
    }


def _make_llm(handler, monkeypatch, **llm_cfg) -> LLM:
    cfg = Config.model_validate({"llm": {"base_url": "http://localhost/v1", "model": "m", "max_retries": 2, **llm_cfg}})
    llm = LLM(cfg)
    llm.client = AsyncOpenAI(base_url="http://localhost/v1", api_key="k", max_retries=0,
                             http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def no_sleep(_):
        return None

    monkeypatch.setattr("ytdub.llm.asyncio.sleep", no_sleep)
    return llm


def test_streaming_ignores_reasoning_and_counts_usage(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.update(body)
        chunks = [
            _chunk({"role": "assistant", "reasoning_content": "让我想想……"}),
            _chunk({"content": '{"zh": "你'}),
            _chunk({"content": '好"}'}),
            _chunk(None, usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        ]
        return httpx.Response(200, content=_sse(chunks), headers={"content-type": "text/event-stream"})

    llm = _make_llm(handler, monkeypatch, reasoning_effort="minimal")
    assert asyncio.run(llm.chat_json("sys", "hi")) == {"zh": "你好"}
    assert seen["stream"] is True and seen["stream_options"] == {"include_usage": True}
    assert seen["reasoning_effort"] == "minimal"
    assert (llm.prompt_tokens, llm.completion_tokens, llm.calls) == (10, 5, 1)


def test_stream_options_unsupported_falls_back(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if "stream_options" in json.loads(request.content):
            return httpx.Response(400, json={"error": {"message": "unknown field stream_options"}})
        return httpx.Response(200, content=_sse([_chunk({"content": '{"a": 1}'})]),
                              headers={"content-type": "text/event-stream"})

    llm = _make_llm(handler, monkeypatch)
    assert asyncio.run(llm.chat_json("sys", "hi")) == {"a": 1}
    assert llm.stream_usage is False


@pytest.mark.parametrize("status", [400, 401, 404])
def test_fatal_errors_fail_fast(monkeypatch, status):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status, json={"error": {"message": "thinking.type disabled is not supported"}})

    llm = _make_llm(handler, monkeypatch)
    with pytest.raises(LLMFatalError):
        asyncio.run(llm.chat_json("sys", "hi"))
    assert len(calls) == 1


def test_content_filter_is_not_fatal(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"code": "SensitiveContentDetected", "message": "sensitive"}})

    llm = _make_llm(handler, monkeypatch)
    with pytest.raises(LLMError) as exc:
        asyncio.run(llm.chat_json("sys", "hi"))
    assert not isinstance(exc.value, LLMFatalError)


def test_transient_errors_are_retried(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(200, content=_sse([_chunk({"content": '{"ok": true}'})]),
                              headers={"content-type": "text/event-stream"})

    llm = _make_llm(handler, monkeypatch)
    assert asyncio.run(llm.chat_json("sys", "hi")) == {"ok": True}
    assert len(calls) == 2

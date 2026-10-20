import json

from ytdub.config import Config
from ytdub.context import Context
from ytdub.steps import translate
from ytdub.workspace import Workspace


class StubLLM:
    def __init__(self, tag: str):
        self.tag = tag
        self.model = tag
        self.calls = 0
        self.prompt_tokens = self.completion_tokens = 0

    async def chat_json(self, system, user, validate=None, label=""):
        self.calls += 1
        if "本地化编辑" in system:
            data = {"summary": "s", "terms": [], "zh_title": f"标题{self.tag}", "cover_title": f"封面{self.tag}"}
        else:
            payload = json.loads(user)
            data = {"translations": [{"id": s["id"], "zh": f"{self.tag}译文{s['id']}"} for s in payload["sentences"]]}
        return validate(data) if validate else data

    async def close(self):
        pass


def _setup(tmp_path):
    ws = Workspace(tmp_path, "VIDEO000001")
    ws.write_json("meta.json", {"title": "T", "language": "en", "duration": 20})
    ws.write_json("sentences.json", [
        {"id": 1, "start": 0, "end": 2, "text": "Hello."},
        {"id": 2, "start": 3, "end": 5, "text": "World."},
    ])
    return ws


def _run(ws, model: str, tmp_path) -> StubLLM:
    cfg = Config.model_validate({"llm": {"base_url": "http://localhost/v1", "model": model},
                                 "translate": {"glossary": str(tmp_path / "none.yaml")}})
    ctx = Context(cfg=cfg, ws=ws, url="VIDEO000001")
    stub = StubLLM(model)
    ctx._llm = stub
    try:
        translate.run(ctx)
    finally:
        ctx.close()
    return stub


def test_translate_cache_reuse_and_invalidation(tmp_path):
    ws = _setup(tmp_path)

    first = _run(ws, "A", tmp_path)
    assert first.calls == 2  # 摘要 + 1 批翻译
    assert [d["zh"] for d in ws.read_json("translation.json")] == ["A译文1", "A译文2"]

    # 配置不变（如中断后续跑）：完全复用缓存
    again = _run(ws, "A", tmp_path)
    assert again.calls == 0

    # 换模型：缓存失效，重新生成标题并翻译
    other = _run(ws, "B", tmp_path)
    assert other.calls == 2
    assert [d["zh"] for d in ws.read_json("translation.json")] == ["B译文1", "B译文2"]
    assert ws.read_json("titles.json")["cover_title"] == "封面B"

    # --from translate：删除缓存，强制重译
    ws.invalidate_from("translate")
    assert not ws.path(translate.CACHE).exists()
    forced = _run(ws, "B", tmp_path)
    assert forced.calls == 2

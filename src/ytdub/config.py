from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """任意 OpenAI 兼容接口。留空的字段从环境变量 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 读取。"""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    json_mode: bool = True
    # 关闭/减少推理模型的思考：翻译不需要长时间思考，可快数倍并大幅节省 token。
    # 接口明确不支持其中某个参数时会自动去掉；设为 "" / {} 则不发送。
    reasoning_effort: str = "minimal"  # minimal/low/medium/high
    extra_body: dict = Field(default_factory=lambda: {"thinking": {"type": "disabled"}})
    concurrency: int = 4
    temperature: float = 0.3
    max_retries: int = 3
    timeout: float = 180

    def resolved_base_url(self) -> str:
        return (self.base_url or os.environ.get("LLM_BASE_URL", "")).strip()

    def resolved_api_key(self) -> str:
        return (self.api_key or os.environ.get("LLM_API_KEY", "")).strip()

    def resolved_model(self) -> str:
        return (self.model or os.environ.get("LLM_MODEL", "")).strip()

    def identity(self) -> str:
        """用于判断翻译等步骤是否需要重跑。"""
        return f"{self.resolved_base_url()}|{self.resolved_model()}"


class DownloadConfig(BaseModel):
    cookies: str = "cookies.txt"
    max_height: int = 1080
    retries: int = 10
    sleep_requests: float = 0.0
    ratelimit: str = ""
    proxy: str = ""


class SegmentConfig(BaseModel):
    mode: str = "auto"  # auto | punct | ai
    max_gap_seconds: float = 3.0  # punct：超过该时长的静音也断开，0 关闭
    max_sentence_seconds: float = 15.0  # 仅 AI 断句使用
    ai_chunk_words: int = 250


class TranslateConfig(BaseModel):
    batch_size: int = 30
    context_size: int = 5
    chars_per_second: float = 4.5
    glossary: str = "glossary.yaml"


class TTSConfig(BaseModel):
    voice: str = "zh-CN-YunyangNeural"
    rate: int = 5
    max_rate: int = 30
    concurrency: int = 6
    retries: int = 3
    trim_silence_db: float = -45.0
    proxy: str = ""


class AlignConfig(BaseModel):
    shorten_rounds: int = 2
    overflow_tolerance: float = 0.1
    atempo_max: float = 1.15
    max_drift: float = 1.5
    drift_atempo_max: float = 1.25


class SubtitleConfig(BaseModel):
    font: str = "霞鹜文楷等宽"  # LXGW WenKai Mono：https://github.com/lxgw/LxgwWenKai
    font_scale: float = 0.055
    max_chars_per_line: int = 40  # 实际还受画面宽度限制：每行不超过 画面宽 × 90% ÷ 字号
    max_lines: int = 2  # 每屏最多行数：一句译文尽量整句一屏显示，超长时才分屏
    outline: float = 3.0
    box_color: str = "000000"  # 背景框颜色（RGB 十六进制）
    box_opacity: float = 0.90  # 背景框不透明度 0~1（0.9 = 透明度 10%），0 表示不加背景框
    margin_scale: float = 0.01  # 字幕距底部 = 画面高度 × 该比例
    strip_punctuation: bool = True


class CoverConfig(BaseModel):
    font_path: str = ""


class ComposeConfig(BaseModel):
    encoder: str = "auto"  # auto | h264_nvenc | libx264
    nvenc_cq: int = 23
    nvenc_preset: str = "p5"
    x264_crf: int = 21
    x264_preset: str = "medium"
    audio_bitrate: str = "192k"
    loudnorm: str = "I=-16:TP=-1.5:LRA=11"


class Config(BaseModel):
    workdir: str = "work"
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    align: AlignConfig = Field(default_factory=AlignConfig)
    subtitle: SubtitleConfig = Field(default_factory=SubtitleConfig)
    cover: CoverConfig = Field(default_factory=CoverConfig)
    compose: ComposeConfig = Field(default_factory=ComposeConfig)

    # 运行时选项（来自命令行）
    source_lang: str = ""


def load_config(path: Path | None) -> Config:
    load_dotenv()
    data: dict = {}
    if path is None:
        candidate = Path("config.yaml")
        path = candidate if candidate.exists() else None
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config.model_validate(data)


def parse_rate(value: str | int) -> int:
    """'+10%' / '10' / 10 -> 10"""
    if isinstance(value, int):
        return value
    return int(str(value).strip().rstrip("%"))

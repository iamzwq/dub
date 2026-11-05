# coding=utf-8
"""Smoke test for DashScope cloud TTS. Requires DASHSCOPE_API_KEY env var."""
import os
import sys
import time

import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer

dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
dashscope.base_websocket_api_url = "wss://ws-47hq6nx4t38rk982.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference"

model = "qwen-audio-3.1-tts-flash"
voice = "longanhuan_v3.1"
text = sys.argv[1] if len(sys.argv) > 1 else "How is the weather today?"
out = sys.argv[2] if len(sys.argv) > 2 else "output.mp3"

t0 = time.time()
synthesizer = SpeechSynthesizer(model=model, voice=voice)
audio = synthesizer.call(text)
elapsed = time.time() - t0

if not audio:
    print(f"no audio returned; request_id={synthesizer.get_last_request_id()}")
    sys.exit(1)

with open(out, "wb") as f:
    f.write(audio)
print(f"ok: {len(audio)} bytes -> {out} in {elapsed:.2f}s; "
      f"request_id={synthesizer.get_last_request_id()}, first_pkg_delay={synthesizer.get_first_package_delay()}ms")

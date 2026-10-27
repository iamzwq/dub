from ytdub.llm import extract_json
from ytdub.steps.download import choose_subtitle
from ytdub.steps.mix import _ass_time, subtitle_events
from ytdub.steps.segment import align_sentences
from ytdub.subtitles import find_sentence_breaks, parse_json3, parse_vtt, segment_punct
from ytdub.models import Token
from ytdub.text import has_punctuation, is_abbreviation_dot, split_display_lines, split_units, strip_display_punct, zh_len
from ytdub.timeline import Clip, schedule
from ytdub.workspace import parse_video_id

import pytest


# ---------------------------------------------------------------- 解析
AUTO_JSON3 = {
    "events": [
        {"tStartMs": 0, "dDurationMs": 5000, "id": 1, "wWinId": 1},
        {"tStartMs": 100, "dDurationMs": 3000, "wWinId": 1, "segs": [
            {"utf8": "hello", "acAsrConf": 0}, {"utf8": " everyone", "tOffsetMs": 400}, {"utf8": " today", "tOffsetMs": 900}]},
        {"tStartMs": 2000, "dDurationMs": 10, "wWinId": 1, "aAppend": 1, "segs": [{"utf8": "\n"}]},
        {"tStartMs": 2010, "dDurationMs": 3000, "wWinId": 1, "segs": [
            {"utf8": "we", "acAsrConf": 0}, {"utf8": " talk", "tOffsetMs": 300}, {"utf8": " [Music]", "tOffsetMs": 600}]},
    ]
}


def test_parse_json3_auto_word_level():
    toks = parse_json3(AUTO_JSON3)
    assert [t.text for t in toks] == ["hello", "everyone", "today", "we", "talk"]
    assert toks[1].start == pytest.approx(0.5)
    assert toks[2].end <= toks[3].start + 1e-9  # 重叠被裁掉
    assert all(t.end >= t.start for t in toks)


def test_parse_json3_manual_cue_level():
    data = {"events": [
        {"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "Hello there.\nHow are"}]},
        {"tStartMs": 2000, "dDurationMs": 2000, "segs": [{"utf8": "you today?"}]},
    ]}
    toks = parse_json3(data)
    assert [t.text for t in toks] == ["Hello there. How are", "you today?"]


def test_parse_vtt():
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.500\n<c>Hi</c> there.\n\n00:00:02.500 --> 00:00:04.000\nBye!\n"
    toks = parse_vtt(vtt)
    assert [(t.text, t.start, t.end) for t in toks] == [("Hi there.", 1.0, 2.5), ("Bye!", 2.5, 4.0)]


# ---------------------------------------------------------------- 文本
def test_abbreviation_dot():
    assert not is_abbreviation_dot("It works.", 8)
    assert is_abbreviation_dot("Talk to Mr.", 10)
    assert is_abbreviation_dot("John F.", 6)
    assert is_abbreviation_dot('("e.g.', 5)


def test_has_punctuation():
    words = " ".join(["word"] * 200)
    assert not has_punctuation(words, "en")
    assert has_punctuation(". ".join(["some words here"] * 50) + ".", "en")
    assert has_punctuation("short text.", "en")
    assert not has_punctuation("short text", "en")


def test_zh_len_and_display_split():
    assert zh_len("你好世界") == 4
    assert zh_len("使用GPT模型") == 4 + 2
    lines = split_display_lines("今天我们来聊一聊大语言模型的训练方法，以及它在实际工作中的各种应用场景。", 20)
    assert all(len(x) <= 20 for x in lines)
    assert "".join(lines) == "今天我们来聊一聊大语言模型的训练方法，以及它在实际工作中的各种应用场景。"
    assert strip_display_punct("你好，世界。") == "你好 世界"


# ---------------------------------------------------------------- 断句
def _tokens(spec):
    return [Token(t, s, e) for t, s, e in spec]


def test_punct_segment_splits_cues_and_merges():
    toks = _tokens([("Hello there. How are", 0, 2), ("you today? I am", 2, 4), ("fine.", 4, 5)])
    sents = segment_punct(toks, "en")
    assert [s.text for s in sents] == ["Hello there.", "How are you today?", "I am fine."]
    assert sents[0].start == 0 and sents[-1].end == 5
    assert [s.id for s in sents] == [1, 2, 3]


def test_punct_segment_keeps_long_sentences_whole():
    toks = [Token(f"w{i}" + ("," if i % 5 == 4 else ""), i * 0.5, i * 0.5 + 0.4) for i in range(40)]
    toks[-1].text += "."
    sents = segment_punct(toks, "en")
    assert len(sents) == 1  # 20 秒的长句、含逗号也不切
    assert sents[0].end - sents[0].start > 19


def test_punct_segment_abbreviations_and_quotes():
    toks = _tokens([('Mr. Smith said "yes." Then', 0, 3), ("he left.", 3, 4)])
    sents = segment_punct(toks, "en")
    assert [s.text for s in sents] == ['Mr. Smith said "yes."', "Then he left."]


def test_punct_segment_long_gap_guard():
    toks = _tokens([("la la la", 0, 2), ("Welcome back.", 10, 11)])
    assert len(segment_punct(toks, "en", max_gap=3)) == 2
    assert len(segment_punct(toks, "en", max_gap=0)) == 1


def _split(text, lang="en"):
    breaks = find_sentence_breaks(text, lang)
    out, prev = [], 0
    for b in breaks:
        if text[prev:b].strip():
            out.append(text[prev:b].strip())
        prev = b
    return out


@pytest.mark.parametrize("text, expected", [
    # 句末标点后的闭合引号归入本句
    ('He said "yes." Then he left.', ['He said "yes."', "Then he left."]),
    # 引号闭合后是小写 → 句子延续
    ('"Why?" he asked. "Because."', ['"Why?" he asked.', '"Because."']),
    # 引号内多句话保持在一起
    ('She said "I am leaving. Do not follow me." Then she left.',
     ['She said "I am leaving. Do not follow me."', "Then she left."]),
    # 弯引号与嵌套单引号
    ("He wrote “she told me ‘go.’” Next one.", ["He wrote “she told me ‘go.’”", "Next one."]),
    # 句末标点在引号外
    ('He called it "magic". Really!', ['He called it "magic".', "Really!"]),
    # 括号
    ("It failed (again.) We retried.", ["It failed (again.)", "We retried."]),
    # 缩写、小数、网址、省略号
    ("Mr. Smith paid 3.5 dollars at example.com today... Wow!",
     ["Mr. Smith paid 3.5 dollars at example.com today...", "Wow!"]),
    # 撇号不影响
    ("Don't stop. It's fine.", ["Don't stop.", "It's fine."]),
])
def test_sentence_breaks_quotes_en(text, expected):
    assert _split(text) == expected


@pytest.mark.parametrize("text, lang, expected", [
    ("他说：“我走了。你别跟着我。”然后离开了。今天下雨。", "zh", ["他说：“我走了。你别跟着我。”然后离开了。", "今天下雨。"]),
    ("“你好。”“你好。”", "zh", ["“你好。”", "“你好。”"]),
    ("「行くよ。」と彼は言った。そうか。", "ja", ["「行くよ。」と彼は言った。", "そうか。"]),
    ("价格是3.5元。好的。", "zh", ["价格是3.5元。", "好的。"]),
])
def test_sentence_breaks_quotes_cjk(text, lang, expected):
    assert _split(text, lang) == expected


def test_unclosed_quote_falls_back_after_limit():
    # 引号不配对：超过 QUOTE_MAX_SECONDS 后恢复在句末标点处断句
    words = ['He said "this'] + [f"w{i}." for i in range(40)]
    toks = [Token(w, i * 1.0, i * 1.0 + 0.9) for i, w in enumerate(words)]
    sents = segment_punct(toks, "en")
    assert len(sents) > 1
    assert all(s.end - s.start <= 17 for s in sents)


# ---------------------------------------------------------------- 数字（千分位/小数）
@pytest.mark.parametrize("text, lang, expected", [
    ("It cost 1,234,567.89 dollars. Next.", "en", ["It cost 1,234,567.89 dollars.", "Next."]),
    ("In Germany it is 1.234.567,89 Euro. Next.", "de", ["In Germany it is 1.234.567,89 Euro.", "Next."]),
    ("It grew 3.5 percent to 1,000. Then it fell.", "en", ["It grew 3.5 percent to 1,000.", "Then it fell."]),
    ("收入是1,234,567.89美元。下一句。", "zh", ["收入是1,234,567.89美元。", "下一句。"]),
])
def test_sentence_breaks_numbers(text, lang, expected):
    assert _split(text, lang) == expected


def test_punctuation_detection_ignores_numbers():
    no_punct = " ".join(["revenue grew to 1.000.000 and 2.500.000 then 3.5 or 4.75 percent at example.com"] * 6)
    assert not has_punctuation(no_punct, "en")
    with_punct = " ".join(["Revenue grew to 1.000.000 and then 3.5 percent."] * 6)
    assert has_punctuation(with_punct, "en")


@pytest.mark.parametrize("text", [
    "公司去年的总收入达到了12,345,678美元，同比增长了百分之十五",
    "今年我们一共卖出了大约1,234,567台设备真的很多很多",
    "德国的价格大约是1.234.567,89欧元这个数字非常大的",
])
def test_display_lines_keep_numbers_whole(text):
    lines = split_display_lines(text, 20)
    assert "".join(lines) == text
    import re as _re
    numbers = _re.findall(r"\d[\d,.]*\d", text)
    for num in numbers:
        assert any(num in ln for ln in lines), (num, lines)


def test_zh_len_numbers():
    assert zh_len("1,000,000") == zh_len("1000000")
    assert zh_len("3.5") == zh_len("35")


def test_segment_punct_times_follow_chars():
    toks = _tokens([("Hi. How are you?", 0, 4)])
    sents = segment_punct(toks, "en")
    assert [s.text for s in sents] == ["Hi.", "How are you?"]
    assert sents[0].start == 0 and sents[0].end == pytest.approx(0.75)
    assert sents[1].start == pytest.approx(1.0) and sents[1].end == 4


def test_align_sentences_maps_back_to_tokens():
    words = "so today we are going to talk about models they are very useful".split()
    units = [split_units(w, "en") for w in words]
    out = ["So today we are going to talk about models.", "They are very useful."]
    res = align_sentences(units, out, "en")
    assert res == [(0, out[0]), (9, out[1])]


def test_align_sentences_rejects_rewritten_text():
    units = [split_units(w, "en") for w in "the quick brown fox jumps".split()]
    with pytest.raises(ValueError):
        align_sentences(units, ["Something completely different here."], "en")


# ---------------------------------------------------------------- 时间线
def test_schedule_no_overflow():
    ps = schedule([Clip(1, 0, 3, 2.0), Clip(2, 3, 6, 2.5)])
    assert [p.play_start for p in ps] == [0, 3]
    assert all(p.tempo == 1.0 for p in ps)


def test_schedule_atempo_then_drift():
    ps = schedule([Clip(1, 0, 2, 2.6), Clip(2, 2, 6, 1.0)], atempo_max=1.15, max_drift=1.5)
    assert ps[0].tempo == pytest.approx(1.15, abs=1e-3)
    assert ps[1].play_start == pytest.approx(2.6 / 1.15, abs=1e-3)
    assert ps[1].drift > 0


def test_schedule_respects_drift_cap():
    ps = schedule([Clip(1, 0, 1, 5.0), Clip(2, 1, 2, 0.5)], atempo_max=1.15, max_drift=1.5, drift_atempo_max=1.25)
    assert ps[0].tempo == pytest.approx(1.25)
    # 压到上限仍放不下：下一句不重叠，顺延超过上限
    assert ps[1].play_start >= ps[0].play_end - 1e-6


def test_schedule_zero_drift_last_clip():
    ps = schedule([Clip(1, 0, 2, 2.2, max_drift=0)], atempo_max=1.15)
    assert ps[0].play_end <= 2 + 1e-3


def test_schedule_small_overflow_tolerated():
    ps = schedule([Clip(1, 0, 2, 2.05), Clip(2, 2, 4, 1)], tolerance=0.1)
    assert ps[0].tempo == 1.0
    assert ps[1].play_start == pytest.approx(2.05)


# ---------------------------------------------------------------- 其他
def test_choose_subtitle_prefers_manual():
    info = {
        "language": "en",
        "subtitles": {"en-US": [{"ext": "vtt", "url": "m"}, {"ext": "json3", "url": "mj"}], "live_chat": [{"ext": "json", "url": "x"}]},
        "automatic_captions": {"en-orig": [{"ext": "json3", "url": "a"}], "zh-Hans": [{"ext": "json3", "url": "t"}]},
    }
    assert choose_subtitle(info) == ("en", "manual", "en-US", {"ext": "json3", "url": "mj"})


def test_choose_subtitle_auto_orig_and_missing():
    info = {"subtitles": {"zh-Hans": [{"ext": "json3", "url": "z"}]},
            "automatic_captions": {"ja-orig": [{"ext": "json3", "url": "a"}], "zh-Hans": [{"ext": "json3", "url": "t"}]}}
    lang, kind, key, fmt = choose_subtitle(info)
    assert (lang, kind, key) == ("ja", "auto", "ja-orig")
    from ytdub.context import StepError
    with pytest.raises(StepError):
        choose_subtitle({"language": "en", "subtitles": {}, "automatic_captions": {}})


def test_parse_video_id():
    assert parse_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=1") == "dQw4w9WgXcQ"
    assert parse_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert parse_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert parse_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert parse_video_id("https://example.com") is None


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('<think>hmm</think>{"a": 2}') == {"a": 2}
    assert extract_json('Here: {"a": 3} done') == {"a": 3}


def test_llm_config_env_fallback_and_override(monkeypatch):
    from ytdub.config import Config

    monkeypatch.setenv("LLM_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    monkeypatch.setenv("LLM_MODEL", "env-model")
    cfg = Config.model_validate({"llm": {"model": "file-model"}})
    assert cfg.llm.resolved_base_url() == "https://env.example/v1"
    assert cfg.llm.resolved_api_key() == "env-key"
    assert cfg.llm.resolved_model() == "file-model"
    assert cfg.llm.identity() == "https://env.example/v1|file-model"


def test_llm_missing_settings(monkeypatch):
    from ytdub.config import Config
    from ytdub.llm import LLM, LLMError

    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(LLMError, match="base_url, model, api_key"):
        LLM(Config())
    # 本地服务不强制 api_key
    LLM(Config.model_validate({"llm": {"base_url": "http://localhost:11434/v1", "model": "m"}}))
    with pytest.raises(LLMError, match="api_key"):
        LLM(Config.model_validate({"llm": {"base_url": "https://remote.example/v1", "model": "m"}}))


def test_subtitle_events_follow_play_time():
    tl = [
        {"zh": "第一句话。", "play_start": 1.0, "play_end": 2.0, "start": 0.8, "end": 2, "duration": 1.0},
        {"zh": "第二句。", "play_start": 2.2, "play_end": 3.0, "start": 2.2, "end": 3, "duration": 0.8},
    ]
    ev = subtitle_events(tl, 20, True)
    assert ev[0] == (1.0, 2.2, "第一句话")
    assert ev[1][2] == "第二句"
    assert _ass_time(3725.456) == "1:02:05.46"


def test_ass_box_style():
    from ytdub.config import SubtitleConfig
    from ytdub.steps.mix import ass_color, build_ass

    assert ass_color("000000", 0.1) == "&HE6000000"
    assert ass_color("000000", 0.9) == "&H19000000"
    assert ass_color("#FF8000", 1.0) == "&H000080FF"
    ass = build_ass([(0.0, 1.0, "你好")], 1280, 720, SubtitleConfig())
    box = next(ln for ln in ass.splitlines() if ln.startswith("Style: Box"))
    assert box.split(",")[3] == "&H19000000"
    dialogues = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogues) == 2 and r"\p1}m " in dialogues[0] and dialogues[1].endswith("你好")
    ass = build_ass([(0.0, 1.0, "你好")], 1280, 720, SubtitleConfig(box_opacity=0))
    assert len([ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]) == 1


def test_display_lines_avoid_short_first_line():
    text = "FreeBSD 的内核和基本系统在同一个仓库里开发，版本号是14.2"
    lines = split_display_lines(text, 20)
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")
    assert all(len(ln) >= 10 for ln in lines[:-1]), lines


# ---------------------------------------------------------------- 字幕分屏
from ytdub.text import balanced_split, display_width, split_screens  # noqa: E402


def test_short_sentence_one_screen_two_balanced_lines():
    screens = split_screens("FreeBSD 的内核和基本系统是在同一个仓库里一起开发的。", 20, 2)
    assert len(screens) == 1 and len(screens[0]) == 2
    a, b = (display_width(x) for x in screens[0])
    assert abs(a - b) <= 4
    assert "".join(screens[0]).startswith("FreeBSD") and "同一个" in "".join(screens[0])


def test_split_prefers_punctuation_and_keeps_words():
    text = "Linux 发行版则不同，它们把内核和来自许多不同项目的用户态工具组合在一起，所以升级方式和文档风格差别很大。"
    screens = split_screens(text, 20, 2)
    assert len(screens) == 2
    assert screens[1][0].startswith("所以")
    joined = ["".join(s) for s in screens]
    assert "".join(joined).replace(" ", "") == text.replace(" ", "")
    for s in screens:
        assert all(display_width(ln) <= 20 for ln in s)


def test_no_line_starts_with_punct_or_particle():
    text = "这就是为什么很多工程师在生产环境里更喜欢用 ZFS 和 jail 来做隔离，而不是容器。"
    for screen in split_screens(text, 20, 2):
        for ln in screen:
            assert ln[0] not in "，。、；：里的了"


def test_ascii_counts_half_width():
    assert display_width("ABCD") == 2 and display_width("中文") == 2
    # 全英文 40 个半角字符 = 20 宽，一行放得下
    assert balanced_split("a" * 40, 20) == ["a" * 40]


def test_ass_multiline_escape():
    from ytdub.steps.mix import _ass_escape
    assert _ass_escape("第一行\n第二行{x}") == "第一行\\N第二行｛x｝"

def test_punct_split_single_lines():
    from ytdub.text import punct_split

    text = "FreeBSD 的内核和基本系统在一起开发，版本号会同步发布。"
    pieces = punct_split(text, 20)
    assert pieces == ["FreeBSD 的内核和基本系统在一起开发，", "版本号会同步发布。"]
    assert punct_split("FreeBSD 的内核和基本系统是在同一个仓库里一起开发的。", 20) is None  # 无标点断点
    assert punct_split(text, 20, min_width=12) is None  # 第二段太短
    assert punct_split("苹果、香蕉、橘子、葡萄、西瓜、草莓、芒果、菠萝、樱桃、柠檬、桃子", 20) is None  # 不在顿号处分屏


def test_split_screens_prefer_single_line():
    text = "FreeBSD 的内核和基本系统在一起开发，版本号会同步发布。"
    default = split_screens(text, 20, 2)
    assert len(default) == 1 and len(default[0]) == 2
    assert split_screens(text, 20, 2, min_split_width=0) == [
        ["FreeBSD 的内核和基本系统在一起开发，"], ["版本号会同步发布。"]
    ]


def test_subtitle_events_prefer_single_line():
    text = "FreeBSD 的内核和基本系统在一起开发，版本号会同步发布。"
    tl = [{"zh": text, "play_start": 0.0, "play_end": 6.0, "start": 0.0, "end": 6.0, "duration": 6.0}]
    ev = subtitle_events(tl, 20, True, 2, prefer_single_line=True)
    assert [e[2] for e in ev] == ["FreeBSD 的内核和基本系统在一起开发", "版本号会同步发布"]
    assert ev[0][0] == 0.0 and abs(ev[-1][1] - 6.5) < 1e-6 and ev[0][1] == ev[1][0]
    # 关闭、语速过快、每屏太短时仍一屏两行
    assert len(subtitle_events(tl, 20, True, 2, prefer_single_line=False)) == 1
    fast = [dict(tl[0], play_end=2.0, end=2.0, duration=2.0)]
    assert len(subtitle_events(fast, 20, True, 2, prefer_single_line=True)) == 1
    assert len(subtitle_events(tl, 20, True, 2, prefer_single_line=True, min_screen_duration=3.0)) == 1


def test_safe_filename_and_output_name(tmp_path):
    from ytdub.steps.compose import build_command, output_name, safe_filename
    from ytdub.config import Config
    from ytdub.workspace import Workspace

    assert safe_filename('AI 会取代程序员吗？: "真相" / 未来') == 'AI 会取代程序员吗？ 真相 未来'
    assert safe_filename("--标题. ") == "标题"
    assert safe_filename("CON") == ""
    assert safe_filename(" ?*") == ""
    assert len(safe_filename("长" * 200)) == 80

    ws = Workspace(tmp_path, "abcdefghijk")
    assert output_name(ws) == "output.mp4"
    ws.write_json("titles.json", {"zh_title": "中文标题：测试"})
    assert output_name(ws) == "中文标题：测试.mp4"
    ws.write_json("titles.json", {"zh_title": "a|b"})
    assert output_name(ws) == "a b.mp4"
    assert build_command("video.mp4", "libx264", Config(), None, "a b.mp4")[-1] == "a b.mp4"

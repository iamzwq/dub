# ytdub — YouTube 视频中文配音工具

下载 YouTube 视频 / 封面 / 字幕 → AI 断句与翻译（上下文 + 术语表 + 批量）→ edge-tts 中文配音 → 时长对齐 → ffmpeg 烧录中文字幕并合成新视频。

## 流程

| # | 步骤 | 产物（`work/<video_id>/`） |
|---|---|---|
| 1 | 下载：yt-dlp + cookies，优先手动字幕，无则自动字幕，都没有则中断 | `meta.json` `video.mp4` `cover.jpg` `subs.src.json3` |
| 2 | 断句：字幕有标点 → 只按句末标点断句（保留完整句子）；无标点 → AI 断句 | `sentences.json` |
| 3 | 翻译：摘要 + 自动术语 + 用户术语表，带上下文批量翻译，按时长限制字数 | `titles.json` `glossary.json` `translation.json` |
| 4 | 配音：edge-tts 并发合成（默认 `zh-CN-YunyangNeural`） | `tts/*.wav` `tts.json` |
| 5 | 对齐：提语速 → AI 精简译文 → atempo 微调 → 有限顺延（画面保持原速） | `timeline.json` |
| 6 | 拼接：配音轨 + 中文字幕 | `dub.wav` `zh.ass` |
| 7 | 封面：原封面 + 中文标题 | `cover.zh.jpg` |
| 8 | 合成：烧录硬字幕、替换音轨、嵌入封面（优先 h264_nvenc） | `output.mp4` |

每一步完成后记录到 `state.json`，重跑时自动跳过；修改了相关配置（音色、模型、术语表等）会自动从受影响的步骤重跑。翻译和配音结果有缓存，中断后重跑可以接着做。

## 安装

以下以 Windows（PowerShell）为例，新电脑上按顺序执行即可。

**1. 安装 uv 和 ffmpeg**（只需一次；装完后重新打开终端）

```powershell
winget install astral-sh.uv
winget install Gyan.FFmpeg        # full build，自带 libass（烧录字幕）和 NVENC（显卡编码）
```

也可以用 scoop：`scoop install uv ffmpeg`。不需要单独安装 Python：uv 会按 `.python-version` 自动下载 Python 3.12。

可选：安装 [deno](https://deno.com/)（`winget install DenoLand.Deno`）。yt-dlp 会自动检测并用它解析 YouTube 的 JS 验证，部分视频能获得更完整的格式；所需的 `yt-dlp-ejs` 脚本已随 `yt-dlp[default]` 依赖安装。不装 deno 也能用，只会提示一次警告并下载可用的最佳格式。

**2. 获取项目代码**

复制整个项目目录（不需要 `.venv`、`work`），或从 git 仓库 clone。注意 `uv.lock` 要一起复制，它锁定了所有依赖的版本。

**3. 安装 Python 依赖**

```powershell
cd D:\code\dub
uv sync
```

**4. 准备配置和私密文件**（这些文件不在版本库里，需要从旧电脑复制或重新创建）

```powershell
copy .env.example .env                    # 填入 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
copy config.example.yaml config.yaml      # 可选
copy glossary.example.yaml glossary.yaml  # 可选
```

cookies：用浏览器扩展（如 "Get cookies.txt LOCALLY"）在登录 YouTube 后导出 Netscape 格式的 `cookies.txt` 放到项目根目录。

**5. 验证**

```powershell
ffmpeg -hide_banner -encoders | findstr "h264_nvenc libx264"   # 能看到编码器即可
uv run pytest -q                                              # 测试应全部通过
uv run ytdub voices                                           # 能列出音色说明 edge-tts 网络正常
```

没有 NVIDIA 显卡或驱动时，合成步骤会自动改用 libx264（CPU 编码），无需额外设置。

LLM 支持任意 OpenAI 兼容接口，只需三项配置：`base_url`、`api_key`、`model`。可写在 `.env`（`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`），也可写在 `config.yaml` 的 `llm` 段（优先）。本地服务（localhost）可不填 api_key。若接口不支持 JSON 模式，设置 `llm.json_mode: false`。

## 使用

```powershell
uv run ytdub run "https://www.youtube.com/watch?v=VIDEO_ID"
uv run ytdub voices
```

也可以写成 `uv run python -m ytdub ...`。查看帮助：`uv run ytdub --help`、`uv run ytdub run --help`。

### `ytdub run <URL>` — 执行完整流程

`<URL>`：YouTube 视频地址（支持 `watch?v=`、`youtu.be/`、`/shorts/`、`/embed/`、`/live/`），或直接写 11 位视频 ID。仅支持单个视频，不支持播放列表和直播。

| 参数 | 说明 | 默认值 / 对应配置 |
|---|---|---|
| `-c, --config PATH` | 指定配置文件 | `./config.yaml`（不存在则全部用默认值） |
| `--cookies PATH` | cookies.txt 路径（Netscape 格式），用于防限流 | `download.cookies` = `cookies.txt` |
| `--lang CODE` | 手动指定源语言代码（如 `en`、`ja`、`de`），自动识别失败或想用其他语言字幕时使用 | 自动识别 |
| `--segment MODE` | 断句方式：`auto`（有标点 → `punct`，无标点 → `ai`）、`punct`（只按句末标点断句）、`ai`（LLM 补标点并断句） | `segment.mode` = `auto` |
| `--voice NAME` | edge-tts 音色，可用 `ytdub voices` 查看 | `tts.voice` = `zh-CN-YunyangNeural` |
| `--rate RATE` | 基准语速，如 `+10%`、`10`、`-5%` | `tts.rate` = `+5%` |
| `--from STEP` | 从指定步骤开始强制重跑（该步骤及之后的步骤都会重新执行） | — |
| `--to STEP` | 执行到指定步骤为止 | 执行到最后一步 |
| `-v, --verbose` | 终端显示调试日志（`run.log` 始终记录完整日志） | 关闭 |

命令行参数优先于 `config.yaml`。下载分辨率（`download.max_height`）、LLM 模型（`llm.model`）、视频编码器（`compose.encoder`）等其他设置请在 `config.yaml` 中修改。

`STEP` 可选值（按执行顺序）：

| STEP | 步骤 |
|---|---|
| `download` | 下载视频 / 封面 / 字幕 |
| `segment` | 字幕解析与断句 |
| `translate` | AI 翻译 |
| `tts` | edge-tts 配音 |
| `align` | 时长对齐 |
| `mix` | 拼接配音轨与字幕 |
| `cover` | 生成中文封面 |
| `compose` | ffmpeg 合成视频 |

不加 `--from` 时，已完成且配置未变的步骤会自动跳过，中断后重跑会接着做（例如已翻译的句子不会重复翻译）。如果修改了会影响结果的设置（如 `--voice`、`--rate`、`--segment`、`config.yaml` 中的模型或术语表），程序会自动从受影响的步骤开始重跑，并丢弃该步骤的旧结果。`--from` 则无条件从指定步骤重做。

### `ytdub voices` — 列出 edge-tts 音色

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--locale PREFIX` | 按语言区域前缀筛选，如 `zh-CN`、`zh`（含粤语、台湾等） | `zh-CN` |

### 示例

```powershell
# 完整流程
uv run ytdub run "https://www.youtube.com/watch?v=VIDEO_ID"

# 使用指定的配置文件和 cookies
uv run ytdub run VIDEO_ID -c my.yaml --cookies D:\cookies\youtube.txt

# 先只做到翻译，人工检查 / 修改 work\VIDEO_ID\translation.json
uv run ytdub run VIDEO_ID --to translate
# 改完后从配音继续（不会重新翻译）
uv run ytdub run VIDEO_ID --from tts

# 换音色、调快语速（只会从配音步骤开始重跑）
uv run ytdub run VIDEO_ID --voice zh-CN-YunxiNeural --rate +10%

# 重新翻译（丢弃已有译文）；在 config.yaml 中换了模型时会自动检测并重译，无需此参数
uv run ytdub run VIDEO_ID --from translate

# 字幕语言识别错误时手动指定；强制使用 AI 断句
uv run ytdub run VIDEO_ID --lang en --segment ai

# 只下载视频和字幕
uv run ytdub run VIDEO_ID --to download

# 查看所有中文音色（含粤语、台湾）
uv run ytdub voices --locale zh
```

### 输出

所有文件在 `work/<video_id>/` 下（目录可通过 `workdir` 配置）：

- `output.mp4`：成品视频
- `run.log`：完整日志
- `report.json`：断句、LLM 用量、时长对齐等统计
- 每个步骤的中间产物见上方「流程」表

## 语速与时长对齐

中文配音往往比原文长，按以下顺序处理，画面始终原速：

1. 翻译时按每句可用时长 × `chars_per_second`（默认 4.5 字/秒）限制字数；
2. 超时则按比例提高 edge-tts 语速（最高 `max_rate` = +30%）；
3. 仍超时让 LLM 精简译文（最多 `shorten_rounds` 轮）；
4. 用 atempo 压缩（≤ `atempo_max` = 1.15）；
5. 有限顺延：允许本句延后结束、下一句顺延开始，延迟在后续停顿中消化；累计不超过 `max_drift`（1.5s），超出部分继续压缩（≤ `drift_atempo_max` = 1.25）。`max_drift: 0` 即关闭顺延。

字幕时间跟随配音实际播放时间。

## 测试

```powershell
uv run pytest
```

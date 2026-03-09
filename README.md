# yt-agent

**Open-source YouTube intelligence tool for AI agents.**

Extract metadata, transcripts, and key frames from any YouTube video — via CLI, Python library, or MCP server.

[![CI](https://github.com/kvksatish/yt-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/kvksatish/yt-agent/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/yt-agent)](https://pypi.org/project/yt-agent/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

- **Metadata** — title, channel, duration, views, tags, chapters via yt-dlp
- **Transcripts** — auto/manual subtitles with timestamps; multi-language; translate-to
- **Whisper fallback** — local transcription via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) when captions aren't available
- **Frame extraction** — interval, scene-change, or keyframe modes via ffmpeg
- **VLM descriptions** — optional vision-LLM descriptions of frames (ollama / OpenAI-compatible)
- **Batch processing** — playlists, channels, or URL files with parallel workers
- **File cache** — 30-day cache in `~/.cache/yt-agent/` by video ID; `--no-cache` to bypass
- **MCP server** — expose `extract`, `transcript`, and `frames` as MCP tools for AI agents
- **Python API** — `from yt_agent import extract` for programmatic access
- **Pydantic schemas** — fully typed `VideoAnalysis`, `VideoMetadata`, `TranscriptSegment`, `FrameInfo`

---

## Installation

```bash
pip install yt-agent
```

With Whisper support:

```bash
pip install "yt-agent[whisper]"
```

With MCP server:

```bash
pip install "yt-agent[mcp]"
```

> **Requirements:** Python 3.12+, `ffmpeg` in PATH (for frame extraction), `yt-dlp` is bundled.

---

## Quick Start

### CLI

```bash
# Extract metadata + transcript
yt-agent "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# JSON output to a directory
yt-agent "https://youtu.be/dQw4w9WgXcQ" --output ./out --format json

# Extract frames every 30s
yt-agent "https://youtu.be/dQw4w9WgXcQ" --frames --frames-interval 30

# Scene-change frames + VLM descriptions (local ollama)
yt-agent "https://youtu.be/dQw4w9WgXcQ" --frames --frames-mode scene --vlm

# Whisper transcription (no internet captions needed)
yt-agent "https://youtu.be/dQw4w9WgXcQ" --whisper --whisper-model small

# French transcript, translated to English
yt-agent "https://youtu.be/dQw4w9WgXcQ" --language fr --translate-to en

# List available transcript languages
yt-agent languages "https://youtu.be/dQw4w9WgXcQ"
```

### Batch Processing

```bash
# Process an entire playlist (4 parallel workers)
yt-agent batch --playlist "https://www.youtube.com/playlist?list=PLxxxxxx" --workers 4

# Process a channel
yt-agent batch --channel "https://www.youtube.com/@channelname" --output ./channel-out

# Process URLs from a file
yt-agent batch --batch urls.txt --workers 8 --output ./batch-out
```

### Python API

```python
from yt_agent import extract, ExtractOptions

result = extract(
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    ExtractOptions(languages=["en"]),
)

print(result.metadata.title)             # "Rick Astley - Never Gonna Give You Up"
print(result.metadata.duration_seconds)  # 212
print(result.transcript[0].text)         # "We're no strangers to love..."

for seg in result.transcript:
    print(f"{seg.start:.1f}s: {seg.text}")
```

With frames and VLM:

```python
from yt_agent import extract, ExtractOptions

result = extract(
    "https://youtu.be/dQw4w9WgXcQ",
    ExtractOptions(
        extract_frames=True,
        frames_mode="scene",
        vlm=True,
        vlm_backend="ollama",
        vlm_model="llava",
    ),
)

for frame in result.frames:
    print(f"{frame.timestamp_seconds:.1f}s: {frame.path}")
    if frame.description:
        print(f"  -> {frame.description}")
```

### MCP Server

Add to Claude Desktop or any MCP-compatible client:

```json
{
  "mcpServers": {
    "yt-agent": {
      "command": "yt-agent-mcp"
    }
  }
}
```

Exposed tools: `extract_video`, `get_transcript`, `get_frames`.

---

## Output Structure

```
yt-agent-output/
├── metadata.json      # VideoMetadata (title, channel, tags, chapters, ...)
├── transcript.md      # Timestamped transcript segments
└── frames/
    ├── frame_0000.jpg
    ├── frame_0000.txt  # VLM description (if --vlm)
    └── ...
```

---

## Schemas

```python
class VideoMetadata(BaseModel):
    video_id: str
    url: str
    title: str
    channel: str
    duration_seconds: int
    upload_date: datetime | None
    view_count: int | None
    description: str
    tags: list[str]
    chapters: list[Chapter]

class TranscriptSegment(BaseModel):
    start: float        # seconds
    duration: float
    text: str

class FrameInfo(BaseModel):
    path: Path
    timestamp_seconds: float
    description: str | None  # set by VLM

class VideoAnalysis(BaseModel):
    metadata: VideoMetadata
    transcript: list[TranscriptSegment]
    frames: list[FrameInfo]
```

---

## Development

```bash
git clone https://github.com/kvksatish/yt-agent.git
cd yt-agent

# Install with uv (recommended)
uv sync --all-extras --dev

# Run tests
uv run pytest

# Lint
uv run ruff check src/
```

### Publishing (maintainers)

Tag a release to trigger the PyPI publish workflow:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `publish.yml` workflow uses PyPI's OIDC trusted publisher — no API token needed.

---

## License

MIT — see [LICENSE](LICENSE).

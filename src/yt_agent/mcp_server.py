"""MCP server for yt-agent using FastMCP.

Exposes three tools:
  - extract:    Full pipeline — metadata + transcript + optional frames
  - transcript: Fetch transcript only (fast, no video download)
  - frames:     Extract frames only (downloads video)

Run as a standalone server:
    python -m yt_agent.mcp_server

Or register in Claude Desktop / any MCP host:
    {
      "mcpServers": {
        "yt-agent": {
          "command": "uvx",
          "args": ["yt-agent[mcp]", "mcp"]
        }
      }
    }
"""

from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        "mcp is required for the MCP server: pip install yt-agent[mcp]"
    ) from exc

from yt_agent.api import ExtractOptions, extract as _extract
from yt_agent.schemas import FrameInfo, TranscriptSegment, VideoAnalysis, VideoMetadata

mcp = FastMCP(
    name="yt-agent",
    instructions=(
        "YouTube intelligence tool. Use `extract` for full analysis, "
        "`transcript` for subtitles only, or `frames` to get key video frames."
    ),
)


# ---------------------------------------------------------------------------
# Tool: extract
# ---------------------------------------------------------------------------

@mcp.tool()
def extract(
    url: str,
    language: str = "auto",
    translate_to: str = "",
    use_whisper: bool = False,
    whisper_model: str = "base",
    extract_frames: bool = False,
    frames_mode: str = "interval",
    frames_interval: float = 30.0,
    frames_threshold: float = 0.3,
    no_cache: bool = False,
) -> dict:
    """Extract full intelligence from a YouTube video.

    Returns metadata, timestamped transcript, and optionally key frames.

    Args:
        url: YouTube video URL.
        language: Preferred transcript language BCP-47 code, or "auto" (default).
        translate_to: Translate transcript to this language (BCP-47). Empty = no translation.
        use_whisper: Use Whisper (faster-whisper) for transcription instead of YouTube subtitles.
        whisper_model: Whisper model size: tiny, base (default), small, medium, large-v3.
        extract_frames: Extract video frames via ffmpeg (slow — downloads full video).
        frames_mode: Frame extraction mode: interval, scene, or keyframe.
        frames_interval: Seconds between frames in interval mode (default 30).
        frames_threshold: Scene change threshold 0.0-1.0 (default 0.3).
        no_cache: Bypass file cache and always re-fetch.
    """
    opts = ExtractOptions(
        languages=[language] if language and language != "auto" else None,
        translate_to=translate_to or None,
        use_whisper=use_whisper,
        whisper_model=whisper_model,
        extract_frames=extract_frames,
        frames_mode=frames_mode,
        frames_interval=frames_interval,
        frames_threshold=frames_threshold,
        no_cache=no_cache,
    )
    result: VideoAnalysis = _extract(url, opts)
    return _analysis_to_dict(result)


# ---------------------------------------------------------------------------
# Tool: transcript
# ---------------------------------------------------------------------------

@mcp.tool()
def transcript(
    url: str,
    language: str = "auto",
    translate_to: str = "",
    use_whisper: bool = False,
    whisper_model: str = "base",
    no_cache: bool = False,
) -> dict:
    """Fetch the transcript for a YouTube video.

    Returns timestamped segments and detected language. Fast — no video download
    unless use_whisper=True.

    Args:
        url: YouTube video URL.
        language: Preferred language BCP-47 code, or "auto".
        translate_to: Translate to this language (BCP-47). Empty = no translation.
        use_whisper: Use Whisper for transcription (downloads audio).
        whisper_model: Whisper model size (tiny/base/small/medium/large-v3).
        no_cache: Bypass cache.
    """
    from yt_agent.cache import load_transcript, store_transcript
    from yt_agent.transcript import fetch_transcript
    import re

    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    video_id = m.group(1) if m else url

    segments: list[TranscriptSegment] = []
    lang = "en"

    if use_whisper:
        from yt_agent.whisper_transcribe import transcribe as whisper_fn
        segments, lang = whisper_fn(url, model_name=whisper_model)
        store_transcript(video_id, [s.model_dump() for s in segments], lang)
    else:
        if not no_cache:
            cached = load_transcript(video_id)
            if cached:
                raw_segs, lang = cached
                segments = [TranscriptSegment.model_validate(s) for s in raw_segs]

        if not segments:
            preferred = [language] if language and language != "auto" else None
            segments, lang = fetch_transcript(url, languages=preferred, translate_to=translate_to or None)
            store_transcript(video_id, [s.model_dump() for s in segments], lang)

    return {
        "language": lang,
        "segment_count": len(segments),
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in segments
        ],
    }


# ---------------------------------------------------------------------------
# Tool: frames
# ---------------------------------------------------------------------------

@mcp.tool()
def frames(
    url: str,
    mode: str = "interval",
    interval: float = 30.0,
    scene_threshold: float = 0.3,
) -> dict:
    """Extract key frames from a YouTube video using ffmpeg.

    Downloads the video — use sparingly. Returns frame timestamps and file paths.

    Args:
        url: YouTube video URL.
        mode: Extraction mode: interval (default), scene, or keyframe.
        interval: Seconds between frames for interval mode (default 30).
        scene_threshold: Scene change sensitivity 0.0-1.0 for scene mode (default 0.3).
    """
    from pathlib import Path
    import re
    from yt_agent.frames import extract_frames

    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    video_id = m.group(1) if m else url
    output_dir = Path("yt-agent-output") / video_id

    frame_list: list[FrameInfo] = extract_frames(
        url, output_dir,
        mode=mode,
        interval=interval,
        scene_threshold=scene_threshold,
    )

    return {
        "frame_count": len(frame_list),
        "output_dir": str(output_dir / "frames"),
        "frames": [
            {"timestamp": f.timestamp, "path": f.path}
            for f in frame_list
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _analysis_to_dict(result: VideoAnalysis) -> dict:
    meta = result.metadata
    return {
        "metadata": {
            "video_id": meta.video_id,
            "title": meta.title,
            "channel": meta.channel,
            "duration_seconds": meta.duration_seconds,
            "view_count": meta.view_count,
            "upload_date": meta.upload_date.isoformat() if meta.upload_date else None,
            "url": str(meta.url),
            "tags": meta.tags[:20],
            "chapters": [
                {"title": c.title, "start_time": c.start_time, "end_time": c.end_time}
                for c in meta.chapters
            ],
        },
        "transcript": {
            "segment_count": len(result.transcript),
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in result.transcript
            ],
        },
        "frames": {
            "frame_count": len(result.frames),
            "frames": [
                {"timestamp": f.timestamp, "path": f.path}
                for f in result.frames
            ],
        },
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

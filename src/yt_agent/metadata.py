"""Metadata extraction from YouTube videos via yt-dlp."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from yt_agent.schemas import Chapter, VideoMetadata

try:
    import yt_dlp
except ImportError as exc:
    raise ImportError("yt-dlp is required: pip install yt-dlp") from exc


_YDL_OPTS: dict[str, Any] = {
    "quiet": True,
    "no_warnings": True,
    "extract_flat": False,
    "skip_download": True,
}


def _parse_chapters(raw: list[dict[str, Any]]) -> list[Chapter]:
    chapters: list[Chapter] = []
    for i, ch in enumerate(raw):
        start = float(ch.get("start_time", 0))
        # end_time may be absent on the last chapter
        raw_end = ch.get("end_time")
        end: float | None = float(raw_end) if raw_end is not None else None
        chapters.append(Chapter(title=ch.get("title", ""), start_time=start, end_time=end))
    return chapters


def extract_metadata(url: str) -> VideoMetadata:
    """Extract video metadata from a YouTube URL using yt-dlp.

    Args:
        url: YouTube video URL.

    Returns:
        Populated VideoMetadata instance.

    Raises:
        ValueError: If the URL is not a valid YouTube video.
        RuntimeError: If yt-dlp fails to extract info.
    """
    with yt_dlp.YoutubeDL(_YDL_OPTS) as ydl:
        try:
            info: dict[str, Any] = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise RuntimeError(f"yt-dlp extraction failed: {exc}") from exc

    if info.get("_type") == "playlist":
        raise ValueError("URL points to a playlist; provide a single video URL.")

    video_id: str = info.get("id", "")
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    raw_date = info.get("upload_date")  # YYYYMMDD string or None

    return VideoMetadata(
        video_id=video_id,
        url=canonical_url,
        title=info.get("title", ""),
        channel=info.get("uploader") or info.get("channel", ""),
        channel_id=info.get("channel_id") or info.get("uploader_id"),
        duration_seconds=int(info.get("duration", 0)),
        upload_date=_parse_date(raw_date),
        view_count=info.get("view_count"),
        description=info.get("description", ""),
        thumbnail_url=info.get("thumbnail"),
        tags=info.get("tags") or [],
        chapters=_parse_chapters(info.get("chapters") or []),
    )


def _parse_date(raw: str | None):
    """Convert YYYYMMDD string to datetime, or return None."""
    if not raw:
        return None
    from datetime import datetime
    try:
        return datetime.strptime(raw, "%Y%m%d")
    except ValueError:
        return None


def save_metadata(meta: VideoMetadata, output_dir: Path) -> Path:
    """Serialise VideoMetadata to <output_dir>/metadata.json.

    Args:
        meta: VideoMetadata instance.
        output_dir: Directory to write into (created if absent).

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "metadata.json"
    dest.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    return dest

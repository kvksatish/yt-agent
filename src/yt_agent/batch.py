"""Batch URL resolution for playlists and channels."""

from __future__ import annotations

from typing import Any

try:
    import yt_dlp
except ImportError as exc:
    raise ImportError("yt-dlp is required: pip install yt-dlp") from exc


_FLAT_OPTS: dict[str, Any] = {
    "quiet": True,
    "no_warnings": True,
    "extract_flat": "in_playlist",
    "skip_download": True,
}


def resolve_playlist(url: str) -> list[str]:
    """Return canonical watch URLs for all entries in a playlist or channel."""
    with yt_dlp.YoutubeDL(_FLAT_OPTS) as ydl:
        try:
            info: dict[str, Any] = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise RuntimeError(f"yt-dlp failed: {exc}") from exc

    entries = info.get("entries") or []
    urls: list[str] = []
    for entry in entries:
        vid = entry.get("id") or entry.get("url", "")
        if vid and not vid.startswith("http"):
            urls.append(f"https://www.youtube.com/watch?v={vid}")
        elif vid:
            urls.append(vid)
    return urls


def read_batch_file(path: str) -> list[str]:
    """Read newline-separated URLs from a file, skipping blank lines and comments."""
    from pathlib import Path
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]

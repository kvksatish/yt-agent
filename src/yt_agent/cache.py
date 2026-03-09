"""File-based caching for yt-agent.

Cache layout:
    ~/.cache/yt-agent/<video_id>/metadata.json
    ~/.cache/yt-agent/<video_id>/transcript.json
    ~/.cache/yt-agent/<video_id>/frames/        (future)

Each cache entry stores a `_cached_at` ISO timestamp in its JSON.
Entries older than TTL_DAYS are considered stale and re-fetched.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

TTL_DAYS: int = 30
_CACHE_ROOT = Path.home() / ".cache" / "yt-agent"

# Keys written alongside cached data
_CACHED_AT_KEY = "_cached_at"


def _cache_dir(video_id: str) -> Path:
    return _CACHE_ROOT / video_id


def _is_fresh(cached_at_iso: str | None, ttl_days: int = TTL_DAYS) -> bool:
    """Return True if the timestamp is within ttl_days of now."""
    if cached_at_iso is None:
        return False
    try:
        ts = datetime.fromisoformat(cached_at_iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return datetime.now(tz=timezone.utc) - ts < timedelta(days=ttl_days)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Metadata cache
# ---------------------------------------------------------------------------

def load_metadata(video_id: str, ttl_days: int = TTL_DAYS) -> dict[str, Any] | None:
    """Return cached metadata dict if fresh, else None."""
    path = _cache_dir(video_id) / "metadata.json"
    if not path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not _is_fresh(data.get(_CACHED_AT_KEY), ttl_days):
        return None
    return data


def store_metadata(video_id: str, data: dict[str, Any]) -> None:
    """Write metadata dict to cache, stamping _cached_at."""
    path = _cache_dir(video_id) / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**data, _CACHED_AT_KEY: datetime.now(tz=timezone.utc).isoformat()}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Transcript cache
# ---------------------------------------------------------------------------

def load_transcript(
    video_id: str, ttl_days: int = TTL_DAYS
) -> tuple[list[dict[str, Any]], str] | None:
    """Return (segments_list, language_code) if cached and fresh, else None."""
    path = _cache_dir(video_id) / "transcript.json"
    if not path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not _is_fresh(data.get(_CACHED_AT_KEY), ttl_days):
        return None
    return data.get("segments", []), data.get("language", "en")


def store_transcript(
    video_id: str, segments: list[dict[str, Any]], language: str
) -> None:
    """Write transcript to cache."""
    path = _cache_dir(video_id) / "transcript.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "language": language,
        "segments": segments,
        _CACHED_AT_KEY: datetime.now(tz=timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def invalidate(video_id: str) -> None:
    """Delete all cached files for a video ID."""
    import shutil

    d = _cache_dir(video_id)
    if d.exists():
        shutil.rmtree(d)


def cache_info(video_id: str) -> dict[str, Any]:
    """Return a summary of what is cached for a given video ID."""
    d = _cache_dir(video_id)
    info: dict[str, Any] = {"video_id": video_id, "cache_dir": str(d), "entries": {}}
    for name in ("metadata.json", "transcript.json"):
        p = d / name
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                cached_at = raw.get(_CACHED_AT_KEY)
                fresh = _is_fresh(cached_at)
            except (json.JSONDecodeError, OSError):
                cached_at = None
                fresh = False
            info["entries"][name] = {"cached_at": cached_at, "fresh": fresh}
    return info

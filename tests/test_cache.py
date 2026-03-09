"""Tests for file-based caching — no network required."""

import pytest

from yt_agent.cache import (
    invalidate,
    load_metadata,
    load_transcript,
    store_metadata,
    store_transcript,
)


VIDEO_ID = "_test_video_abc123"


@pytest.fixture(autouse=True)
def cleanup():
    """Remove test cache entries before and after each test."""
    invalidate(VIDEO_ID)
    yield
    invalidate(VIDEO_ID)


def test_store_and_load_metadata():
    data = {"title": "Test Video", "video_id": VIDEO_ID}
    store_metadata(VIDEO_ID, data)
    loaded = load_metadata(VIDEO_ID)
    assert loaded is not None
    assert loaded["title"] == "Test Video"


def test_load_metadata_missing():
    assert load_metadata(VIDEO_ID) is None


def test_store_and_load_transcript():
    segs = [{"start": 0.0, "duration": 2.0, "text": "Hello"}]
    store_transcript(VIDEO_ID, segs, "en")
    result = load_transcript(VIDEO_ID)
    assert result is not None
    raw_segs, lang = result
    assert lang == "en"
    assert raw_segs[0]["text"] == "Hello"


def test_load_transcript_missing():
    assert load_transcript(VIDEO_ID) is None


def test_invalidate():
    store_metadata(VIDEO_ID, {"title": "x"})
    invalidate(VIDEO_ID)
    assert load_metadata(VIDEO_ID) is None

"""Smoke tests for Pydantic schemas — no network required."""

from datetime import datetime

import pytest

from yt_agent.schemas import (
    Chapter,
    FrameInfo,
    TranscriptSegment,
    VideoAnalysis,
    VideoMetadata,
)


def _make_meta(**kwargs) -> VideoMetadata:
    defaults = dict(
        video_id="dQw4w9WgXcQ",
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Never Gonna Give You Up",
        channel="Rick Astley",
        channel_id="UCuAXFkgsw1L7xaCfnd5JJOw",
        duration_seconds=212,
        upload_date=datetime(1987, 7, 27),
        view_count=1_000_000_000,
        description="",
        tags=["pop", "80s"],
        chapters=[],
    )
    defaults.update(kwargs)
    return VideoMetadata(**defaults)


def test_video_metadata_roundtrip():
    meta = _make_meta()
    dumped = meta.model_dump(mode="json")
    restored = VideoMetadata.model_validate(dumped)
    assert restored.video_id == meta.video_id
    assert restored.duration_seconds == 212


def test_video_metadata_optional_fields():
    meta = _make_meta(upload_date=None, view_count=None)
    assert meta.upload_date is None
    assert meta.view_count is None


def test_chapter():
    ch = Chapter(title="Intro", start_time=0.0, end_time=30.0)
    assert ch.title == "Intro"
    assert ch.end_time == 30.0


def test_chapter_no_end():
    ch = Chapter(title="Outro", start_time=200.0)
    assert ch.end_time is None


def test_transcript_segment():
    seg = TranscriptSegment(start=1.5, end=4.5, text="Hello world")
    assert seg.start == pytest.approx(1.5)
    assert seg.end == pytest.approx(4.5)
    assert seg.text == "Hello world"
    assert seg.confidence is None


def test_transcript_segment_with_confidence():
    seg = TranscriptSegment(start=0.0, end=2.0, text="Test", confidence=0.95)
    assert seg.confidence == pytest.approx(0.95)


def test_frame_info():
    fi = FrameInfo(timestamp=10.0, path="/tmp/frame.jpg", width=1280, height=720)
    assert fi.timestamp == pytest.approx(10.0)
    assert fi.scene_label is None


def test_video_analysis_empty():
    meta = _make_meta()
    analysis = VideoAnalysis(metadata=meta, transcript=[], frames=[])
    assert analysis.metadata.title == "Never Gonna Give You Up"
    assert analysis.transcript == []
    assert analysis.frames == []
    assert analysis.summary is None

"""Unit tests for frame extraction helpers — no network, no ffmpeg."""
from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import pytest

from yt_agent.frames import _jpeg_dimensions
from yt_agent.schemas import FrameInfo


def _make_jpeg(width: int, height: int, dest: Path) -> Path:
    """Write a minimal valid JPEG SOF0 segment to dest for dimension parsing."""
    sof0 = struct.pack(">BBHBHH", 0xFF, 0xC0, 11, 8, height, width)
    # Minimal JPEG: SOI + SOF0 + EOI
    data = b"\xff\xd8" + sof0 + b"\xff\xd9"
    dest.write_bytes(data)
    return dest


def test_jpeg_dimensions_standard():
    with tempfile.TemporaryDirectory() as d:
        p = _make_jpeg(1920, 1080, Path(d) / "frame.jpg")
        w, h = _jpeg_dimensions(p)
    assert w == 1920
    assert h == 1080


def test_jpeg_dimensions_small():
    with tempfile.TemporaryDirectory() as d:
        p = _make_jpeg(320, 240, Path(d) / "frame.jpg")
        w, h = _jpeg_dimensions(p)
    assert w == 320
    assert h == 240


def test_jpeg_dimensions_not_jpeg():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "notajpeg.jpg"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG magic
        w, h = _jpeg_dimensions(p)
    assert w == 0
    assert h == 0


def test_jpeg_dimensions_empty():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "empty.jpg"
        p.write_bytes(b"")
        w, h = _jpeg_dimensions(p)
    assert w == 0
    assert h == 0


def test_jpeg_dimensions_missing():
    w, h = _jpeg_dimensions(Path("/tmp/does-not-exist-xyz.jpg"))
    assert w == 0
    assert h == 0


def test_frame_info_schema():
    """FrameInfo returned by extract_frames has correct field types."""
    fi = FrameInfo(
        timestamp=5.0,
        path="frames/frame_000005000.jpg",
        width=1280,
        height=720,
        scene_label=None,
    )
    assert fi.timestamp == 5.0
    assert fi.width == 1280
    assert fi.height == 720
    assert fi.path == "frames/frame_000005000.jpg"
    assert fi.scene_label is None
    # Roundtrip
    restored = FrameInfo.model_validate(fi.model_dump())
    assert restored == fi

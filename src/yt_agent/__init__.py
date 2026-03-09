"""yt-agent: Open-source YouTube intelligence tool for AI agents."""

__version__ = "0.1.0"

from yt_agent.api import ExtractOptions, extract
from yt_agent.schemas import (
    Chapter,
    FrameInfo,
    TranscriptSegment,
    VideoAnalysis,
    VideoMetadata,
)

__all__ = [
    "__version__",
    "extract",
    "ExtractOptions",
    "VideoAnalysis",
    "VideoMetadata",
    "TranscriptSegment",
    "FrameInfo",
    "Chapter",
]

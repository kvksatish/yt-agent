"""Pydantic output schemas for yt-agent."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class Chapter(BaseModel):
    """A named chapter within a YouTube video."""

    title: str = Field(description="Chapter title.")
    start_time: float = Field(description="Chapter start time in seconds.")
    end_time: float | None = Field(default=None, description="Chapter end time in seconds.")


class VideoMetadata(BaseModel):
    """Core metadata extracted from a YouTube video."""

    video_id: str = Field(description="YouTube video ID.")
    title: str = Field(description="Video title.")
    channel: str = Field(description="Channel name.")
    channel_id: str = Field(description="YouTube channel ID.")
    duration_seconds: int = Field(description="Video duration in seconds.")
    upload_date: datetime | None = Field(
        default=None, description="Video upload date."
    )
    view_count: int | None = Field(default=None, description="View count at fetch time.")
    description: str = Field(default="", description="Video description text.")
    url: HttpUrl = Field(description="Canonical video URL.")
    thumbnail_url: HttpUrl | None = Field(
        default=None, description="Thumbnail image URL."
    )
    tags: list[str] = Field(default_factory=list, description="Video tags.")
    chapters: list[Chapter] = Field(default_factory=list, description="Video chapters.")


class TranscriptSegment(BaseModel):
    """A single timed segment of transcript text."""

    start: float = Field(description="Start time in seconds.")
    end: float = Field(description="End time in seconds.")
    text: str = Field(description="Transcript text for this segment.")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Transcription confidence score (0-1).",
    )


class FrameInfo(BaseModel):
    """Metadata for a single extracted video frame."""

    timestamp: float = Field(description="Frame timestamp in seconds.")
    path: str = Field(description="File path to the extracted frame image.")
    width: int = Field(description="Frame width in pixels.")
    height: int = Field(description="Frame height in pixels.")
    scene_label: str | None = Field(
        default=None, description="Optional scene or content label."
    )


class VideoAnalysis(BaseModel):
    """Complete analysis output for a processed YouTube video."""

    metadata: VideoMetadata = Field(description="Video metadata.")
    transcript: list[TranscriptSegment] = Field(
        default_factory=list, description="Transcript segments."
    )
    frames: list[FrameInfo] = Field(
        default_factory=list, description="Extracted frame information."
    )
    summary: str | None = Field(
        default=None, description="AI-generated summary of the video."
    )
    processed_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp of processing."
    )

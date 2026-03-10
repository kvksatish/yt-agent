"""High-level Python API for yt-agent.

Programmatic entry point:

    from yt_agent import extract

    result = extract("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    print(result.metadata.title)
    for seg in result.transcript:
        print(seg.start, seg.text)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from yt_agent.schemas import FrameInfo, TranscriptSegment, VideoAnalysis, VideoMetadata


@dataclass
class ExtractOptions:
    """Options for the extract() function.

    All fields are optional; sane defaults are used when omitted.
    """

    # Transcript options
    languages: list[str] | None = None
    """Preferred transcript languages (BCP-47 list). None = auto-detect."""

    translate_to: str | None = None
    """Translate transcript to this language via YouTube's translation API."""

    use_whisper: bool = False
    """Use Whisper (faster-whisper) instead of youtube-transcript-api."""

    whisper_model: str = "base"
    """Whisper model size: tiny, base, small, medium, large-v3."""

    # Frame options
    extract_frames: bool = False
    """Extract video frames via ffmpeg."""

    frames_mode: str = "interval"
    """Frame extraction mode: interval, scene, or keyframe."""

    frames_interval: float = 30.0
    """Seconds between frames (interval mode)."""

    frames_threshold: float = 0.3
    """Scene change threshold 0.0-1.0 (scene mode)."""

    # VLM options (off by default — requires --frames to produce images first)
    vlm: bool = False
    """Run a vision LLM on each extracted frame to generate descriptions."""

    vlm_backend: str = "ollama"
    """VLM backend: 'ollama' (local) or 'openai' (OpenAI-compatible API)."""

    vlm_model: str | None = None
    """VLM model name. Defaults: ollama→llava, openai→gpt-4o."""

    vlm_api_base: str | None = None
    """Override API base URL for the VLM backend."""

    vlm_api_key: str = ""
    """API key for openai-compatible VLM backend."""

    vlm_prompt: str = ""
    """Custom prompt for the VLM. Empty = use default."""

    # Output / cache
    output_dir: Path | None = None
    """Directory to write artifacts. Defaults to yt-agent-output/<video_id>/."""

    no_cache: bool = False
    """Bypass the file cache; always re-fetch."""


def extract(
    url: str,
    options: ExtractOptions | None = None,
) -> VideoAnalysis:
    """Extract metadata, transcript, and optionally frames from a YouTube video.

    This is the primary programmatic entry point for yt-agent.

    Args:
        url: YouTube video URL (watch, youtu.be, shorts, embed).
        options: Extraction options. Uses defaults when None.

    Returns:
        VideoAnalysis with .metadata, .transcript, .frames populated.

    Raises:
        RuntimeError: If metadata extraction fails.
        ValueError: If the URL is invalid or points to a playlist.

    Example::

        from yt_agent import extract, ExtractOptions

        result = extract(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            ExtractOptions(languages=["en"], extract_frames=False),
        )
        print(result.metadata.title)
        print(result.transcript[0].text)
    """
    if options is None:
        options = ExtractOptions()

    import re

    # Resolve video ID for cache and output dir
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    video_id = m.group(1) if m else url

    output_dir = options.output_dir or Path("yt-agent-output") / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    from yt_agent.cache import (
        load_metadata,
        load_transcript,
        store_metadata,
        store_transcript,
    )
    from yt_agent.metadata import extract_metadata, save_metadata

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta: VideoMetadata | None = None
    if not options.no_cache:
        cached = load_metadata(video_id)
        if cached:
            cached.pop("_cached_at", None)
            meta = VideoMetadata.model_validate(cached)

    if meta is None:
        meta = extract_metadata(url)
        store_metadata(video_id, meta.model_dump(mode="json"))

    save_metadata(meta, output_dir)

    # ── Transcript ────────────────────────────────────────────────────────────
    segments: list[TranscriptSegment] = []
    lang: str = "en"

    if options.use_whisper:
        from yt_agent.whisper_transcribe import transcribe as whisper_transcribe
        segments, lang = whisper_transcribe(url, model_name=options.whisper_model)
        store_transcript(video_id, [s.model_dump() for s in segments], lang)
    else:
        if not options.no_cache:
            cached_tx = load_transcript(video_id)
            if cached_tx is not None:
                raw_segs, lang = cached_tx
                segments = [TranscriptSegment.model_validate(s) for s in raw_segs]

        if not segments:
            try:
                from yt_agent.transcript import fetch_transcript
                segments, lang = fetch_transcript(
                    url,
                    languages=options.languages,
                    translate_to=options.translate_to,
                )
                store_transcript(video_id, [s.model_dump() for s in segments], lang)
            except (ValueError, RuntimeError):
                # Auto-fallback: try Whisper when youtube-transcript-api has no captions
                try:
                    from yt_agent.whisper_transcribe import transcribe as _whisper_fn
                    segments, lang = _whisper_fn(
                        url,
                        model_name=options.whisper_model,
                    )
                    store_transcript(video_id, [s.model_dump() for s in segments], lang)
                except ImportError:
                    segments = []
                except (RuntimeError, ValueError):
                    segments = []

    if segments:
        from yt_agent.transcript import save_transcript
        save_transcript(segments, lang, output_dir)

    # ── Frames ────────────────────────────────────────────────────────────────
    frames: list[FrameInfo] = []
    if options.extract_frames:
        from yt_agent.frames import extract_frames
        frames = extract_frames(
            url,
            output_dir,
            mode=options.frames_mode,
            interval=options.frames_interval,
            scene_threshold=options.frames_threshold,
        )

        # ── VLM descriptions (optional, requires frames) ──────────────────
        if options.vlm and frames:
            from yt_agent.vlm import DEFAULT_PROMPT, describe_frames, make_backend
            vlm_backend = make_backend(
                backend=options.vlm_backend,
                model=options.vlm_model,
                api_base=options.vlm_api_base,
                api_key=options.vlm_api_key,
            )
            frames = describe_frames(
                frames,
                output_dir,
                vlm_backend,
                prompt=options.vlm_prompt or DEFAULT_PROMPT,
            )

    return VideoAnalysis(
        metadata=meta,
        transcript=segments,
        frames=frames,
    )

"""Transcript extraction from YouTube videos via youtube-transcript-api."""

from __future__ import annotations

from pathlib import Path

from yt_agent.schemas import TranscriptSegment

try:
    from youtube_transcript_api import (
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
        YouTubeTranscriptApi,
    )
    from youtube_transcript_api._transcripts import Transcript
except ImportError as exc:
    raise ImportError(
        "youtube-transcript-api is required: pip install youtube-transcript-api"
    ) from exc


def _video_id_from_url(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    import re

    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    # Assume bare video ID was passed
    if re.match(r"^[A-Za-z0-9_-]{11}$", url):
        return url
    raise ValueError(f"Cannot extract video ID from: {url!r}")


def list_languages(url: str) -> list[dict[str, str]]:
    """Return available transcript languages for a video.

    Args:
        url: YouTube video URL or bare video ID.

    Returns:
        List of dicts with keys: language, language_code, is_generated, is_translatable.
    """
    video_id = _video_id_from_url(url)
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    except TranscriptsDisabled:
        raise ValueError(f"Transcripts are disabled for video {video_id!r}.")
    except VideoUnavailable:
        raise ValueError(f"Video {video_id!r} is unavailable.")

    result = []
    for t in transcript_list:
        result.append({
            "language": t.language,
            "language_code": t.language_code,
            "is_generated": t.is_generated,
            "is_translatable": t.is_translatable,
        })
    return result


def fetch_transcript(
    url: str,
    languages: list[str] | None = None,
    translate_to: str | None = None,
) -> tuple[list[TranscriptSegment], str]:
    """Fetch auto/manual subtitles for a YouTube video.

    Tries the preferred languages in order, then falls back to any available
    transcript (auto-generated included). Optionally translates via YouTube's
    built-in translation API.

    Args:
        url: YouTube video URL or bare video ID.
        languages: Ordered list of BCP-47 language codes to try first.
                   Pass ["auto"] or None to accept any language without preference.
                   Defaults to ["en"].
        translate_to: BCP-47 code to translate the fetched transcript into.
                      Requires the source transcript to be translatable.

    Returns:
        Tuple of (segments, language_code) where language_code reflects the
        final language (post-translation if requested).

    Raises:
        ValueError: If the video ID cannot be parsed, transcripts are disabled,
                    or translation is not supported for the selected transcript.
        RuntimeError: If no transcript is available.
    """
    auto_mode = languages is None or languages == ["auto"]
    if auto_mode:
        effective_languages: list[str] = []
    else:
        effective_languages = languages  # type: ignore[assignment]

    video_id = _video_id_from_url(url)

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    except TranscriptsDisabled:
        raise ValueError(f"Transcripts are disabled for video {video_id!r}.")
    except VideoUnavailable:
        raise ValueError(f"Video {video_id!r} is unavailable.")

    transcript: Transcript | None = None

    if not auto_mode:
        # Try manual transcripts first, then generated, in preferred order
        for lang in effective_languages:
            try:
                transcript = transcript_list.find_manually_created_transcript([lang])
                break
            except NoTranscriptFound:
                pass

        if transcript is None:
            for lang in effective_languages:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    break
                except NoTranscriptFound:
                    pass

    if transcript is None:
        # Fall back to whatever is available
        try:
            transcript = next(iter(transcript_list))
        except StopIteration:
            raise RuntimeError(f"No transcripts found for video {video_id!r}.")

    # Apply translation if requested
    if translate_to and translate_to != transcript.language_code:
        if not transcript.is_translatable:
            raise ValueError(
                f"Transcript in {transcript.language_code!r} does not support translation."
            )
        transcript = transcript.translate(translate_to)

    raw = transcript.fetch()
    language_code: str = transcript.language_code

    segments: list[TranscriptSegment] = []
    for entry in raw:
        start = float(entry["start"])
        duration = float(entry.get("duration", 0.0))
        segments.append(
            TranscriptSegment(
                start=start,
                end=start + duration,
                text=entry["text"].strip(),
            )
        )

    return segments, language_code


def save_transcript(
    segments: list[TranscriptSegment],
    language_code: str,
    output_dir: Path,
) -> Path:
    """Write transcript segments to <output_dir>/transcript.md.

    Format: timestamped markdown with HH:MM:SS anchors.

    Args:
        segments: List of TranscriptSegment instances.
        language_code: BCP-47 language code of the transcript.
        output_dir: Directory to write into (created if absent).

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "transcript.md"

    lines: list[str] = [f"# Transcript\n\n**Language:** `{language_code}`\n"]
    for seg in segments:
        ts = _fmt_timestamp(seg.start)
        lines.append(f"**[{ts}]** {seg.text}")

    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def _fmt_timestamp(seconds: float) -> str:
    """Format float seconds as HH:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

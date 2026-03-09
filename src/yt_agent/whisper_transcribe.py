"""Whisper fallback transcription using faster-whisper + yt-dlp audio download.

Used when youtube-transcript-api has no subtitles or --whisper is explicitly requested.

Workflow:
    1. Download audio-only stream via yt-dlp to a temp file.
    2. Transcribe with faster-whisper (CTranslate2 backend).
    3. Return TranscriptSegment list compatible with the rest of the pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from yt_agent.schemas import TranscriptSegment

if TYPE_CHECKING:
    pass

DEFAULT_MODEL = "base"
SUPPORTED_MODELS = ("tiny", "tiny.en", "base", "base.en", "small", "small.en",
                    "medium", "medium.en", "large-v1", "large-v2", "large-v3")


def _download_audio(url: str, dest: Path) -> Path:
    """Download audio-only stream to dest using yt-dlp.

    Returns the actual output path (yt-dlp may add an extension).
    """
    try:
        import yt_dlp
    except ImportError as exc:
        raise ImportError("yt-dlp is required: pip install yt-dlp") from exc

    template = str(dest / "audio.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": template,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            ydl.download([url])
        except yt_dlp.utils.DownloadError as exc:
            raise RuntimeError(f"Audio download failed: {exc}") from exc

    wav = dest / "audio.wav"
    if wav.exists():
        return wav
    # Fallback: find any audio file yt-dlp wrote
    for p in dest.iterdir():
        if p.suffix in {".wav", ".m4a", ".opus", ".webm", ".mp3"}:
            return p
    raise RuntimeError("Audio download produced no output file.")


def transcribe(
    url: str,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    device: str = "auto",
) -> tuple[list[TranscriptSegment], str]:
    """Download audio and transcribe with faster-whisper.

    Args:
        url: YouTube video URL.
        model_name: Whisper model size (e.g. "base", "small", "medium").
        language: BCP-47 code to force (None = auto-detect).
        device: "cpu", "cuda", or "auto" (picks cuda if available).

    Returns:
        Tuple of (segments, detected_language_code).

    Raises:
        ImportError: If faster-whisper is not installed.
        RuntimeError: If audio download or transcription fails.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is required for Whisper transcription: "
            "pip install faster-whisper"
        ) from exc

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model {model_name!r}. Choose from: {', '.join(SUPPORTED_MODELS)}"
        )

    # Resolve device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"

    with tempfile.TemporaryDirectory(prefix="yt-agent-whisper-") as tmpdir:
        tmp = Path(tmpdir)
        audio_path = _download_audio(url, tmp)

        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        segments_gen, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        detected_language: str = info.language
        segments: list[TranscriptSegment] = []
        for seg in segments_gen:
            segments.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                )
            )

    return segments, detected_language

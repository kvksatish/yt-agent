"""CLI entry point for yt-agent."""

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from yt_agent import __version__
from yt_agent.cache import (
    cache_info,
    invalidate,
    load_metadata,
    load_transcript,
    store_metadata,
    store_transcript,
)
from yt_agent.metadata import extract_metadata, save_metadata
from yt_agent.schemas import TranscriptSegment, VideoMetadata
from yt_agent.transcript import fetch_transcript, save_transcript

def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"yt-agent {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="yt-agent",
    help="YouTube intelligence tool for AI agents.",
    no_args_is_help=True,
)


@app.callback()
def _app_options(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True,
                     help="Show version and exit."),
    ] = None,
) -> None:
    pass


class OutputFormat(str, Enum):
    json = "json"
    text = "text"
    markdown = "markdown"


def _video_id(url: str) -> str:
    """Quick video ID extraction for cache lookup."""
    import re
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else url


@app.command()
def main(
    url: Annotated[
        str,
        typer.Argument(help="YouTube video URL to process."),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory path."),
    ] = None,
    format: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format."),
    ] = OutputFormat.json,
    frames: Annotated[
        bool,
        typer.Option("--frames/--no-frames", help="Extract frames from video via ffmpeg."),
    ] = False,
    frames_mode: Annotated[
        str,
        typer.Option("--frames-mode", help="Frame extraction mode: interval, scene, or keyframe."),
    ] = "interval",
    frames_interval: Annotated[
        float,
        typer.Option("--frames-interval", help="Seconds between frames (interval mode)."),
    ] = 30.0,
    frames_threshold: Annotated[
        float,
        typer.Option("--frames-threshold", help="Scene change threshold 0.0-1.0 (scene mode)."),
    ] = 0.3,
    vlm: Annotated[
        bool,
        typer.Option("--vlm/--no-vlm", help="Describe frames with a vision LLM (requires --frames)."),
    ] = False,
    vlm_backend: Annotated[
        str,
        typer.Option("--vlm-backend", help="VLM backend: ollama (default) or openai."),
    ] = "ollama",
    vlm_model: Annotated[
        Optional[str],
        typer.Option("--vlm-model", help="VLM model name (e.g. llava, gpt-4o)."),
    ] = None,
    vlm_api_base: Annotated[
        Optional[str],
        typer.Option("--vlm-api-base", help="Override VLM API base URL."),
    ] = None,
    vlm_api_key: Annotated[
        str,
        typer.Option("--vlm-api-key", envvar="OPENAI_API_KEY", help="API key for openai VLM backend."),
    ] = "",
    whisper: Annotated[
        bool,
        typer.Option("--whisper/--no-whisper", help="Transcribe audio with Whisper instead of youtube-transcript-api."),
    ] = False,
    whisper_model: Annotated[
        str,
        typer.Option("--whisper-model", help="Whisper model size (tiny/base/small/medium/large-v3)."),
    ] = "base",
    language: Annotated[
        Optional[str],
        typer.Option("--language", "-l", help="Preferred transcript language (BCP-47, e.g. 'fr'). Default: auto."),
    ] = None,
    translate_to: Annotated[
        Optional[str],
        typer.Option("--translate-to", help="Translate transcript to this language (BCP-47, e.g. 'en')."),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Bypass cache; always re-fetch."),
    ] = False,
) -> None:
    """Process a YouTube video URL and extract intelligence."""
    typer.echo(f"Processing: {url}")

    output_dir = output if output else Path("yt-agent-output")
    vid = _video_id(url)

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta: VideoMetadata | None = None
    if not no_cache:
        cached = load_metadata(vid)
        if cached:
            typer.echo("[cache] metadata hit")
            cached.pop("_cached_at", None)
            meta = VideoMetadata.model_validate(cached)

    if meta is None:
        try:
            meta = extract_metadata(url)
        except (RuntimeError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        store_metadata(vid, meta.model_dump(mode="json"))

    meta_path = save_metadata(meta, output_dir)
    typer.echo(f"Metadata saved: {meta_path}")

    if format == OutputFormat.json:
        typer.echo(meta.model_dump_json(indent=2))
    elif format == OutputFormat.text:
        typer.echo(f"Title:    {meta.title}")
        typer.echo(f"Channel:  {meta.channel}")
        typer.echo(f"Duration: {meta.duration_seconds}s")
        typer.echo(f"Views:    {meta.view_count}")
        typer.echo(f"Tags:     {', '.join(meta.tags[:10])}")
    elif format == OutputFormat.markdown:
        typer.echo(f"# {meta.title}")
        typer.echo(f"**Channel:** {meta.channel}  ")
        typer.echo(f"**Duration:** {meta.duration_seconds}s  ")
        typer.echo(f"**Views:** {meta.view_count}  ")
        if meta.tags:
            typer.echo(f"**Tags:** {', '.join(meta.tags[:10])}")

    # ── Transcript ────────────────────────────────────────────────────────────
    if not whisper:
        segments: list[TranscriptSegment] | None = None
        lang: str = "en"

        if not no_cache:
            cached_tx = load_transcript(vid)
            if cached_tx is not None:
                raw_segs, lang = cached_tx
                typer.echo("[cache] transcript hit")
                segments = [TranscriptSegment.model_validate(s) for s in raw_segs]

        if segments is None:
            preferred = [language] if language else None
            try:
                segments, lang = fetch_transcript(url, languages=preferred, translate_to=translate_to)
                store_transcript(vid, [s.model_dump() for s in segments], lang)
            except (ValueError, RuntimeError) as exc:
                typer.echo(f"Transcript unavailable: {exc}", err=True)
                # Auto-fallback to Whisper when youtube-transcript-api has no captions
                typer.echo(f"[whisper-fallback] Attempting Whisper transcription (model={whisper_model})...")
                try:
                    from yt_agent.whisper_transcribe import transcribe as _whisper_fn
                    segments, lang = _whisper_fn(url, model_name=whisper_model, language=language)
                    store_transcript(vid, [s.model_dump() for s in segments], lang)
                    typer.echo(f"[whisper-fallback] Transcription succeeded ({len(segments)} segments, lang={lang}).")
                except ImportError:
                    typer.echo(
                        "[whisper-fallback] faster-whisper not installed. "
                        "Run: pip install faster-whisper",
                        err=True,
                    )
                except (RuntimeError, ValueError) as w_exc:
                    typer.echo(f"[whisper-fallback] Whisper also failed: {w_exc}", err=True)

        if segments is not None:
            transcript_path = save_transcript(segments, lang, output_dir)
            typer.echo(
                f"Transcript saved: {transcript_path} "
                f"({len(segments)} segments, lang={lang})"
            )

    if frames:
        from yt_agent.frames import extract_frames
        typer.echo(f"Frame extraction: mode={frames_mode}")
        try:
            frame_list = extract_frames(
                url, output_dir,
                mode=frames_mode,
                interval=frames_interval,
                scene_threshold=frames_threshold,
            )
            typer.echo(f"Frames saved: {output_dir / 'frames'} ({len(frame_list)} frames)")
            if vlm and frame_list:
                from yt_agent.vlm import DEFAULT_PROMPT, describe_frames, make_backend
                typer.echo(f"VLM descriptions: backend={vlm_backend}, model={vlm_model or 'default'}")
                try:
                    backend = make_backend(vlm_backend, model=vlm_model, api_base=vlm_api_base, api_key=vlm_api_key)
                    describe_frames(frame_list, output_dir, backend)
                    typer.echo("VLM descriptions saved alongside frames (.txt)")
                except (RuntimeError, ValueError) as vlm_exc:
                    typer.echo(f"VLM description failed: {vlm_exc}", err=True)
        except (RuntimeError, ValueError) as exc:
            typer.echo(f"Frame extraction failed: {exc}", err=True)
    if whisper:
        from yt_agent.whisper_transcribe import transcribe
        typer.echo(f"Whisper transcription: model={whisper_model}")
        try:
            segments, lang = transcribe(url, model_name=whisper_model)
            store_transcript(vid, [s.model_dump() for s in segments], lang)
            transcript_path = save_transcript(segments, lang, output_dir)
            typer.echo(
                f"Transcript saved: {transcript_path} "
                f"({len(segments)} segments, lang={lang})"
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            typer.echo(f"Whisper transcription failed: {exc}", err=True)


@app.command("languages")
def cmd_languages(
    url: Annotated[str, typer.Argument(help="YouTube video URL.")],
) -> None:
    """List available transcript languages for a video."""
    from yt_agent.transcript import list_languages
    try:
        langs = list_languages(url)
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    for entry in langs:
        generated = " [auto]" if entry["is_generated"] else ""
        translatable = " [translatable]" if entry["is_translatable"] else ""
        typer.echo(f"{entry['language_code']:8s}  {entry['language']}{generated}{translatable}")


def _run_one(url: str, kwargs: dict) -> tuple[str, bool, str]:
    """Process a single URL; returns (url, success, message)."""
    import io
    import contextlib

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            # Invoke the main command programmatically
            from yt_agent.metadata import extract_metadata, save_metadata
            from yt_agent.cache import load_metadata, store_metadata, load_transcript, store_transcript
            from yt_agent.transcript import fetch_transcript, save_transcript
            from yt_agent.schemas import VideoMetadata, TranscriptSegment

            output_dir = kwargs["output_dir"]
            no_cache = kwargs["no_cache"]
            whisper = kwargs["whisper"]
            whisper_model = kwargs["whisper_model"]
            language = kwargs["language"]
            translate_to = kwargs["translate_to"]

            vid = _video_id(url)

            meta: VideoMetadata | None = None
            if not no_cache:
                cached = load_metadata(vid)
                if cached:
                    cached.pop("_cached_at", None)
                    meta = VideoMetadata.model_validate(cached)
            if meta is None:
                meta = extract_metadata(url)
                store_metadata(vid, meta.model_dump(mode="json"))
            save_metadata(meta, output_dir / vid)

            if not whisper:
                segs: list[TranscriptSegment] | None = None
                lang = "en"
                if not no_cache:
                    cached_tx = load_transcript(vid)
                    if cached_tx is not None:
                        raw_segs, lang = cached_tx
                        segs = [TranscriptSegment.model_validate(s) for s in raw_segs]
                if segs is None:
                    preferred = [language] if language else None
                    try:
                        segs, lang = fetch_transcript(url, languages=preferred, translate_to=translate_to)
                        store_transcript(vid, [s.model_dump() for s in segs], lang)
                    except (ValueError, RuntimeError):
                        pass
                if segs is not None:
                    save_transcript(segs, lang, output_dir / vid)
            else:
                from yt_agent.whisper_transcribe import transcribe
                segs, lang = transcribe(url, model_name=whisper_model)
                store_transcript(vid, [s.model_dump() for s in segs], lang)
                save_transcript(segs, lang, output_dir / vid)

        return url, True, meta.title
    except Exception as exc:  # noqa: BLE001
        return url, False, str(exc)


@app.command("batch")
def cmd_batch(
    playlist: Annotated[
        Optional[str],
        typer.Option("--playlist", help="YouTube playlist URL to process all videos."),
    ] = None,
    channel: Annotated[
        Optional[str],
        typer.Option("--channel", help="YouTube channel URL to process all videos."),
    ] = None,
    batch_file: Annotated[
        Optional[str],
        typer.Option("--batch", help="Path to a file with one YouTube URL per line."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory (each video gets a sub-folder)."),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Number of parallel workers."),
    ] = 4,
    whisper: Annotated[
        bool,
        typer.Option("--whisper/--no-whisper", help="Use Whisper for transcription."),
    ] = False,
    whisper_model: Annotated[
        str,
        typer.Option("--whisper-model", help="Whisper model size."),
    ] = "base",
    language: Annotated[
        Optional[str],
        typer.Option("--language", "-l", help="Preferred transcript language (BCP-47)."),
    ] = None,
    translate_to: Annotated[
        Optional[str],
        typer.Option("--translate-to", help="Translate transcript to this language."),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Bypass cache."),
    ] = False,
) -> None:
    """Process multiple YouTube videos from a playlist, channel, or URL file."""
    from yt_agent.batch import resolve_playlist, read_batch_file
    import concurrent.futures

    urls: list[str] = []

    if playlist:
        typer.echo(f"Resolving playlist: {playlist}")
        try:
            urls += resolve_playlist(playlist)
        except RuntimeError as exc:
            typer.echo(f"Error resolving playlist: {exc}", err=True)
            raise typer.Exit(code=1)

    if channel:
        typer.echo(f"Resolving channel: {channel}")
        try:
            urls += resolve_playlist(channel)
        except RuntimeError as exc:
            typer.echo(f"Error resolving channel: {exc}", err=True)
            raise typer.Exit(code=1)

    if batch_file:
        try:
            urls += read_batch_file(batch_file)
        except (OSError, ValueError) as exc:
            typer.echo(f"Error reading batch file: {exc}", err=True)
            raise typer.Exit(code=1)

    if not urls:
        typer.echo("No URLs found. Provide --playlist, --channel, or --batch.", err=True)
        raise typer.Exit(code=1)

    urls = list(dict.fromkeys(urls))  # deduplicate, preserve order
    typer.echo(f"Processing {len(urls)} videos with {workers} workers...")

    output_dir = output if output else Path("yt-agent-batch-output")
    kwargs = {
        "output_dir": output_dir,
        "no_cache": no_cache,
        "whisper": whisper,
        "whisper_model": whisper_model,
        "language": language,
        "translate_to": translate_to,
    }

    ok = 0
    fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, url, kwargs): url for url in urls}
        for future in concurrent.futures.as_completed(futures):
            url, success, msg = future.result()
            if success:
                ok += 1
                typer.echo(f"  [ok] {msg}")
            else:
                fail += 1
                typer.echo(f"  [fail] {url}: {msg}", err=True)

    typer.echo(f"\nDone: {ok} succeeded, {fail} failed.")
    if fail:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

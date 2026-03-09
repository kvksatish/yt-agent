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

app = typer.Typer(
    name="yt-agent",
    help="YouTube intelligence tool for AI agents.",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    json = "json"
    text = "text"
    markdown = "markdown"


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"yt-agent {__version__}")
        raise typer.Exit()


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
        typer.Option("--frames/--no-frames", help="Extract key frames from video."),
    ] = False,
    whisper: Annotated[
        bool,
        typer.Option("--whisper/--no-whisper", help="Transcribe audio with Whisper."),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Bypass cache; always re-fetch."),
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version", "-V", callback=version_callback, is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
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
            try:
                segments, lang = fetch_transcript(url)
                store_transcript(vid, [s.model_dump() for s in segments], lang)
            except (ValueError, RuntimeError) as exc:
                typer.echo(f"Transcript unavailable: {exc}", err=True)

        if segments is not None:
            transcript_path = save_transcript(segments, lang, output_dir)
            typer.echo(
                f"Transcript saved: {transcript_path} "
                f"({len(segments)} segments, lang={lang})"
            )

    if frames:
        typer.echo("Frame extraction: not yet implemented")
    if whisper:
        typer.echo("Whisper transcription: not yet implemented")


if __name__ == "__main__":
    app()

"""CLI entry point for yt-agent."""

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from yt_agent import __version__
from yt_agent.metadata import extract_metadata, save_metadata

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


@app.command()
def main(
    url: Annotated[
        str,
        typer.Argument(help="YouTube video URL to process."),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path."),
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

    output_dir = Path(output) if output else Path(f"yt-agent-output")

    try:
        meta = extract_metadata(url)
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

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

    if frames:
        typer.echo("Frame extraction: not yet implemented")
    if whisper:
        typer.echo("Whisper transcription: not yet implemented")


if __name__ == "__main__":
    app()

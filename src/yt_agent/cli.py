"""CLI entry point for yt-agent."""

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from yt_agent import __version__

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
    typer.echo(f"Format: {format.value}")
    if output:
        typer.echo(f"Output: {output}")
    if frames:
        typer.echo("Frame extraction: enabled")
    if whisper:
        typer.echo("Whisper transcription: enabled")


if __name__ == "__main__":
    app()

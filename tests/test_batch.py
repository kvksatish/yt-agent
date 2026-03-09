"""Tests for batch URL utilities — no network required."""

import textwrap
from pathlib import Path

import pytest

from yt_agent.batch import read_batch_file


def test_read_batch_file(tmp_path: Path):
    f = tmp_path / "urls.txt"
    f.write_text(textwrap.dedent("""\
        https://youtu.be/aaa
        # comment line
        https://youtu.be/bbb

        https://youtu.be/ccc
    """))
    urls = read_batch_file(str(f))
    assert urls == [
        "https://youtu.be/aaa",
        "https://youtu.be/bbb",
        "https://youtu.be/ccc",
    ]


def test_read_batch_file_missing(tmp_path: Path):
    with pytest.raises(OSError):
        read_batch_file(str(tmp_path / "nonexistent.txt"))

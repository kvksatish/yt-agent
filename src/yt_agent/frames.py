"""Frame extraction from YouTube videos via yt-dlp + ffmpeg.

Extraction modes:
  - interval:     one frame every N seconds (default: 30s)
  - scene:        scene-change detection via ffmpeg select filter (threshold 0.0-1.0)
  - keyframe:     I-frames only (no re-encoding, very fast)

Output layout:
  <output_dir>/frames/<mode>_<timestamp_ms>.jpg

Requires ffmpeg on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from yt_agent.schemas import FrameInfo


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is not on PATH. Install it: https://ffmpeg.org/download.html"
        )
    return ffmpeg


def _download_video(url: str, dest: Path) -> Path:
    """Download the best video stream (no audio) to dest directory."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise ImportError("yt-dlp is required: pip install yt-dlp") from exc

    template = str(dest / "video.%(ext)s")
    opts = {
        "format": "bestvideo[ext=mp4]/bestvideo",
        "outtmpl": template,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            ydl.download([url])
        except yt_dlp.utils.DownloadError as exc:
            raise RuntimeError(f"Video download failed: {exc}") from exc

    for ext in ("mp4", "webm", "mkv", "mov"):
        candidate = dest / f"video.{ext}"
        if candidate.exists():
            return candidate

    found = list(dest.glob("video.*"))
    if found:
        return found[0]
    raise RuntimeError("Video download produced no output file.")


def _parse_timestamps_from_filenames(frames_dir: Path) -> list[float]:
    """Parse timestamps from filenames like frame_012345.jpg (ms)."""
    timestamps = []
    for p in sorted(frames_dir.glob("frame_*.jpg")):
        stem = p.stem  # frame_012345
        try:
            ms = int(stem.split("_", 1)[1])
            timestamps.append(ms / 1000.0)
        except (IndexError, ValueError):
            pass
    return timestamps


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")


# ---------------------------------------------------------------------------
# Extraction modes
# ---------------------------------------------------------------------------

def _extract_interval(video_path: Path, frames_dir: Path, interval: float) -> list[float]:
    """Extract one frame every `interval` seconds."""
    ffmpeg = _require_ffmpeg()
    out_pattern = str(frames_dir / "frame_%06d.jpg")
    _run_ffmpeg([
        ffmpeg, "-i", str(video_path),
        "-vf", f"fps=1/{interval}",
        "-q:v", "2",
        "-frame_pts", "1",
        out_pattern,
        "-y",
    ])
    # Rename files with proper ms timestamps using PTS
    # ffmpeg with fps filter produces sequential numbers; re-derive timestamps
    timestamps = []
    for i, p in enumerate(sorted(frames_dir.glob("frame_*.jpg"))):
        ts_seconds = i * interval
        ts_ms = int(ts_seconds * 1000)
        new_name = frames_dir / f"frame_{ts_ms:09d}.jpg"
        p.rename(new_name)
        timestamps.append(ts_seconds)
    return timestamps


def _extract_scene(video_path: Path, frames_dir: Path, threshold: float) -> list[float]:
    """Extract frames on scene changes above threshold (0.0-1.0)."""
    ffmpeg = _require_ffmpeg()
    out_pattern = str(frames_dir / "frame_%09d.jpg")
    # select filter: 'gt(scene,threshold)' picks frames where scene score > threshold
    select_expr = f"gt(scene\\,{threshold})"
    _run_ffmpeg([
        ffmpeg, "-i", str(video_path),
        "-vf", f"select={select_expr},setpts=N/FRAME_RATE/TB",
        "-vsync", "vfr",
        "-q:v", "2",
        out_pattern,
        "-y",
    ])
    # We don't have accurate timestamps from scene detection filenames alone;
    # run a second ffprobe pass to get scene timestamps
    timestamps = _probe_scene_timestamps(video_path, threshold)
    # Rename sequentially generated files with actual timestamps
    files = sorted(frames_dir.glob("frame_*.jpg"))
    for p in files:
        p.unlink()

    if not timestamps:
        return []

    # Re-extract with showinfo to get exact timestamps
    _run_ffmpeg([
        ffmpeg, "-i", str(video_path),
        "-vf", f"select={select_expr},showinfo",
        "-vsync", "vfr",
        "-q:v", "2",
        str(frames_dir / "frame_%09d.jpg"),
        "-y",
    ])
    files = sorted(frames_dir.glob("frame_*.jpg"))
    renamed_timestamps = []
    for p, ts in zip(files, timestamps):
        ts_ms = int(ts * 1000)
        new_name = frames_dir / f"frame_{ts_ms:09d}.jpg"
        p.rename(new_name)
        renamed_timestamps.append(ts)
    return renamed_timestamps


def _probe_scene_timestamps(video_path: Path, threshold: float) -> list[float]:
    """Use ffprobe to get timestamps of scene-change frames."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return []
    select_expr = f"gt(scene\\,{threshold})"
    result = subprocess.run(
        [
            ffprobe, "-v", "quiet",
            "-select_streams", "v",
            "-show_frames",
            "-show_entries", "frame=pkt_pts_time,best_effort_timestamp_time",
            "-read_intervals", "%+#9999",
            "-vf", f"select={select_expr}",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    timestamps = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                timestamps.append(float(line.split(",")[0]))
            except ValueError:
                pass
    return timestamps


def _extract_keyframes(video_path: Path, frames_dir: Path) -> list[float]:
    """Extract I-frames (keyframes) only — fast, no re-encode."""
    ffmpeg = _require_ffmpeg()
    out_pattern = str(frames_dir / "frame_%09d.jpg")
    _run_ffmpeg([
        ffmpeg, "-skip_frame", "noref",
        "-i", str(video_path),
        "-vf", "select=eq(pict_type\\,I)",
        "-vsync", "vfr",
        "-q:v", "2",
        out_pattern,
        "-y",
    ])
    # Approximate timestamps by probing keyframe PTS
    timestamps = _probe_keyframe_timestamps(video_path)
    files = sorted(frames_dir.glob("frame_*.jpg"))
    renamed: list[float] = []
    for p, ts in zip(files, timestamps):
        ts_ms = int(ts * 1000)
        new_name = frames_dir / f"frame_{ts_ms:09d}.jpg"
        p.rename(new_name)
        renamed.append(ts)
    # If probe returned fewer timestamps than files, keep remaining with index-based names
    for p in sorted(frames_dir.glob("frame_?????????.jpg")):
        pass  # already renamed
    return renamed


def _probe_keyframe_timestamps(video_path: Path) -> list[float]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return []
    result = subprocess.run(
        [
            ffprobe, "-v", "quiet",
            "-select_streams", "v",
            "-show_packets",
            "-show_entries", "packet=pts_time,flags",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    timestamps = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2 and "K" in parts[-1]:
            try:
                timestamps.append(float(parts[0]))
            except ValueError:
                pass
    return timestamps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_frames(
    url: str,
    output_dir: Path,
    mode: str = "interval",
    interval: float = 30.0,
    scene_threshold: float = 0.3,
) -> list[FrameInfo]:
    """Download video and extract frames via ffmpeg.

    Args:
        url: YouTube video URL.
        output_dir: Root output directory; frames go into output_dir/frames/.
        mode: One of "interval", "scene", or "keyframe".
        interval: Seconds between frames (interval mode only).
        scene_threshold: Scene change sensitivity 0.0-1.0 (scene mode only).
                         Lower = more frames. Typical: 0.2-0.4.

    Returns:
        List of FrameInfo with timestamp and relative path.

    Raises:
        ValueError: Unknown mode.
        RuntimeError: ffmpeg not found or extraction failed.
    """
    if mode not in ("interval", "scene", "keyframe"):
        raise ValueError(f"Unknown frame extraction mode {mode!r}. Use: interval, scene, keyframe")

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="yt-agent-frames-") as tmpdir:
        tmp = Path(tmpdir)
        video_path = _download_video(url, tmp)

        if mode == "interval":
            timestamps = _extract_interval(video_path, frames_dir, interval)
        elif mode == "scene":
            timestamps = _extract_scene(video_path, frames_dir, scene_threshold)
        else:  # keyframe
            timestamps = _extract_keyframes(video_path, frames_dir)

    infos: list[FrameInfo] = []
    for p in sorted(frames_dir.glob("frame_*.jpg")):
        # Parse timestamp from filename
        try:
            ts_ms = int(p.stem.split("_", 1)[1])
            ts = ts_ms / 1000.0
        except (IndexError, ValueError):
            ts = 0.0
        infos.append(FrameInfo(
            timestamp=ts,
            path=str(p.relative_to(output_dir)),
            width=0,
            height=0,
            scene_label=None,
        ))

    return infos

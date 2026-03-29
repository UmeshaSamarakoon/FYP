from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


PREVIEW_DIR = Path("uploads") / "previews"
PREVIEW_TTL_SECONDS = int(os.getenv("CAUSALX_PREVIEW_TTL_SECONDS", str(24 * 60 * 60)))
PREVIEW_PRUNE_INTERVAL_SECONDS = int(os.getenv("CAUSALX_PREVIEW_PRUNE_INTERVAL_SECONDS", str(10 * 60)))
PREVIEW_X264_PRESET = os.getenv("CAUSALX_PREVIEW_X264_PRESET", "veryfast")
PREVIEW_X264_CRF = os.getenv("CAUSALX_PREVIEW_X264_CRF", "23")

_LAST_PRUNE_TS = 0.0


def _safe_analysis_id(analysis_id: str) -> str:
    return Path(str(analysis_id)).name


def preview_path_for_analysis(analysis_id: str) -> Path:
    return PREVIEW_DIR / f"{_safe_analysis_id(analysis_id)}.mp4"


def preview_url_for_analysis(analysis_id: str) -> str:
    return f"/preview/{_safe_analysis_id(analysis_id)}"


def preview_exists(analysis_id: str) -> bool:
    path = preview_path_for_analysis(analysis_id)
    try:
        return path.exists() and path.stat().st_size > 0
    except FileNotFoundError:
        return False


def cleanup_preview_cache(force: bool = False) -> None:
    global _LAST_PRUNE_TS

    now = time.time()
    if not force and now - _LAST_PRUNE_TS < PREVIEW_PRUNE_INTERVAL_SECONDS:
        return
    _LAST_PRUNE_TS = now

    if PREVIEW_TTL_SECONDS <= 0 or not PREVIEW_DIR.exists():
        return

    cutoff = now - PREVIEW_TTL_SECONDS
    for path in PREVIEW_DIR.glob("*.mp4"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except FileNotFoundError:
            continue


def is_browser_compatible_mp4(
    video_codec: str | None,
    pix_fmt: str | None,
    audio_codec: str | None,
) -> bool:
    v_codec = str(video_codec or "").strip().lower()
    v_pix_fmt = str(pix_fmt or "").strip().lower()
    a_codec = str(audio_codec or "").strip().lower()

    if v_codec != "h264":
        return False
    if v_pix_fmt and not (
        v_pix_fmt.startswith("yuv420")
        or v_pix_fmt.startswith("yuvj420")
        or v_pix_fmt == "nv12"
    ):
        return False
    return a_codec in {"", "aac", "mp3"}


def _probe_streams(video_path: str) -> dict[str, int | str | None]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,pix_fmt,width,height",
        "-of",
        "default=noprint_wrappers=1:nokey=0",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return {}

    video_codec = None
    pix_fmt = None
    audio_codec = None
    width = None
    height = None
    current_type = None

    for raw_line in (result.stdout or "").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "codec_type":
            current_type = value
        elif key == "codec_name":
            if current_type == "video" and video_codec is None:
                video_codec = value
            elif current_type == "audio" and audio_codec is None:
                audio_codec = value
        elif key == "pix_fmt" and current_type == "video" and pix_fmt is None:
            pix_fmt = value
        elif key == "width" and current_type == "video" and width is None:
            try:
                width = int(value)
            except ValueError:
                width = None
        elif key == "height" and current_type == "video" and height is None:
            try:
                height = int(value)
            except ValueError:
                height = None

    return {
        "video_codec": video_codec,
        "pix_fmt": pix_fmt,
        "audio_codec": audio_codec,
        "width": width,
        "height": height,
    }


def ensure_video_preview(analysis_id: str, video_path: str) -> str | None:
    cleanup_preview_cache()

    source_path = Path(video_path)
    if not source_path.exists():
        return None

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    final_path = preview_path_for_analysis(analysis_id)
    if final_path.exists() and final_path.stat().st_size > 0:
        return preview_url_for_analysis(analysis_id)

    temp_path = final_path.with_suffix(".tmp.mp4")
    if temp_path.exists():
        temp_path.unlink()

    stream_info = _probe_streams(str(source_path))
    is_compatible = is_browser_compatible_mp4(
        stream_info.get("video_codec"),
        stream_info.get("pix_fmt"),
        stream_info.get("audio_codec"),
    )

    try:
        if is_compatible:
            shutil.copy2(source_path, temp_path)
        else:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                return None

            cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c:v",
                "libx264",
                "-preset",
                PREVIEW_X264_PRESET,
                "-crf",
                PREVIEW_X264_CRF,
            ]

            width = stream_info.get("width")
            height = stream_info.get("height")
            if isinstance(width, int) and isinstance(height, int) and ((width % 2) or (height % 2)):
                cmd.extend(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])

            cmd.extend(
                [
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    str(temp_path),
                ]
            )

            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        temp_path.replace(final_path)
        return preview_url_for_analysis(analysis_id)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        return None

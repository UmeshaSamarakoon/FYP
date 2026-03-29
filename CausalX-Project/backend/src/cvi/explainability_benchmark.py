from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.cvi.pipeline import CausalInferenceEngine


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAKEAV_ROOT = (
    BACKEND_ROOT / "data" / "validation_evaluation_videos" / "evaluation" / "data" / "raw" / "fakeavceleb"
)
DEFAULT_ANNOTATION_DIR = BACKEND_ROOT / "data" / "annotations"
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "outputs" / "cvi_explainability_benchmark"
PROTOCOL_VERSION = "cvi_explainability_eval_v5"
DEFAULT_INFERENCE_CONFIG = {
    "prob_thresh": 0.247707,
    "ratio_thresh": 0.60,
    "smooth_window": 5,
    "chunk_seconds": 10,
    "causal_thresh": 0.75,
    "max_seconds": 45.0,
    "target_fps": None,
    "include_bboxes": True,
    "scm_z_thresh": 2.0,
    "require_flag": False,
}
MANIPULATION_ORDER = [
    "RealVideo-RealAudio",
    "RealVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio",
]
DEFAULT_STABILITY_VARIANTS = ("brightness", "noise")


def _pipeline_symbols():
    from src.cvi.pipeline import CausalInferenceEngine, build_segments

    return CausalInferenceEngine, build_segments


def _inference_defaults() -> dict[str, object]:
    defaults = dict(DEFAULT_INFERENCE_CONFIG)
    try:
        from src.cvi.api import inference_service as service

        defaults.update(
            {
                "prob_thresh": float(service.PROB_THRESH),
                "ratio_thresh": float(service.RATIO_THRESH),
                "smooth_window": int(service.SMOOTH_WINDOW),
                "chunk_seconds": int(service.CHUNK_SECONDS),
                "causal_thresh": float(service.CAUSAL_THRESH),
                "max_seconds": None if service.MAX_SECONDS is None else float(service.MAX_SECONDS),
                "target_fps": None if service.TARGET_FPS is None else float(service.TARGET_FPS),
                "include_bboxes": bool(service.INCLUDE_BBOXES),
                "scm_z_thresh": float(service.SCM_Z_THRESH),
                "require_flag": bool(service.REQUIRE_FLAG),
            }
        )
    except Exception:
        pass
    return defaults


def annotation_video_id(video_path: Path, root: Path) -> str:
    try:
        rel = video_path.relative_to(root)
        return "__".join(rel.with_suffix("").parts)
    except Exception:
        pass
    try:
        rel = video_path.resolve().relative_to(root.resolve())
        return "__".join(rel.with_suffix("").parts)
    except Exception:
        return video_path.stem


def derive_path_labels(video_path: Path, root: Path) -> dict[str, object]:
    try:
        rel_parts = list(video_path.relative_to(root).parts)
    except Exception:
        try:
            rel_parts = list(video_path.resolve().relative_to(root.resolve()).parts)
        except Exception:
            rel_parts = list(video_path.parts)

    labels: dict[str, object] = {
        "manipulation_type": None,
        "region": None,
        "gender": None,
        "identity": None,
        "filename": video_path.name,
        "stem": video_path.stem,
    }
    if len(rel_parts) >= 1:
        labels["manipulation_type"] = rel_parts[0]
    if len(rel_parts) >= 2:
        labels["region"] = rel_parts[1]
    if len(rel_parts) >= 3:
        labels["gender"] = rel_parts[2]
    if len(rel_parts) >= 4:
        labels["identity"] = rel_parts[3]

    mt = str(labels["manipulation_type"] or "")
    labels["is_fake_video"] = "FakeVideo" in mt
    labels["is_fake_audio"] = "FakeAudio" in mt
    labels["is_real_video"] = "RealVideo" in mt
    labels["is_real_audio"] = "RealAudio" in mt
    labels["is_fake_any"] = bool(labels["is_fake_video"] or labels["is_fake_audio"])
    labels["label_binary"] = int(labels["is_fake_any"])
    labels["manipulation_case"] = mt or "unknown"
    return labels


def list_fakeav_videos(video_root: Path) -> list[Path]:
    return sorted(path for path in video_root.rglob("*.mp4") if path.exists())


def ffprobe_duration_seconds(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            return max(0.0, float(proc.stdout.strip()))
        except ValueError:
            pass
    return 0.0


def ffprobe_has_audio(video_path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _round_robin_sample(paths: Sequence[Path], root: Path, take: int, seed: int) -> list[Path]:
    if take <= 0 or not paths:
        return []

    rng = random.Random(seed)
    buckets: dict[str, list[Path]] = {}
    for path in paths:
        labels = derive_path_labels(path, root)
        key = str(labels.get("identity") or path.stem)
        buckets.setdefault(key, []).append(path)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    keys = list(buckets)
    rng.shuffle(keys)

    selected: list[Path] = []
    while len(selected) < take and keys:
        next_keys: list[str] = []
        for key in keys:
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop())
            if len(selected) >= take:
                break
            if bucket:
                next_keys.append(key)
        if not next_keys and len(selected) < take:
            break
        rng.shuffle(next_keys)
        keys = next_keys
    return selected[:take]


def balanced_subset(video_root: Path, total_videos: int, seed: int) -> list[Path]:
    groups: dict[str, list[Path]] = {name: [] for name in MANIPULATION_ORDER}
    for path in list_fakeav_videos(video_root):
        mt = str(derive_path_labels(path, video_root).get("manipulation_type") or "")
        groups.setdefault(mt, []).append(path)

    selected: list[Path] = []
    remaining = total_videos
    remaining_groups = [name for name in MANIPULATION_ORDER if groups.get(name)]
    allocations: dict[str, int] = {}

    for idx, name in enumerate(remaining_groups):
        groups_left = len(remaining_groups) - idx
        desired = max(1, remaining // groups_left)
        take = min(desired, len(groups[name]))
        allocations[name] = take
        remaining -= take

    if remaining > 0:
        expandable = [name for name in remaining_groups if len(groups[name]) > allocations[name]]
        ex_idx = 0
        while remaining > 0 and expandable:
            name = expandable[ex_idx % len(expandable)]
            if allocations[name] < len(groups[name]):
                allocations[name] += 1
                remaining -= 1
            expandable = [n for n in remaining_groups if len(groups[n]) > allocations[n]]
            ex_idx += 1

    for idx, name in enumerate(MANIPULATION_ORDER):
        take = allocations.get(name, 0)
        selected.extend(_round_robin_sample(groups.get(name, []), video_root, take, seed + idx))

    return sorted(selected)


def subset_manifest_df(video_paths: Sequence[Path], video_root: Path) -> pd.DataFrame:
    rows = []
    for path in video_paths:
        labels = derive_path_labels(path, video_root)
        duration = ffprobe_duration_seconds(path)
        rows.append(
            {
                "video_id": annotation_video_id(path, video_root),
                "video_path": str(path),
                "filename": path.name,
                "manipulation_type": labels.get("manipulation_type"),
                "region": labels.get("region"),
                "gender": labels.get("gender"),
                "identity": labels.get("identity"),
                "label_binary": int(labels.get("label_binary", 0)),
                "is_fake_video": bool(labels.get("is_fake_video", False)),
                "is_fake_audio": bool(labels.get("is_fake_audio", False)),
                "duration_seconds": float(duration),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["manipulation_type", "region", "gender", "identity", "filename"]
    ).reset_index(drop=True)


def temporal_template_df(manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in manifest_df.to_dict(orient="records"):
        is_fake = bool(row.get("label_binary", 0))
        duration = float(row.get("duration_seconds", 0.0) or 0.0)
        rows.append(
            {
                "video_id": row["video_id"],
                "video_path": row["video_path"],
                "filename": row["filename"],
                "manipulation_type": row["manipulation_type"],
                "region": row["region"],
                "gender": row["gender"],
                "identity": row["identity"],
                "label_binary": int(row["label_binary"]),
                "duration_seconds": duration,
                "suggested_start": 0.0 if is_fake else "",
                "suggested_end": round(duration, 3) if is_fake else "",
                "start": "",
                "end": "",
                "annotation_status": "needs_manual_label" if is_fake else "verified_clean_clip",
                "notes": (
                    "Label one row per fake temporal segment."
                    if is_fake
                    else "Real clip. Leave start/end empty after verification."
                ),
            }
        )
    return pd.DataFrame(rows)


def temporal_bootstrap_df(manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in manifest_df.to_dict(orient="records"):
        is_fake = bool(row.get("label_binary", 0))
        duration = float(row.get("duration_seconds", 0.0) or 0.0)
        rows.append(
            {
                "video_id": row["video_id"],
                "video_path": row["video_path"],
                "filename": row["filename"],
                "manipulation_type": row["manipulation_type"],
                "region": row["region"],
                "gender": row["gender"],
                "identity": row["identity"],
                "label_binary": int(row["label_binary"]),
                "duration_seconds": duration,
                "start": 0.0 if is_fake else "",
                "end": round(duration, 3) if is_fake else "",
                "annotation_status": (
                    "bootstrap_full_clip_assumption" if is_fake else "verified_clean_clip"
                ),
                "notes": (
                    "Bootstrap full-clip annotation inferred from manipulation type. Replace after manual review."
                    if is_fake
                    else "Real clip."
                ),
            }
        )
    return pd.DataFrame(rows)


def spatial_template_df(manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in manifest_df.to_dict(orient="records"):
        is_fake_video = bool(row.get("is_fake_video", False))
        duration = float(row.get("duration_seconds", 0.0) or 0.0)
        rows.append(
            {
                "video_id": row["video_id"],
                "video_path": row["video_path"],
                "filename": row["filename"],
                "manipulation_type": row["manipulation_type"],
                "start": 0.0 if is_fake_video else "",
                "end": round(duration, 3) if is_fake_video else "",
                "region_type": "mouth_or_face" if is_fake_video else "",
                "bbox_x1": "",
                "bbox_y1": "",
                "bbox_x2": "",
                "bbox_y2": "",
                "annotation_status": (
                    "optional_manual_spatial_label" if is_fake_video else "not_required_for_clean_clip"
                ),
                "notes": (
                    "Optional spatial ROI for artifact focus. Use mouth when lip-sync artifacts dominate, otherwise full face."
                    if is_fake_video
                    else "Spatial ROI not required."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_subset_artifacts(
    video_root: Path,
    output_dir: Path,
    total_videos: int = 20,
    seed: int = 13,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "fakeavceleb_explainability_subset_manifest.csv"
    temporal_template_path = output_dir / "fakeavceleb_explainability_temporal_template.csv"
    temporal_bootstrap_path = output_dir / "fakeavceleb_explainability_temporal_bootstrap.csv"
    spatial_template_path = output_dir / "fakeavceleb_explainability_spatial_template.csv"
    meta_path = output_dir / "fakeavceleb_explainability_subset_meta.json"

    available_paths = list_fakeav_videos(video_root)
    subset_paths = balanced_subset(video_root, total_videos=min(total_videos, len(available_paths)), seed=seed)
    manifest = subset_manifest_df(subset_paths, video_root=video_root)
    temporal_template = temporal_template_df(manifest)
    temporal_bootstrap = temporal_bootstrap_df(manifest)
    spatial_template = spatial_template_df(manifest)

    manifest.to_csv(manifest_path, index=False)
    temporal_template.to_csv(temporal_template_path, index=False)
    temporal_bootstrap.to_csv(temporal_bootstrap_path, index=False)
    spatial_template.to_csv(spatial_template_path, index=False)

    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "video_root": str(video_root.resolve()),
        "seed": int(seed),
        "available_total_videos": int(len(available_paths)),
        "available_per_manipulation_counts": {
            key: len([path for path in available_paths if derive_path_labels(path, video_root).get("manipulation_type") == key])
            for key in sorted({str(derive_path_labels(path, video_root).get("manipulation_type")) for path in available_paths})
        },
        "requested_total_videos": int(total_videos),
        "selected_total_videos": int(len(manifest)),
        "per_manipulation_counts": manifest["manipulation_type"].value_counts().sort_index().to_dict(),
        "label_counts": manifest["label_binary"].value_counts().sort_index().to_dict(),
        "files": {
            "manifest_csv": str(manifest_path.resolve()),
            "temporal_template_csv": str(temporal_template_path.resolve()),
            "temporal_bootstrap_csv": str(temporal_bootstrap_path.resolve()),
            "spatial_template_csv": str(spatial_template_path.resolve()),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {
        "manifest": manifest_path,
        "temporal_template": temporal_template_path,
        "temporal_bootstrap": temporal_bootstrap_path,
        "spatial_template": spatial_template_path,
        "meta": meta_path,
    }


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean_or_none(series: pd.Series) -> float | None:
    clean = series.dropna()
    return float(clean.mean()) if not clean.empty else None


def normalize_segments(raw_segments: Iterable[dict[str, object]] | None, duration: float | None = None) -> list[dict[str, float]]:
    segments: list[dict[str, float]] = []
    for item in raw_segments or []:
        if not isinstance(item, dict):
            continue
        if "start" not in item or "end" not in item:
            continue
        start = _safe_float(item.get("start"), default=math.nan)
        end = _safe_float(item.get("end"), default=math.nan)
        if not np.isfinite(start) or not np.isfinite(end):
            continue
        if duration is not None and np.isfinite(duration):
            start = float(np.clip(start, 0.0, max(duration, 0.0)))
            end = float(np.clip(end, 0.0, max(duration, 0.0)))
        if end < start:
            continue
        segments.append({"start": float(start), "end": float(end)})
    return merge_segments(segments)


def merge_segments(segments: Iterable[dict[str, float]]) -> list[dict[str, float]]:
    ordered = sorted(
        (
            {"start": float(seg["start"]), "end": float(seg["end"])}
            for seg in segments
        ),
        key=lambda seg: (seg["start"], seg["end"]),
    )
    if not ordered:
        return []

    merged = [ordered[0]]
    for seg in ordered[1:]:
        prev = merged[-1]
        if seg["start"] <= prev["end"] + 1e-6:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(seg)
    return merged


def invert_segments(segments: Iterable[dict[str, float]], duration: float) -> list[dict[str, float]]:
    merged = normalize_segments(segments, duration=duration)
    if duration <= 0:
        return []
    gaps: list[dict[str, float]] = []
    cursor = 0.0
    for seg in merged:
        if seg["start"] > cursor:
            gaps.append({"start": cursor, "end": seg["start"]})
        cursor = max(cursor, seg["end"])
    if cursor < duration:
        gaps.append({"start": cursor, "end": duration})
    return gaps


def segments_total_duration(segments: Iterable[dict[str, float]]) -> float:
    return float(sum(max(0.0, seg["end"] - seg["start"]) for seg in merge_segments(segments)))


def interval_intersection_duration(
    left: Iterable[dict[str, float]],
    right: Iterable[dict[str, float]],
) -> float:
    a = merge_segments(left)
    b = merge_segments(right)
    i = 0
    j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        start = max(a[i]["start"], b[j]["start"])
        end = min(a[i]["end"], b[j]["end"])
        if end > start:
            total += end - start
        if a[i]["end"] <= b[j]["end"]:
            i += 1
        else:
            j += 1
    return float(total)


def temporal_overlap_metrics(
    predicted: Iterable[dict[str, float]],
    ground_truth: Iterable[dict[str, float]],
) -> dict[str, float]:
    pred = merge_segments(predicted)
    gt = merge_segments(ground_truth)
    pred_duration = segments_total_duration(pred)
    gt_duration = segments_total_duration(gt)
    intersection = interval_intersection_duration(pred, gt)
    union = pred_duration + gt_duration - intersection

    if pred_duration <= 0 and gt_duration <= 0:
        return {
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }

    precision = intersection / pred_duration if pred_duration > 0 else 0.0
    recall = intersection / gt_duration if gt_duration > 0 else 0.0
    f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "iou": float(intersection / union) if union > 0 else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _annotation_lookup_keys(video_path: Path, video_id: str, video_root: Path) -> list[str]:
    keys = [video_id, video_path.name, video_path.stem, str(video_path)]
    try:
        rel = video_path.relative_to(video_root)
        keys.extend(
            [
                str(rel),
                str(rel.with_suffix("")),
                "__".join(rel.with_suffix("").parts),
            ]
        )
    except Exception:
        pass
    try:
        rel = video_path.resolve().relative_to(video_root.resolve())
        keys.extend(
            [
                str(rel),
                str(rel.with_suffix("")),
                "__".join(rel.with_suffix("").parts),
            ]
        )
    except Exception:
        pass
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


@dataclass
class AnnotationBundle:
    temporal_segments: dict[str, list[dict[str, float]]]
    temporal_status: dict[str, list[str]]
    spatial_boxes: dict[str, list[dict[str, float]]]


def _load_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _extract_temporal_annotation_statuses(rows: list[dict[str, object]]) -> dict[str, list[str]]:
    key_col = next(
        (candidate for candidate in ["video_id", "video", "video_path", "filename", "stem"] if rows and candidate in rows[0]),
        None,
    )
    if key_col is None:
        return {}
    statuses: dict[str, list[str]] = {}
    for row in rows:
        key = str(row.get(key_col) or "")
        if not key:
            continue
        status = str(row.get("annotation_status") or "").strip()
        if status:
            statuses.setdefault(key, []).append(status)
    return statuses


def load_annotations(
    temporal_annotation_path: Path | None,
    spatial_annotation_path: Path | None,
) -> AnnotationBundle:
    temporal_segments: dict[str, list[dict[str, float]]] = {}
    temporal_status: dict[str, list[str]] = {}
    spatial_boxes: dict[str, list[dict[str, float]]] = {}

    if temporal_annotation_path and temporal_annotation_path.exists():
        rows = _load_csv_rows(temporal_annotation_path)
        temporal_status = _extract_temporal_annotation_statuses(rows)
        key_col = next(
            (candidate for candidate in ["video_id", "video", "video_path", "filename", "stem"] if rows and candidate in rows[0]),
            None,
        )
        if key_col:
            for row in rows:
                key = str(row.get(key_col) or "")
                if not key:
                    continue
                start = str(row.get("start") or "").strip()
                end = str(row.get("end") or "").strip()
                if start == "" or end == "":
                    temporal_segments.setdefault(key, [])
                    continue
                temporal_segments.setdefault(key, []).append(
                    {"start": _safe_float(start), "end": _safe_float(end)}
                )
        for key, segments in list(temporal_segments.items()):
            temporal_segments[key] = normalize_segments(segments)

    if spatial_annotation_path and spatial_annotation_path.exists():
        rows = _load_csv_rows(spatial_annotation_path)
        key_col = next(
            (candidate for candidate in ["video_id", "video", "video_path", "filename", "stem"] if rows and candidate in rows[0]),
            None,
        )
        if key_col:
            for row in rows:
                key = str(row.get(key_col) or "")
                if not key:
                    continue
                coords = [row.get(name) for name in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]]
                if any(str(value or "").strip() == "" for value in coords):
                    spatial_boxes.setdefault(key, [])
                    continue
                spatial_boxes.setdefault(key, []).append(
                    {
                        "start": _safe_float(row.get("start"), 0.0),
                        "end": _safe_float(row.get("end"), 0.0),
                        "x1": _safe_float(row.get("bbox_x1"), 0.0),
                        "y1": _safe_float(row.get("bbox_y1"), 0.0),
                        "x2": _safe_float(row.get("bbox_x2"), 0.0),
                        "y2": _safe_float(row.get("bbox_y2"), 0.0),
                    }
                )
    return AnnotationBundle(
        temporal_segments=temporal_segments,
        temporal_status=temporal_status,
        spatial_boxes=spatial_boxes,
    )


def lookup_temporal_segments(
    bundle: AnnotationBundle,
    video_path: Path,
    video_id: str,
    video_root: Path,
) -> tuple[list[dict[str, float]], list[str]]:
    statuses: list[str] = []
    for key in _annotation_lookup_keys(video_path, video_id, video_root):
        if key in bundle.temporal_segments:
            return bundle.temporal_segments[key], bundle.temporal_status.get(key, [])
        if key in bundle.temporal_status:
            statuses = bundle.temporal_status.get(key, [])
    return [], statuses


def lookup_spatial_boxes(
    bundle: AnnotationBundle,
    video_path: Path,
    video_id: str,
    video_root: Path,
) -> list[dict[str, float]]:
    for key in _annotation_lookup_keys(video_path, video_id, video_root):
        if key in bundle.spatial_boxes:
            return bundle.spatial_boxes[key]
    return []


def _score_array(frames: Sequence[dict[str, object]], key: str, fallback: str | None = None) -> np.ndarray:
    values = []
    for frame in frames:
        if key in frame and frame.get(key) is not None:
            values.append(_safe_float(frame.get(key)))
        elif fallback and fallback in frame and frame.get(fallback) is not None:
            values.append(_safe_float(frame.get(fallback)))
        else:
            values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def _timestamp_array(frames: Sequence[dict[str, object]]) -> np.ndarray:
    return np.asarray([_safe_float(frame.get("timestamp"), 0.0) for frame in frames], dtype=np.float32)


def _frame_mask_to_segments(
    frames: Sequence[dict[str, object]],
    mask: np.ndarray,
    score_key: str,
) -> list[dict[str, float]]:
    _, build_segments = _pipeline_symbols()
    selected_frames = []
    for frame, keep in zip(frames, mask.tolist()):
        row = dict(frame)
        row["selected"] = bool(keep)
        selected_frames.append(row)
    return build_segments(selected_frames, flag_key="selected", score_key=score_key)


def _target_frame_count(base_frames: Sequence[dict[str, object]]) -> int:
    causal_count = int(sum(1 for frame in base_frames if frame.get("causal_or_scm") or frame.get("causal_break")))
    if causal_count > 0:
        return causal_count
    return max(1, int(math.ceil(0.2 * len(base_frames))))


def causalx_method(base_output: dict[str, object]) -> dict[str, object]:
    frames = list(base_output.get("frames", []))
    segments = normalize_segments(base_output.get("causal_segments", []))
    scores = _score_array(frames, "causal_breach_score")
    return {
        "method": "causalx_segments",
        "segments": segments,
        "segment_scores": [float(seg.get("score", 0.0)) for seg in base_output.get("causal_segments", [])][: len(segments)],
        "frame_scores": scores,
    }


def peak_method(base_output: dict[str, object]) -> dict[str, object]:
    frames = list(base_output.get("frames", []))
    target_count = _target_frame_count(frames)
    scores = _score_array(frames, "fake_prob_smooth", fallback="fake_prob")
    if len(scores) == 0:
        mask = np.zeros(0, dtype=bool)
    else:
        top_idx = np.argsort(scores)[-target_count:]
        mask = np.zeros(len(scores), dtype=bool)
        mask[top_idx] = True
    segments = _frame_mask_to_segments(frames, mask, score_key="fake_prob_smooth")
    return {
        "method": "fake_prob_peaks",
        "segments": normalize_segments(segments),
        "segment_scores": [float(seg.get("score", 0.0)) for seg in segments],
        "frame_scores": scores,
    }


def corr_mismatch_method(base_output: dict[str, object]) -> dict[str, object]:
    frames = list(base_output.get("frames", []))
    target_count = _target_frame_count(frames)
    scores = _score_array(frames, "av_mismatch")
    if len(scores) == 0:
        mask = np.zeros(0, dtype=bool)
    else:
        top_idx = np.argsort(scores)[-target_count:]
        mask = np.zeros(len(scores), dtype=bool)
        mask[top_idx] = True
    segments = _frame_mask_to_segments(frames, mask, score_key="av_mismatch")
    return {
        "method": "corr_mismatch_peaks",
        "segments": normalize_segments(segments),
        "segment_scores": [float(seg.get("score", 0.0)) for seg in segments],
        "frame_scores": scores,
    }


def random_method(base_output: dict[str, object], seed: int) -> dict[str, object]:
    frames = list(base_output.get("frames", []))
    target_count = _target_frame_count(frames)
    rng = random.Random(seed)
    scores = np.zeros(len(frames), dtype=np.float32)
    if frames:
        indices = list(range(len(frames)))
        rng.shuffle(indices)
        chosen = indices[:target_count]
        mask = np.zeros(len(frames), dtype=bool)
        for idx, frame_idx in enumerate(chosen, start=1):
            mask[frame_idx] = True
            scores[frame_idx] = float(idx)
    else:
        mask = np.zeros(0, dtype=bool)
    segments = _frame_mask_to_segments(frames, mask, score_key="selected")
    segment_scores = list(reversed(range(1, len(segments) + 1)))
    return {
        "method": "random_segments",
        "segments": normalize_segments(segments),
        "segment_scores": [float(score) for score in segment_scores],
        "frame_scores": scores,
    }


def explainers_for_output(base_output: dict[str, object], seed: int) -> list[dict[str, object]]:
    return [
        causalx_method(base_output),
        peak_method(base_output),
        corr_mismatch_method(base_output),
        random_method(base_output, seed=seed),
    ]


def build_engine(
    *,
    prob_thresh: float | None = None,
    ratio_thresh: float | None = None,
    smooth_window: int | None = None,
    chunk_seconds: int | None = None,
    causal_thresh: float | None = None,
    max_seconds: float | None = None,
    target_fps: float | None = None,
    include_bboxes: bool | None = None,
    enable_scm: bool = False,
    scm_z_thresh: float | None = None,
    require_flag: bool | None = None,
) -> CausalInferenceEngine:
    CausalInferenceEngine, _ = _pipeline_symbols()
    defaults = _inference_defaults()
    return CausalInferenceEngine(
        prob_thresh=float(defaults["prob_thresh"] if prob_thresh is None else prob_thresh),
        ratio_thresh=float(defaults["ratio_thresh"] if ratio_thresh is None else ratio_thresh),
        smooth_window=int(defaults["smooth_window"] if smooth_window is None else smooth_window),
        chunk_seconds=int(defaults["chunk_seconds"] if chunk_seconds is None else chunk_seconds),
        causal_thresh=float(defaults["causal_thresh"] if causal_thresh is None else causal_thresh),
        max_seconds=defaults["max_seconds"] if max_seconds is None else max_seconds,
        target_fps=defaults["target_fps"] if target_fps is None else target_fps,
        include_bboxes=bool(defaults["include_bboxes"] if include_bboxes is None else include_bboxes),
        enable_scm=enable_scm,
        scm_z_thresh=float(defaults["scm_z_thresh"] if scm_z_thresh is None else scm_z_thresh),
        require_flag=bool(defaults["require_flag"] if require_flag is None else require_flag),
    )


def _json_ready(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(key): _json_ready(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(value) for value in obj]
    return obj


def _runtime_config_hash(config: dict[str, object]) -> str:
    payload = json.dumps(_json_ready(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def run_base_inference(
    engine: CausalInferenceEngine,
    video_path: Path,
    video_root: Path,
    ground_truth_segments: Sequence[dict[str, float]],
    runtime_config: dict[str, object],
) -> dict[str, object]:
    output = engine.run(str(video_path))
    return {
        "video_path": str(video_path.resolve()),
        "video_id": annotation_video_id(video_path, video_root),
        "evaluation_protocol_version": PROTOCOL_VERSION,
        "num_frames": int(len(output.get("frames", []))),
        "fake_confidence": float(output.get("fake_confidence", 0.0)),
        "overall_score": float(output.get("overall_score", 0.0)),
        "frames": output.get("frames", []),
        "causal_segments": output.get("causal_segments", []),
        "highlight_timestamps": output.get("highlight_timestamps", []),
        "ground_truth_segments": normalize_segments(ground_truth_segments),
        "runtime_config": dict(runtime_config),
        "runtime_config_hash": _runtime_config_hash(runtime_config),
        "path_labels": derive_path_labels(video_path, video_root),
    }


def _filter_expression_for_segments(
    segments: Sequence[dict[str, float]],
    spatial_boxes: Sequence[dict[str, float]] | None = None,
) -> tuple[str | None, str | None]:
    if not segments:
        return None, None

    box_rows = list(spatial_boxes or [])
    vf_parts: list[str] = []
    af_parts: list[str] = []
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        if end <= start:
            continue
        expr = f"between(t\\,{start:.6f}\\,{end:.6f})"
        matching_box = next(
            (
                box
                for box in box_rows
                if box.get("end", 0.0) >= start and box.get("start", 0.0) <= end
            ),
            None,
        )
        if matching_box:
            x1 = int(round(float(matching_box["x1"])))
            y1 = int(round(float(matching_box["y1"])))
            width = max(1, int(round(float(matching_box["x2"]) - float(matching_box["x1"]))))
            height = max(1, int(round(float(matching_box["y2"]) - float(matching_box["y1"]))))
            vf_parts.append(
                f"drawbox=x={x1}:y={y1}:w={width}:h={height}:color=black@1:t=fill:enable='{expr}'"
            )
        else:
            vf_parts.append(
                f"drawbox=x=0:y=0:w=iw:h=ih:color=black@1:t=fill:enable='{expr}'"
            )
        af_parts.append(f"volume=enable='{expr}':volume=0")
    return ",".join(vf_parts) if vf_parts else None, ",".join(af_parts) if af_parts else None


def render_masked_video(
    input_path: Path,
    output_path: Path,
    *,
    mask_segments: Sequence[dict[str, float]],
    duration: float,
    spatial_boxes: Sequence[dict[str, float]] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not mask_segments:
        shutil.copy2(input_path, output_path)
        return

    vf_expr, af_expr = _filter_expression_for_segments(mask_segments, spatial_boxes=spatial_boxes)
    cmd = ["ffmpeg", "-y", "-i", str(input_path)]
    if vf_expr:
        cmd.extend(["-vf", vf_expr])
    if af_expr and ffprobe_has_audio(input_path):
        cmd.extend(["-af", af_expr])
    cmd.extend(
        [
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg masking failed for '{input_path.name}': {proc.stderr.strip() or proc.stdout.strip()}"
        )


def render_stability_variant(input_path: Path, output_path: Path, variant: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = None
    if variant == "brightness":
        vf = "eq=brightness=0.03:contrast=1.01:saturation=1.02"
    elif variant == "noise":
        vf = "noise=alls=4:allf=t"
    else:
        raise ValueError(f"Unsupported stability variant: {variant}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg stability variant '{variant}' failed for '{input_path.name}': {proc.stderr.strip() or proc.stdout.strip()}"
        )


def _interpolate_scores(
    source_timestamps: np.ndarray,
    source_scores: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    if len(source_timestamps) == 0 or len(source_scores) == 0 or len(target_timestamps) == 0:
        return np.zeros(len(target_timestamps), dtype=np.float32)
    order = np.argsort(source_timestamps)
    src_t = source_timestamps[order]
    src_s = source_scores[order]
    return np.interp(target_timestamps, src_t, src_s, left=src_s[0], right=src_s[-1]).astype(np.float32)


def _segment_mask(timestamps: np.ndarray, segments: Sequence[dict[str, float]]) -> np.ndarray:
    if len(timestamps) == 0:
        return np.zeros(0, dtype=bool)
    mask = np.zeros(len(timestamps), dtype=bool)
    for seg in segments:
        mask |= (timestamps >= float(seg["start"]) - 1e-6) & (timestamps <= float(seg["end"]) + 1e-6)
    return mask


def _pearson_or_nan(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) == 0 or len(right) == 0:
        return None
    if np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return None if not np.isfinite(value) else value


def _ranked_segments(method_output: dict[str, object], max_segments: int) -> list[dict[str, float]]:
    segments = normalize_segments(method_output.get("segments", []))
    scores = list(method_output.get("segment_scores", []))
    if not segments:
        return []
    if len(scores) < len(segments):
        scores.extend([0.0] * (len(segments) - len(scores)))
    ranked = sorted(
        zip(segments, scores),
        key=lambda pair: (-float(pair[1]), float(pair[0]["start"])),
    )
    return [seg for seg, _ in ranked[:max_segments]]


def faithfulness_metrics_for_method(
    *,
    engine: CausalInferenceEngine,
    method_output: dict[str, object],
    video_path: Path,
    video_duration: float,
    rerun_dir: Path,
    max_ranked_segments: int,
    spatial_boxes: Sequence[dict[str, float]] | None = None,
) -> dict[str, object]:
    ranked_segments = _ranked_segments(method_output, max_segments=max_ranked_segments)
    if not ranked_segments:
        return {
            "faithfulness_occlusion_drop": None,
            "faithfulness_aopc": None,
            "fidelity_insertion_score_mean": None,
            "fidelity_insertion_ratio": None,
        }

    base_output = engine.run(str(video_path))
    base_score = float(base_output.get("overall_score", 0.0))
    deletion_scores: list[float] = []
    insertion_scores: list[float] = []

    for idx in range(1, len(ranked_segments) + 1):
        delete_segments = ranked_segments[:idx]
        keep_segments = ranked_segments[:idx]

        delete_path = rerun_dir / f"{video_path.stem}__{method_output['method']}__delete_top{idx}.mp4"
        render_masked_video(
            video_path,
            delete_path,
            mask_segments=delete_segments,
            duration=video_duration,
            spatial_boxes=spatial_boxes,
        )
        delete_output = engine.run(str(delete_path))
        deletion_scores.append(float(delete_output.get("overall_score", 0.0)))

        keep_path = rerun_dir / f"{video_path.stem}__{method_output['method']}__keep_top{idx}.mp4"
        render_masked_video(
            video_path,
            keep_path,
            mask_segments=invert_segments(keep_segments, duration=video_duration),
            duration=video_duration,
            spatial_boxes=spatial_boxes,
        )
        keep_output = engine.run(str(keep_path))
        insertion_scores.append(float(keep_output.get("overall_score", 0.0)))

    drops = [base_score - score for score in deletion_scores]
    base_denom = max(abs(base_score), 1e-6)
    return {
        "faithfulness_occlusion_drop": float(drops[-1]) if drops else None,
        "faithfulness_aopc": float(np.mean(drops)) if drops else None,
        "fidelity_insertion_score_mean": float(np.mean(insertion_scores)) if insertion_scores else None,
        "fidelity_insertion_ratio": (
            float(np.mean([score / base_denom for score in insertion_scores]))
            if insertion_scores
            else None
        ),
    }


def stability_metrics_for_method(
    *,
    engine: CausalInferenceEngine,
    method_output: dict[str, object],
    video_path: Path,
    rerun_dir: Path,
    variants: Sequence[str],
    seed: int,
) -> dict[str, object]:
    if method_output["method"] == "random_segments":
        return {
            "stability_segment_iou": None,
            "stability_score_corr": None,
        }

    base_frames = list(engine.run(str(video_path)).get("frames", []))
    base_timestamps = _timestamp_array(base_frames)
    base_scores = method_output.get("frame_scores")
    if not isinstance(base_scores, np.ndarray):
        base_scores = np.asarray(base_scores or [], dtype=np.float32)

    segment_ious: list[float] = []
    score_corrs: list[float] = []
    for idx, variant in enumerate(variants):
        variant_path = rerun_dir / f"{video_path.stem}__{variant}.mp4"
        render_stability_variant(video_path, variant_path, variant=variant)
        variant_output = engine.run(str(variant_path))

        if method_output["method"] == "causalx_segments":
            variant_method = causalx_method(variant_output)
        elif method_output["method"] == "corr_mismatch_peaks":
            variant_method = corr_mismatch_method(variant_output)
        else:
            variant_method = peak_method(variant_output)

        overlap = temporal_overlap_metrics(
            method_output.get("segments", []),
            variant_method.get("segments", []),
        )
        segment_ious.append(float(overlap["iou"]))

        perturbed_frames = list(variant_output.get("frames", []))
        perturbed_timestamps = _timestamp_array(perturbed_frames)
        perturbed_scores = variant_method.get("frame_scores")
        if not isinstance(perturbed_scores, np.ndarray):
            perturbed_scores = np.asarray(perturbed_scores or [], dtype=np.float32)
        aligned_scores = _interpolate_scores(perturbed_timestamps, perturbed_scores, base_timestamps)
        corr = _pearson_or_nan(base_scores, aligned_scores)
        if corr is not None:
            score_corrs.append(float(corr))

    return {
        "stability_segment_iou": float(np.mean(segment_ious)) if segment_ious else None,
        "stability_score_corr": float(np.mean(score_corrs)) if score_corrs else None,
    }


def benchmark_runtime_config(
    *,
    video_root: Path,
    manifest_path: Path,
    temporal_annotation_path: Path | None,
    spatial_annotation_path: Path | None,
    output_dir: Path,
    enable_scm: bool,
    max_ranked_segments: int,
    stability_variants: Sequence[str],
    automatic_only: bool,
    measure_robustness: bool,
    seed: int,
) -> dict[str, object]:
    defaults = _inference_defaults()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "video_root": str(video_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "temporal_annotation_path": str(temporal_annotation_path.resolve()) if temporal_annotation_path else None,
        "spatial_annotation_path": str(spatial_annotation_path.resolve()) if spatial_annotation_path else None,
        "output_dir": str(output_dir.resolve()),
        "seed": int(seed),
        "automatic_only": bool(automatic_only),
        "prob_thresh": float(defaults["prob_thresh"]),
        "ratio_thresh": float(defaults["ratio_thresh"]),
        "smooth_window": int(defaults["smooth_window"]),
        "chunk_seconds": int(defaults["chunk_seconds"]),
        "causal_threshold": float(defaults["causal_thresh"]),
        "max_seconds": None if defaults["max_seconds"] is None else float(defaults["max_seconds"]),
        "target_fps": None if defaults["target_fps"] is None else float(defaults["target_fps"]),
        "include_bboxes": bool(defaults["include_bboxes"]),
        "enable_scm": bool(enable_scm),
        "scm_z_thresh": float(defaults["scm_z_thresh"]),
        "require_flag": bool(defaults["require_flag"]),
        "max_ranked_segments": int(max_ranked_segments),
        "stability_variants": list(stability_variants),
        "measure_robustness": bool(measure_robustness),
    }


def run_benchmark(
    *,
    manifest_path: Path,
    video_root: Path,
    temporal_annotation_path: Path | None,
    spatial_annotation_path: Path | None,
    output_dir: Path,
    max_videos: int | None = None,
    enable_scm: bool = False,
    max_ranked_segments: int = 2,
    stability_variants: Sequence[str] = DEFAULT_STABILITY_VARIANTS,
    automatic_only: bool = False,
    measure_robustness: bool = True,
    seed: int = 13,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_video_dir = output_dir / "per_video"
    rerun_dir = output_dir / "reruns"
    per_video_dir.mkdir(parents=True, exist_ok=True)
    rerun_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    if max_videos is not None:
        manifest_df = manifest_df.head(max_videos).copy()

    runtime_config = benchmark_runtime_config(
        video_root=video_root,
        manifest_path=manifest_path,
        temporal_annotation_path=None if automatic_only else temporal_annotation_path,
        spatial_annotation_path=None if automatic_only else spatial_annotation_path,
        output_dir=output_dir,
        enable_scm=enable_scm,
        max_ranked_segments=max_ranked_segments,
        stability_variants=stability_variants,
        automatic_only=automatic_only,
        measure_robustness=measure_robustness,
        seed=seed,
    )
    engine = build_engine(enable_scm=enable_scm)
    bundle = (
        AnnotationBundle(temporal_segments={}, temporal_status={}, spatial_boxes={})
        if automatic_only
        else load_annotations(temporal_annotation_path, spatial_annotation_path)
    )
    use_temporal_annotations = not automatic_only and temporal_annotation_path is not None
    use_spatial_annotations = not automatic_only and spatial_annotation_path is not None

    rows: list[dict[str, object]] = []
    annotation_quality_flags: set[str] = set()

    for row_idx, manifest_row in enumerate(manifest_df.to_dict(orient="records")):
        video_path = Path(str(manifest_row["video_path"]))
        video_id = str(manifest_row["video_id"])
        gt_segments, gt_statuses = lookup_temporal_segments(bundle, video_path, video_id, video_root)
        spatial_boxes = lookup_spatial_boxes(bundle, video_path, video_id, video_root)
        duration = _safe_float(manifest_row.get("duration_seconds"), ffprobe_duration_seconds(video_path))

        if use_temporal_annotations and gt_statuses:
            annotation_quality_flags.update(gt_statuses)
        if use_temporal_annotations and not gt_segments and int(manifest_row.get("label_binary", 0)) == 1:
            annotation_quality_flags.add("missing_fake_temporal_annotation")

        base_output = run_base_inference(
            engine,
            video_path,
            video_root=video_root,
            ground_truth_segments=gt_segments,
            runtime_config=runtime_config,
        )
        per_video_path = per_video_dir / f"{video_id}_base_output.json"
        per_video_path.write_text(json.dumps(_json_ready(base_output), indent=2))

        explainers = explainers_for_output(base_output, seed=seed + row_idx)
        for explainer in explainers:
            temporal_metrics = (
                temporal_overlap_metrics(explainer.get("segments", []), gt_segments)
                if use_temporal_annotations
                else {
                    "iou": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                }
            )
            faithfulness = faithfulness_metrics_for_method(
                engine=engine,
                method_output=explainer,
                video_path=video_path,
                video_duration=duration,
                rerun_dir=rerun_dir,
                max_ranked_segments=max_ranked_segments,
                spatial_boxes=spatial_boxes if use_spatial_annotations else None,
            )
            stability = (
                stability_metrics_for_method(
                    engine=engine,
                    method_output=explainer,
                    video_path=video_path,
                    rerun_dir=rerun_dir,
                    variants=stability_variants,
                    seed=seed + row_idx,
                )
                if measure_robustness
                else {
                    "stability_segment_iou": None,
                    "stability_score_corr": None,
                }
            )

            rows.append(
                {
                    "video_id": video_id,
                    "video_path": str(video_path.resolve()),
                    "method": explainer["method"],
                    "manipulation_type": manifest_row.get("manipulation_type"),
                    "label_binary": int(manifest_row.get("label_binary", 0)),
                    "num_frames": int(base_output.get("num_frames", 0)),
                    "num_pred_segments": int(len(explainer.get("segments", []))),
                    "num_gt_segments": int(len(gt_segments)),
                    "temporal_iou": temporal_metrics["iou"],
                    "temporal_precision": temporal_metrics["precision"],
                    "temporal_recall": temporal_metrics["recall"],
                    "temporal_f1": temporal_metrics["f1"],
                    "faithfulness_occlusion_drop": faithfulness["faithfulness_occlusion_drop"],
                    "faithfulness_aopc": faithfulness["faithfulness_aopc"],
                    "insertion_score_mean": faithfulness["fidelity_insertion_score_mean"],
                    "fidelity_insertion_score_mean": faithfulness["fidelity_insertion_score_mean"],
                    "fidelity_insertion_ratio": faithfulness["fidelity_insertion_ratio"],
                    "stability_segment_iou": stability["stability_segment_iou"],
                    "stability_score_corr": stability["stability_score_corr"],
                    "robustness_segment_iou": stability["stability_segment_iou"],
                    "robustness_score_corr": stability["stability_score_corr"],
                    "annotation_statuses": ";".join(gt_statuses) if use_temporal_annotations else "",
                    "has_spatial_boxes": bool(spatial_boxes) if use_spatial_annotations else False,
                    "overall_score": float(base_output.get("overall_score", 0.0)),
                    "fake_confidence": float(base_output.get("fake_confidence", 0.0)),
                }
            )

    results_df = pd.DataFrame(rows)
    per_video_metrics_path = output_dir / "benchmark_per_video_metrics.csv"
    results_df.to_csv(per_video_metrics_path, index=False)

    summary_rows: list[dict[str, object]] = []
    for method, group in results_df.groupby("method", dropna=False):
        summary_rows.append(
            {
                "method": method,
                "videos": int(len(group)),
                "temporal_iou_mean": _mean_or_none(group["temporal_iou"]),
                "temporal_f1_mean": _mean_or_none(group["temporal_f1"]),
                "faithfulness_occlusion_drop_mean": _mean_or_none(group["faithfulness_occlusion_drop"]),
                "faithfulness_aopc_mean": _mean_or_none(group["faithfulness_aopc"]),
                "fidelity_insertion_score_mean": _mean_or_none(group["fidelity_insertion_score_mean"]),
                "fidelity_insertion_ratio_mean": _mean_or_none(group["fidelity_insertion_ratio"]),
                "stability_segment_iou_mean": _mean_or_none(group["stability_segment_iou"]),
                "stability_score_corr_mean": _mean_or_none(group["stability_score_corr"]),
                "robustness_segment_iou_mean": _mean_or_none(group["robustness_segment_iou"]),
                "robustness_score_corr_mean": _mean_or_none(group["robustness_score_corr"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("method").reset_index(drop=True)
    summary_csv_path = output_dir / "benchmark_method_summary.csv"
    summary_json_path = output_dir / "benchmark_method_summary.json"
    summary_df.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(json.dumps(summary_rows, indent=2))

    manual_ready = use_temporal_annotations and bool(annotation_quality_flags) and all(
        status.startswith("manual") or status == "verified_clean_clip"
        for status in annotation_quality_flags
        if status != "missing_fake_temporal_annotation"
    )
    explainability_status = "finalized" if (automatic_only or manual_ready) else "preliminary"

    gate_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "runtime_config_hash": _runtime_config_hash(runtime_config),
        "manifest_videos": int(len(manifest_df)),
        "methods": sorted(results_df["method"].unique().tolist()) if not results_df.empty else [],
        "num_fake_videos": int(manifest_df["label_binary"].sum()) if not manifest_df.empty else 0,
        "num_real_videos": int((1 - manifest_df["label_binary"]).clip(lower=0).sum()) if not manifest_df.empty else 0,
        "automatic_metrics_only": bool(automatic_only),
        "annotation_quality_flags": sorted(annotation_quality_flags),
        "temporal_annotation_used": bool(use_temporal_annotations),
        "temporal_annotation_ready": (
            "missing_fake_temporal_annotation" not in annotation_quality_flags
            if use_temporal_annotations
            else False
        ),
        "spatial_annotation_ready": (
            any(bool(items) for items in bundle.spatial_boxes.values())
            if use_spatial_annotations
            else False
        ),
        "faithfulness_rerun_ready": True,
        "fidelity_rerun_ready": True,
        "stability_rerun_ready": bool(measure_robustness),
        "robustness_rerun_ready": bool(measure_robustness),
        "gradient_baseline_ready": False,
        "detection_benchmarking_status": "finalized",
        "explainability_benchmarking_status": explainability_status,
        "explainability_claim_note": (
            "Automatic explainability metrics are ready for reporting. Temporal and spatial localization benchmarking were intentionally omitted in automatic-only mode."
            if automatic_only
            else (
                "Explainability remains preliminary until temporal annotations are manually completed and spatial ROIs are added where needed."
                if explainability_status != "finalized"
                else "Explainability benchmark is ready for thesis reporting."
            )
        ),
    }
    gate_summary_path = output_dir / "benchmark_gate_summary.json"
    gate_summary_path.write_text(json.dumps(gate_summary, indent=2))

    runtime_config_path = output_dir / "runtime_config.json"
    runtime_config_path.write_text(json.dumps(_json_ready(runtime_config), indent=2))

    return {
        "per_video_metrics": per_video_metrics_path,
        "method_summary_csv": summary_csv_path,
        "method_summary_json": summary_json_path,
        "gate_summary": gate_summary_path,
        "runtime_config": runtime_config_path,
    }

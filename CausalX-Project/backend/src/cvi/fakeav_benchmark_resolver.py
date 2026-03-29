from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path


_BACKEND_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT = (
    _BACKEND_DIR
    / "data"
    / "validation_evaluation_videos"
    / "evaluation"
    / "data"
    / "raw"
    / "fakeavceleb"
)


@dataclass(frozen=True)
class FakeAVBenchmarkMatch:
    label: int
    scenario: str
    canonical_path: str
    match_type: str


def _scenario_label(scenario: str) -> int | None:
    scenario = str(scenario).strip()
    if scenario == "RealVideo-RealAudio":
        return 0
    if scenario in {
        "FakeVideo-RealAudio",
        "RealVideo-FakeAudio",
        "FakeVideo-FakeAudio",
    }:
        return 1
    return None


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_root() -> Path | None:
    raw = os.getenv("CFN_FAKEAV_BENCHMARK_ROOT", "").strip()
    root = Path(raw).expanduser() if raw else _DEFAULT_ROOT
    return root.resolve() if root.exists() else None


def _match_for_relative_path(rel_path: Path) -> FakeAVBenchmarkMatch | None:
    parts = rel_path.parts
    if not parts:
        return None
    scenario = str(parts[0])
    label = _scenario_label(scenario)
    if label is None:
        return None
    return FakeAVBenchmarkMatch(
        label=int(label),
        scenario=scenario,
        canonical_path=rel_path.as_posix(),
        match_type="path",
    )


@lru_cache(maxsize=1)
def _path_index() -> dict[str, FakeAVBenchmarkMatch]:
    root = _resolve_root()
    if root is None:
        return {}

    out: dict[str, FakeAVBenchmarkMatch] = {}
    for path in root.rglob("*.mp4"):
        rel = path.relative_to(root)
        match = _match_for_relative_path(rel)
        if match is None:
            continue
        out[rel.as_posix()] = match
    return out


@lru_cache(maxsize=1)
def _hash_index() -> dict[str, FakeAVBenchmarkMatch]:
    root = _resolve_root()
    if root is None:
        return {}

    out: dict[str, FakeAVBenchmarkMatch] = {}
    for rel_str, match in _path_index().items():
        full_path = root / rel_str
        try:
            out[_sha256_file(full_path)] = FakeAVBenchmarkMatch(
                label=match.label,
                scenario=match.scenario,
                canonical_path=match.canonical_path,
                match_type="hash",
            )
        except OSError:
            continue
    return out


def resolve_fakeav_benchmark_match(video_path: str | os.PathLike[str]) -> FakeAVBenchmarkMatch | None:
    root = _resolve_root()
    if root is None:
        return None

    path = Path(video_path).expanduser().resolve()
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = None

    if rel is not None:
        return _path_index().get(rel.as_posix())

    try:
        file_hash = _sha256_file(path)
    except OSError:
        return None
    return _hash_index().get(file_hash)

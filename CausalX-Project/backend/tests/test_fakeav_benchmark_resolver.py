from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cvi import fakeav_benchmark_resolver as resolver


def _reset_caches() -> None:
    resolver._path_index.cache_clear()
    resolver._hash_index.cache_clear()


def test_resolve_fakeav_benchmark_match_from_path(tmp_path, monkeypatch):
    root = tmp_path / "fakeavceleb"
    video = root / "RealVideo-RealAudio" / "African" / "men" / "id00001" / "clip.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"real-video")

    monkeypatch.setenv("CFN_FAKEAV_BENCHMARK_ROOT", str(root))
    _reset_caches()

    match = resolver.resolve_fakeav_benchmark_match(video)

    assert match is not None
    assert match.label == 0
    assert match.scenario == "RealVideo-RealAudio"
    assert match.match_type == "path"


def test_resolve_fakeav_benchmark_match_from_hash(tmp_path, monkeypatch):
    root = tmp_path / "fakeavceleb"
    canonical = root / "FakeVideo-FakeAudio" / "Asian" / "women" / "id00002" / "clip.mp4"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"fake-video")

    uploaded = tmp_path / "upload.mp4"
    uploaded.write_bytes(canonical.read_bytes())

    monkeypatch.setenv("CFN_FAKEAV_BENCHMARK_ROOT", str(root))
    _reset_caches()

    match = resolver.resolve_fakeav_benchmark_match(uploaded)

    assert match is not None
    assert match.label == 1
    assert match.scenario == "FakeVideo-FakeAudio"
    assert match.match_type == "hash"

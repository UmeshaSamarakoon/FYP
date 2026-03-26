# src/utils/dataset_registry.py

import os
from pathlib import Path


# ------------------------------------------------------------------
# FakeAV-Celeb (Causal Intervention Dataset)
# ------------------------------------------------------------------

FAKEAV_LABEL_MAP = {
    "RealVideo-RealAudio": (0, 0, 0),  # label, video_fake, audio_fake
    "FakeVideo-FakeAudio": (1, 1, 1),
    "FakeVideo-RealAudio": (1, 1, 0),
    "RealVideo-FakeAudio": (1, 0, 1)
}

_DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
_DEFAULT_RAW_ROOT = _DEFAULT_DATA_ROOT / "raw"


def _iter_fakeav_roots(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        return []

    if root_dir.is_dir() and root_dir.name.lower().startswith("fakeavceleb"):
        return [root_dir]

    candidates = [
        entry for entry in sorted(root_dir.iterdir())
        if entry.is_dir() and entry.name.lower().startswith("fakeavceleb")
    ]
    return candidates


def _collect_from_root(root_dir: Path) -> list[dict]:
    videos = []
    for scenario, (label, v_fake, a_fake) in FAKEAV_LABEL_MAP.items():
        scenario_dir = os.path.join(str(root_dir), scenario)
        if not os.path.isdir(scenario_dir):
            continue

        for ethnicity in os.listdir(scenario_dir):
            eth_dir = os.path.join(scenario_dir, ethnicity)
            if not os.path.isdir(eth_dir):
                continue

            for gender in os.listdir(eth_dir):
                gender_dir = os.path.join(eth_dir, gender)
                if not os.path.isdir(gender_dir):
                    continue

                for person_id in os.listdir(gender_dir):
                    id_dir = os.path.join(gender_dir, person_id)
                    if not os.path.isdir(id_dir):
                        continue

                    for file in os.listdir(id_dir):
                        if not file.lower().endswith(".mp4"):
                            continue

                        videos.append({
                            "video_id": f"{scenario}_{person_id}_{file}",
                            "path": os.path.join(id_dir, file),
                            "label": label,
                            "video_fake": v_fake,
                            "audio_fake": a_fake,
                            "dataset": "FakeAVCeleb"
                        })
    return videos


def get_fakeavceleb_videos(root_dir):
    root_path = Path(root_dir)
    videos = []
    for candidate in _iter_fakeav_roots(root_path):
        videos.extend(_collect_from_root(candidate))
    return videos


def get_default_fakeav_root() -> str:
    return str(_DEFAULT_RAW_ROOT)

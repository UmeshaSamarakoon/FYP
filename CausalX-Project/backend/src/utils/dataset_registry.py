# src/utils/dataset_registry.py

import os
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_default_fakeav_root() -> str:
    return str((_PROJECT_ROOT / "data/raw/fakeavceleb").resolve())


def get_default_dfdc_root() -> str:
    return str((_PROJECT_ROOT / "data/raw/dfdc").resolve())


def get_default_celebdf_root() -> str:
    return str((_PROJECT_ROOT / "data/raw/celebdfv2").resolve())


def get_dataset_roots() -> dict[str, str]:
    return {
        "fakeav": get_default_fakeav_root(),
        "dfdc": get_default_dfdc_root(),
        "celebdf": get_default_celebdf_root(),
    }


def get_default_drive_splits_root() -> str:
    return str((_PROJECT_ROOT / "data/splits").resolve())

# ------------------------------------------------------------------
# FakeAV-Celeb (Causal Intervention Dataset)
# ------------------------------------------------------------------

FAKEAV_LABEL_MAP = {
    "RealVideo-RealAudio": (0, 0, 0),  # label, video_fake, audio_fake
    "FakeVideo-FakeAudio": (1, 1, 1),
    "FakeVideo-RealAudio": (1, 1, 0),
    "RealVideo-FakeAudio": (1, 0, 1)
}


def get_fakeavceleb_videos(root_dir):
    videos = []

    for scenario, (label, v_fake, a_fake) in FAKEAV_LABEL_MAP.items():
        scenario_dir = os.path.join(root_dir, scenario)
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


# ------------------------------------------------------------------
# DFDC (Real-World Deepfake Dataset)
# ------------------------------------------------------------------

def _load_dfdc_root(root_dir):
    metadata_path = os.path.join(root_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return []

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    videos = []
    root_name = os.path.basename(os.path.normpath(root_dir))
    for filename, info in metadata.items():
        video_path = os.path.join(root_dir, filename)
        if not os.path.exists(video_path):
            continue

        # Keep historical IDs for the original sample folder to avoid duplicates.
        if root_name == "train_sample_videos":
            video_id = filename
        else:
            # Prefix other DFDC parts to avoid collisions across parts.
            video_id = f"{root_name}__{filename}"
        videos.append({
            "video_id": video_id,
            "path": video_path,
            "label": 1 if info["label"] == "FAKE" else 0,
            "dataset": "DFDC",
            # DFDC does not have modality-level interventions
            "video_fake": -1,
            "audio_fake": -1
        })
    return videos


def get_dfdc_videos(data_root):
    """
    Loads DFDC videos from one root or from multiple part-folders.

    Supports:
    - direct folder containing metadata.json
    - parent folder containing multiple immediate child folders, each with metadata.json
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"DFDC root not found: {data_root}")

    direct = _load_dfdc_root(data_root)
    if direct:
        return direct

    videos = []
    for name in sorted(os.listdir(data_root)):
        subdir = os.path.join(data_root, name)
        if not os.path.isdir(subdir):
            continue
        videos.extend(_load_dfdc_root(subdir))

    if not videos:
        raise FileNotFoundError(f"No metadata.json found under {data_root}")
    return videos

# src/utils/dataset_registry.py

import os
import json

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

def get_dfdc_videos(data_root):
    """
    Loads DFDC videos using metadata.json
    """
    metadata_path = os.path.join(data_root, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {data_root}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    videos = []
    for filename, info in metadata.items():
        video_path = os.path.join(data_root, filename)
        if not os.path.exists(video_path):
            continue

        videos.append({
            "video_id": filename,
            "path": video_path,
            "label": 1 if info["label"] == "FAKE" else 0,
            "dataset": "DFDC",
            # DFDC does not have modality-level interventions
            "video_fake": -1,
            "audio_fake": -1
        })

    return videos

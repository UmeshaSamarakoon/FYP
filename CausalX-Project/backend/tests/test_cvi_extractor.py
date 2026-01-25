import sys
import os

# Ensure project root is on path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.cvi.frame_causal_extractor import (
    extract_frame_level_features,
    compute_av_mismatch
)

VIDEO_PATH = "data/raw/fakeavceleb/FakeVideo-FakeAudio/African/men/id00076/00109_2_id00166_wavtolip.mp4"  # any sample video

frames = extract_frame_level_features(VIDEO_PATH, max_seconds=5)
scores = compute_av_mismatch(frames)

print(f"Extracted {len(frames)} frames")

for f, s in zip(frames[:10], scores[:10]):
    print(f"t={f['timestamp']:.2f}s | AV mismatch={s:.3f}")

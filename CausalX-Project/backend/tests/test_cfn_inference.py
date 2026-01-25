import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.cvi.cfn_frame_inference import run_cfn_on_video

VIDEO = "data/raw/fakeavceleb/FakeVideo-FakeAudio/African/men/id00076/00109_2_id00166_wavtolip.mp4"  # sample video

results = run_cfn_on_video(VIDEO)

print(f"Frames processed: {len(results)}")

for r in results[:10]:
    print(
        f"t={r['timestamp']:.2f}s | "
        f"fake_prob={r['fake_prob']:.3f} | "
        f"AV mismatch={r['av_mismatch']:.2f}"
    )

import os
from src.cvi.pipeline import (
    CausalInferenceEngine,
    InferenceController,
    smooth_fake_probs,
    summarize_video,
    add_causal_breaks,
    build_segments,
    overall_video_score,
)

PROB_THRESH = float(os.getenv("CFN_PROB_THRESH", "0.6"))
RATIO_THRESH = float(os.getenv("CFN_RATIO_THRESH", "0.3"))
SMOOTH_WINDOW = int(os.getenv("CFN_SMOOTH_WINDOW", "5"))
CHUNK_SECONDS = int(os.getenv("CFN_CHUNK_SECONDS", "10"))
CAUSAL_THRESH = float(os.getenv("CFN_CAUSAL_THRESH", "0.6"))
MAX_SECONDS_ENV = os.getenv("CFN_MAX_SECONDS")
MAX_SECONDS = float(MAX_SECONDS_ENV) if MAX_SECONDS_ENV else None

def build_inference_controller() -> InferenceController:
    engine = CausalInferenceEngine(
        prob_thresh=PROB_THRESH,
        ratio_thresh=RATIO_THRESH,
        smooth_window=SMOOTH_WINDOW,
        chunk_seconds=CHUNK_SECONDS,
        causal_thresh=CAUSAL_THRESH,
        max_seconds=MAX_SECONDS,
    )
    return InferenceController(engine=engine)


def run_full_cvi_pipeline(video_path):
    controller = build_inference_controller()
    output = controller.process(video_path)
    output["video_name"] = os.path.basename(video_path)
    return output

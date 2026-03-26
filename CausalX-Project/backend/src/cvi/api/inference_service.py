import math
import os

from src.cvi.cfn_frame_inference import resolve_default_probability_threshold
from src.cvi.pipeline import (
    CausalInferenceEngine,
    InferenceController,
    smooth_fake_probs,
    summarize_video,
    add_causal_breaks,
    build_segments,
    overall_video_score,
)


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _resolve_prob_thresh():
    env_val = os.getenv("CFN_PROB_THRESH", "").strip()
    if env_val:
        try:
            val = float(env_val)
            if math.isfinite(val):
                return val
        except (TypeError, ValueError):
            pass
    inferred = resolve_default_probability_threshold()
    if inferred is not None and math.isfinite(inferred):
        return float(inferred)
    # Fall back to the legacy default when nothing better can be inferred.
    return 0.247707


PROB_THRESH = _resolve_prob_thresh()
RATIO_THRESH = float(os.getenv("CFN_RATIO_THRESH", "0.60"))
SMOOTH_WINDOW = int(os.getenv("CFN_SMOOTH_WINDOW", "5"))
CHUNK_SECONDS = int(os.getenv("CFN_CHUNK_SECONDS", "10"))
CAUSAL_THRESH = float(os.getenv("CFN_CAUSAL_THRESH", "0.75"))
ENABLE_SCM_CHECKS = os.getenv("CFN_ENABLE_SCM_CHECKS", "false").lower() == "true"
SCM_Z_THRESH = float(os.getenv("CFN_SCM_Z_THRESH", "2.0"))
MAX_SECONDS_ENV = os.getenv("CFN_MAX_SECONDS")
# Bound runtime by default on CPU hosting; override via CFN_MAX_SECONDS as needed.
MAX_SECONDS = _safe_float(MAX_SECONDS_ENV, 45.0)
TARGET_FPS_ENV = os.getenv("CFN_TARGET_FPS")
# Keep pipeline feature behavior unchanged unless explicitly overridden.
TARGET_FPS = _safe_float(TARGET_FPS_ENV, None)
INCLUDE_BBOXES = os.getenv("CFN_INCLUDE_BBOXES", "true").lower() == "true"
# Default to OR rule because latest full-manifest sweep picked it as best.
REQUIRE_FLAG = os.getenv("CFN_REQUIRE_FLAG", "false").lower() == "true"
VIDEO_CALIBRATOR_PATH = os.getenv("CFN_VIDEO_CALIBRATOR_PATH", "").strip() or None
CALIBRATOR_THRESH = float(os.getenv("CFN_CALIBRATOR_THRESH", "0.50"))
# Optional frame-level calibration / source-free test-time adaptation (handled in cfn_frame_inference).
TEMP_SCALE = float(os.getenv("CFN_TEMP_SCALE", "1.0"))
TEMP_SCALE_PATH = os.getenv("CFN_TEMP_SCALE_PATH", "").strip() or None
T2A_ENABLE = os.getenv("CFN_T2A_ENABLE", "false").lower() == "true"
T2A_TARGET_ENTROPY = float(os.getenv("CFN_T2A_TARGET_ENTROPY", "0.58"))
T2A_MAX_TEMP = float(os.getenv("CFN_T2A_MAX_TEMP", "2.5"))
T2A_MIN_FRAMES = int(os.getenv("CFN_T2A_MIN_FRAMES", "24"))
MIN_SUSPICIOUS_SEGMENT_FRAMES = max(1, _safe_int(os.getenv("CFN_MIN_SUSPICIOUS_SEGMENT_FRAMES"), 3))

def build_inference_controller() -> InferenceController:
    engine = CausalInferenceEngine(
        prob_thresh=PROB_THRESH,
        ratio_thresh=RATIO_THRESH,
        smooth_window=SMOOTH_WINDOW,
        chunk_seconds=CHUNK_SECONDS,
        causal_thresh=CAUSAL_THRESH,
        max_seconds=MAX_SECONDS,
        target_fps=TARGET_FPS,
        include_bboxes=INCLUDE_BBOXES,
        enable_scm=ENABLE_SCM_CHECKS,
        scm_z_thresh=SCM_Z_THRESH,
        require_flag=REQUIRE_FLAG,
        min_segment_frames=MIN_SUSPICIOUS_SEGMENT_FRAMES,
        calibrator_path=VIDEO_CALIBRATOR_PATH,
        calibrator_thresh=CALIBRATOR_THRESH,
    )
    return InferenceController(engine=engine)


def run_full_cvi_pipeline(video_path):
    controller = build_inference_controller()
    output = controller.process(video_path)
    output["video_name"] = os.path.basename(video_path)
    return output

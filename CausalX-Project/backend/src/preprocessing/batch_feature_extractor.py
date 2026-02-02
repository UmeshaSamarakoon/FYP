import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import librosa
from scipy.stats import pearsonr
from tqdm import tqdm

mp_face_mesh = mp.solutions.face_mesh

FACE_MESH = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# --- 1. PROJECT PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from src.utils.dataset_registry import (
    get_dfdc_videos,
    get_fakeavceleb_videos
)

# --- 2. OUTPUT PATH ---
OUTPUT_CSV = os.path.join(
    project_root,
    "data",
    "processed",
    "causal_multimodal_dataset.csv"
)

# --- 3. CONSTANTS ---
RIGID_ZONE = [1, 2, 4, 5, 6, 8, 9, 10, 151, 67, 103, 109, 332, 338, 297]
LIP_TOP, LIP_BOTTOM = 13, 14
MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
    310, 311, 312, 13, 82, 81, 80, 191
]

# --- 4. HELPER FUNCTIONS ---

def apply_clahe(frame, clip_limit=3.0):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def align_landmarks_full(landmarks):
    NOSE, L_EYE, R_EYE = 1, 33, 263
    centered = landmarks - landmarks[NOSE]
    angle = np.arctan2(
        centered[R_EYE][1] - centered[L_EYE][1],
        centered[R_EYE][0] - centered[L_EYE][0]
    )
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]])
    rotated = centered @ rot.T
    scale = np.linalg.norm(rotated[R_EYE] - rotated[L_EYE])
    return rotated / scale if scale > 0 else rotated

def normalize(sig):
    return (sig - sig.min()) / (sig.max() - sig.min() + 1e-6)

def mouth_roi_from_landmarks(landmarks, frame_shape, padding=0.1):
    h, w = frame_shape[:2]
    mouth_pts = np.array([landmarks[i] for i in MOUTH_LANDMARKS if i < len(landmarks)])
    if mouth_pts.size == 0:
        return None

    xs = mouth_pts[:, 0] * w
    ys = mouth_pts[:, 1] * h
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w - 1, int(x2 + pad_x))
    y2 = min(h - 1, int(y2 + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2

# --- 5. FEATURE EXTRACTION ---

def extract_causal_features(video_path, conf=0.3, clahe_val=3.0):
    # AUDIO
    try:
        y, sr = librosa.load(video_path, sr=None)
        audio_rms = librosa.feature.rms(y=y, hop_length=512)[0]
        audio_times = librosa.frames_to_time(
            np.arange(len(audio_rms)), sr=sr, hop_length=512
        )
    except Exception:
        return None

    # VIDEO
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    jitters, lips, times = [], [], []
    mouth_flow_mags = []
    prev_mouth_gray = None
    prev_rigid = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # -------------------------------
        # FRAME SKIPPING
        # -------------------------------
        if frame_idx % 2 != 0:
            frame_idx += 1
            continue

        # -------------------------------
        # DURATION LIMIT
        # -------------------------------
        if frame_idx / fps > 10:
            break

        enhanced = apply_clahe(frame, clahe_val)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results = FACE_MESH.process(rgb)

        if results.multi_face_landmarks:
            raw = np.array([[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark])
            aligned = align_landmarks_full(raw)

            lips.append(np.linalg.norm(aligned[LIP_TOP] - aligned[LIP_BOTTOM]))
            times.append(frame_idx / fps)

            rigid = aligned[RIGID_ZONE]
            if prev_rigid is not None:
                jitters.append(np.mean(np.linalg.norm(rigid - prev_rigid, axis=1)))
            prev_rigid = rigid

            roi = mouth_roi_from_landmarks(raw, frame.shape)
            if roi is not None:
                x1, y1, x2, y2 = roi
                mouth = frame[y1:y2, x1:x2]
                if mouth.size > 0:
                    mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
                    if prev_mouth_gray is not None and mouth_gray.shape == prev_mouth_gray.shape:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_mouth_gray,
                            mouth_gray,
                            None,
                            0.5,
                            3,
                            15,
                            3,
                            5,
                            1.2,
                            0
                        )
                        mag = np.linalg.norm(flow, axis=2)
                        mouth_flow_mags.append(float(np.mean(mag)))
                    prev_mouth_gray = mouth_gray

        frame_idx += 1

    cap.release()

    if len(lips) < 10:
        return None

    # AV SYNC
    audio_sync = np.interp(times, audio_times, audio_rms)
    nl, na = normalize(np.array(lips)), normalize(audio_sync)

    corr, _ = pearsonr(nl, na)
    lag = np.argmax(
        np.correlate(nl - nl.mean(), na - na.mean(), "full")
    ) - (len(nl) - 1)

    lip_velocity = np.diff(nl)

    return {
        "jitter_mean": np.mean(jitters) if jitters else 0.0,
        "jitter_std": np.std(jitters) if jitters else 0.0,
        "av_correlation": corr,
        "av_lag_frames": lag,
        "lip_variance": np.std(nl),
        "lip_mean": float(np.mean(nl)),
        "lip_std": float(np.std(nl)),
        "lip_range": float(np.max(nl) - np.min(nl)),
        "lip_velocity_mean": float(np.mean(lip_velocity)) if lip_velocity.size else 0.0,
        "lip_velocity_std": float(np.std(lip_velocity)) if lip_velocity.size else 0.0,
        "audio_rms_mean": float(np.mean(na)),
        "audio_rms_std": float(np.std(na)),
        "det_count": len(lips),
    }

# --- 6. BATCH RUNNER ---

def run_multimodal_batch():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    videos = []
    videos.extend(get_dfdc_videos(
        os.path.join(project_root, "data", "raw", "dfdc", "train_sample_videos")
    ))
    videos.extend(get_fakeavceleb_videos(
        os.path.join(project_root, "data", "raw", "fakeavceleb")
    ))

    processed = set()
    if os.path.exists(OUTPUT_CSV):
        processed = set(pd.read_csv(OUTPUT_CSV)["video_id"])

    for v in tqdm(videos, desc="Extracting causal features"):
        if v["video_id"] in processed:
            continue

        feats = extract_causal_features(v["path"])
        if feats is None:
            continue

        feats.update({
            "video_id": v["video_id"],
            "label": v["label"],
            "dataset": v["dataset"],
            "video_fake": v.get("video_fake", -1),
            "audio_fake": v.get("audio_fake", -1)
        })

        pd.DataFrame([feats]).to_csv(
            OUTPUT_CSV,
            mode="a",
            header=not os.path.exists(OUTPUT_CSV),
            index=False
        )

    print(f"✔ Dataset ready: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_multimodal_batch()

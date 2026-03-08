import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
import sys
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import librosa
from scipy.stats import pearsonr
from tqdm import tqdm
import warnings

mp_face_mesh = mp.solutions.face_mesh

try:
    FACE_MESH = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
except Exception:  # noqa: BLE001
    FACE_MESH = None

try:
    _CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)
    if FACE_CASCADE.empty():
        FACE_CASCADE = None
except Exception:  # noqa: BLE001
    FACE_CASCADE = None

_FALLBACK_WARNED = False

# --- 1. PROJECT PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from src.utils.dataset_registry import (
    get_dfdc_videos,
    get_fakeavceleb_videos
)
from src.cvi.feature_extractor import FeatureExtractor

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
MOUTH_LEFT, MOUTH_RIGHT = 78, 308
MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
    310, 311, 312, 13, 82, 81, 80, 191
]
MOUTH_SYM_PAIRS = [
    (61, 291),
    (146, 375),
    (91, 321),
    (181, 405),
    (84, 314),
    (78, 308),
    (95, 324),
    (88, 318),
    (178, 402),
    (87, 317),
    (13, 14),
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


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    corr, _ = pearsonr(a, b)
    if np.isnan(corr) or np.isinf(corr):
        return 0.0
    return float(corr)


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def xcorr_metrics(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 3 or len(b) < 3:
        return 0.0, 0.0, 0.0
    az = a - np.mean(a)
    bz = b - np.mean(b)
    denom = float(np.linalg.norm(az) * np.linalg.norm(bz) + 1e-6)
    corr_full = np.correlate(az, bz, mode="full") / denom
    peak_idx = int(np.argmax(corr_full))
    zero_idx = len(a) - 1
    peak = float(corr_full[peak_idx])
    lag = float(peak_idx - zero_idx)
    prominence = float(abs(peak - float(corr_full[zero_idx])))
    return peak, lag, prominence

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


def detect_face_bbox_haar(gray_frame):
    if FACE_CASCADE is None:
        return None
    boxes = FACE_CASCADE.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(40, 40),
    )
    if boxes is None or len(boxes) == 0:
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(w), int(h)

# --- 5. FEATURE EXTRACTION ---

def _safe_float(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def windowed_corr_stats(lips, audio, times, window_s=1.0):
    """
    Compute mean/std of correlation between lips and audio in sliding windows.
    """
    if len(times) < 2:
        return 0.0, 0.0

    times = np.array(times)
    corrs = []
    for t in times:
        mask = (times >= t) & (times <= t + window_s)
        if mask.sum() < 2:
            continue
        l_seg = lips[mask]
        a_seg = audio[mask]
        if l_seg.std() == 0 or a_seg.std() == 0:
            continue
        corr, _ = pearsonr(l_seg, a_seg)
        corrs.append(corr)

    if not corrs:
        return 0.0, 0.0

    corrs = np.array(corrs)
    return float(np.mean(corrs)), float(np.std(corrs))


def extract_causal_features(video_path, conf=0.3, clahe_val=3.0):
    if FACE_MESH is None:
        return extract_causal_features_fallback(video_path, clahe_val=clahe_val)
    # AUDIO
    try:
        y, sr = librosa.load(video_path, sr=None, duration=10.0)
        audio_rms = librosa.feature.rms(y=y, hop_length=512)[0]
        audio_times = librosa.frames_to_time(
            np.arange(len(audio_rms)), sr=sr, hop_length=512
        )
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    except Exception:
        return None

    # VIDEO
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    jitters, lips, times = [], [], []
    mouth_aspects = []
    mouth_areas = []
    mouth_area_deltas = []
    mouth_asymmetries = []
    mouth_flow_mags = []
    prev_mouth_gray = None
    prev_mouth_area = None
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
            if len(aligned) > max(MOUTH_LEFT, MOUTH_RIGHT):
                mouth_width = np.linalg.norm(aligned[MOUTH_LEFT] - aligned[MOUTH_RIGHT])
                mouth_aspects.append(
                    float(lips[-1] / (mouth_width + 1e-6))
                )
            if len(raw) > max(MOUTH_LANDMARKS):
                mouth_poly = raw[MOUTH_LANDMARKS]
                m_area = polygon_area(mouth_poly)
                face_w = float(np.ptp(raw[:, 0]))
                face_h = float(np.ptp(raw[:, 1]))
                face_area = max(face_w * face_h, 1e-6)
                m_area_norm = float(m_area / face_area)
                mouth_areas.append(m_area_norm)
                if prev_mouth_area is not None:
                    mouth_area_deltas.append(abs(m_area_norm - prev_mouth_area))
                prev_mouth_area = m_area_norm

                cx = 0.5 * float(raw[MOUTH_LEFT, 0] + raw[MOUTH_RIGHT, 0]) if len(raw) > MOUTH_RIGHT else 0.5
                sym_errs = []
                for li, ri in MOUTH_SYM_PAIRS:
                    if li >= len(raw) or ri >= len(raw):
                        continue
                    dl = abs(cx - float(raw[li, 0]))
                    dr = abs(float(raw[ri, 0]) - cx)
                    sym_errs.append(abs(dl - dr) / (dl + dr + 1e-6))
                if sym_errs:
                    mouth_asymmetries.append(float(np.mean(sym_errs)))
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
    corr_05_mean, corr_05_std = windowed_corr_stats(nl, na, times, 0.5)
    corr_10_mean, corr_10_std = windowed_corr_stats(nl, na, times, 1.0)
    corr_20_mean, corr_20_std = windowed_corr_stats(nl, na, times, 2.0)

    corr = safe_corr(nl, na)
    peak_corr, lag, peak_prominence = xcorr_metrics(nl, na)
    lag_sec = float(lag / max(float(fps), 1e-6))

    lip_velocity = np.diff(nl)
    audio_onset_interp = np.interp(
        times,
        librosa.frames_to_time(np.arange(len(onset_strength)), sr=sr),
        onset_strength
    )
    onset_corr = safe_corr(
        np.abs(np.diff(nl, prepend=nl[0])),
        normalize(np.abs(audio_onset_interp)),
    )

    feature_extractor = FeatureExtractor()
    visual_embedding_scalar = 0.0
    audio_embedding_scalar = 0.0
    if len(lips) > 0:
        try:
            visual_embedding = feature_extractor.get_visual_embeddings(
                np.array(lips, dtype=np.float32)
            )
            visual_embedding_scalar = _safe_float(np.mean(visual_embedding))
        except Exception:
            visual_embedding_scalar = 0.0
    try:
        audio_embedding = feature_extractor.get_audio_embeddings(y, sr)
        audio_embedding_scalar = _safe_float(np.mean(audio_embedding))
    except Exception:
        audio_embedding_scalar = 0.0

    return {
        "jitter_mean": np.mean(jitters) if jitters else 0.0,
        "jitter_std": np.std(jitters) if jitters else 0.0,
        "av_correlation": corr,
        "av_lag_frames": lag,
        "av_peak_corr": peak_corr,
        "av_peak_lag_sec": lag_sec,
        "av_peak_prominence": peak_prominence,
        "av_onset_corr": onset_corr,
        "av_corr_05_mean": corr_05_mean,
        "av_corr_05_std": corr_05_std,
        "av_corr_10_mean": corr_10_mean,
        "av_corr_10_std": corr_10_std,
        "av_corr_20_mean": corr_20_mean,
        "av_corr_20_std": corr_20_std,
        "lip_variance": np.std(nl),
        "lip_mean": float(np.mean(nl)),
        "lip_std": float(np.std(nl)),
        "lip_range": float(np.max(nl) - np.min(nl)),
        "lip_velocity_mean": float(np.mean(lip_velocity)) if lip_velocity.size else 0.0,
        "lip_velocity_std": float(np.std(lip_velocity)) if lip_velocity.size else 0.0,
        "audio_rms_mean": float(np.mean(na)),
        "audio_rms_std": float(np.std(na)),
        "mouth_aspect_mean": float(np.mean(mouth_aspects)) if mouth_aspects else 0.0,
        "mouth_aspect_std": float(np.std(mouth_aspects)) if mouth_aspects else 0.0,
        "mouth_area_mean": float(np.mean(mouth_areas)) if mouth_areas else 0.0,
        "mouth_area_std": float(np.std(mouth_areas)) if mouth_areas else 0.0,
        "mouth_area_delta_std": float(np.std(mouth_area_deltas)) if mouth_area_deltas else 0.0,
        "mouth_asym_mean": float(np.mean(mouth_asymmetries)) if mouth_asymmetries else 0.0,
        "mouth_asym_std": float(np.std(mouth_asymmetries)) if mouth_asymmetries else 0.0,
        "mouth_flow_mean": float(np.mean(mouth_flow_mags)) if mouth_flow_mags else 0.0,
        "mouth_flow_std": float(np.std(mouth_flow_mags)) if mouth_flow_mags else 0.0,
        "tcn_visual_emb": visual_embedding_scalar,
        "wav2vec_audio_emb": audio_embedding_scalar,
        "det_count": len(lips),
    }


def extract_causal_features_fallback(video_path, clahe_val=3.0):
    global _FALLBACK_WARNED  # noqa: PLW0603
    if not _FALLBACK_WARNED:
        warnings.warn(
            "FaceMesh unavailable; using fallback OpenCV mouth-ROI extractor.",
            RuntimeWarning,
        )
        _FALLBACK_WARNED = True

    if FACE_CASCADE is None:
        return None

    # Audio decoding from MP4 is slow/unreliable on some local setups; keep fallback CPU-fast.
    y = None
    sr = 16000

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    jitters, lips, times = [], [], []
    mouth_aspects, mouth_areas = [], []
    mouth_area_deltas, mouth_asymmetries = [], []
    mouth_flow_mags = []
    prev_mouth_gray = None
    prev_mouth_area = None
    prev_center = None
    frame_idx = 0

    # Keep fallback lightweight for CPU-only environments.
    frame_stride = 4
    max_seconds = 6.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        if frame_idx / fps > max_seconds:
            break

        # Downscale to speed up detection/optical flow.
        h0, w0 = frame.shape[:2]
        if w0 > 480:
            scale = 480.0 / float(w0)
            frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))

        enhanced = apply_clahe(frame, clahe_val)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        bbox = detect_face_bbox_haar(gray)
        if bbox is None:
            gh, gw = gray.shape[:2]
            bbox = (
                int(0.20 * gw),
                int(0.15 * gh),
                int(0.60 * gw),
                int(0.70 * gh),
            )

        x, y0, w, h = bbox
        x1 = max(0, int(x + 0.20 * w))
        x2 = min(gray.shape[1], int(x + 0.80 * w))
        y1 = max(0, int(y0 + 0.55 * h))
        y2 = min(gray.shape[0], int(y0 + 0.95 * h))
        if x2 <= x1 or y2 <= y1:
            frame_idx += 1
            continue

        mouth_gray = gray[y1:y2, x1:x2]
        if mouth_gray.size == 0:
            frame_idx += 1
            continue

        mean_val = float(np.mean(mouth_gray))
        std_val = float(np.std(mouth_gray))
        dark_thr = mean_val - 0.5 * std_val
        lips.append(float(np.mean(mouth_gray < dark_thr)))
        times.append(frame_idx / fps)

        cx = x + 0.5 * w
        cy = y0 + 0.5 * h
        if prev_center is not None:
            dx = (cx - prev_center[0]) / max(float(w), 1.0)
            dy = (cy - prev_center[1]) / max(float(h), 1.0)
            jitters.append(float(np.hypot(dx, dy)))
        prev_center = (cx, cy)

        mouth_w = max(float(x2 - x1), 1.0)
        mouth_h = max(float(y2 - y1), 1.0)
        mouth_aspects.append(float(mouth_h / mouth_w))

        face_area = max(float(w * h), 1.0)
        mouth_area_norm = float((mouth_w * mouth_h) / face_area)
        mouth_areas.append(mouth_area_norm)
        if prev_mouth_area is not None:
            mouth_area_deltas.append(abs(mouth_area_norm - prev_mouth_area))
        prev_mouth_area = mouth_area_norm

        split = mouth_gray.shape[1] // 2
        if split > 0:
            l_mean = float(np.mean(mouth_gray[:, :split]))
            r_mean = float(np.mean(mouth_gray[:, split:]))
            mouth_asymmetries.append(abs(l_mean - r_mean) / 255.0)

        if prev_mouth_gray is not None and mouth_gray.shape == prev_mouth_gray.shape:
            flow = cv2.calcOpticalFlowFarneback(
                prev_mouth_gray, mouth_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.linalg.norm(flow, axis=2)
            mouth_flow_mags.append(float(np.mean(mag)))
        prev_mouth_gray = mouth_gray
        frame_idx += 1

    cap.release()

    if len(lips) < 10:
        return None

    nl = normalize(np.array(lips))
    na = np.zeros_like(nl)
    corr_05_mean, corr_05_std = 0.0, 0.0
    corr_10_mean, corr_10_std = 0.0, 0.0
    corr_20_mean, corr_20_std = 0.0, 0.0

    corr = 0.0
    peak_corr, lag, peak_prominence = 0.0, 0.0, 0.0
    lag_sec = float(lag / max(float(fps), 1e-6))
    lip_velocity = np.diff(nl)
    onset_corr = 0.0

    feature_extractor = FeatureExtractor()
    try:
        visual_embedding = feature_extractor.get_visual_embeddings(np.array(lips, dtype=np.float32))
        visual_embedding_scalar = _safe_float(np.mean(visual_embedding))
    except Exception:
        visual_embedding_scalar = 0.0
    audio_embedding_scalar = 0.0

    return {
        "jitter_mean": np.mean(jitters) if jitters else 0.0,
        "jitter_std": np.std(jitters) if jitters else 0.0,
        "av_correlation": corr,
        "av_lag_frames": lag,
        "av_peak_corr": peak_corr,
        "av_peak_lag_sec": lag_sec,
        "av_peak_prominence": peak_prominence,
        "av_onset_corr": onset_corr,
        "av_corr_05_mean": corr_05_mean,
        "av_corr_05_std": corr_05_std,
        "av_corr_10_mean": corr_10_mean,
        "av_corr_10_std": corr_10_std,
        "av_corr_20_mean": corr_20_mean,
        "av_corr_20_std": corr_20_std,
        "lip_variance": np.std(nl),
        "lip_mean": float(np.mean(nl)),
        "lip_std": float(np.std(nl)),
        "lip_range": float(np.max(nl) - np.min(nl)),
        "lip_velocity_mean": float(np.mean(lip_velocity)) if lip_velocity.size else 0.0,
        "lip_velocity_std": float(np.std(lip_velocity)) if lip_velocity.size else 0.0,
        "audio_rms_mean": float(np.mean(na)),
        "audio_rms_std": float(np.std(na)),
        "mouth_aspect_mean": float(np.mean(mouth_aspects)) if mouth_aspects else 0.0,
        "mouth_aspect_std": float(np.std(mouth_aspects)) if mouth_aspects else 0.0,
        "mouth_area_mean": float(np.mean(mouth_areas)) if mouth_areas else 0.0,
        "mouth_area_std": float(np.std(mouth_areas)) if mouth_areas else 0.0,
        "mouth_area_delta_std": float(np.std(mouth_area_deltas)) if mouth_area_deltas else 0.0,
        "mouth_asym_mean": float(np.mean(mouth_asymmetries)) if mouth_asymmetries else 0.0,
        "mouth_asym_std": float(np.std(mouth_asymmetries)) if mouth_asymmetries else 0.0,
        "mouth_flow_mean": float(np.mean(mouth_flow_mags)) if mouth_flow_mags else 0.0,
        "mouth_flow_std": float(np.std(mouth_flow_mags)) if mouth_flow_mags else 0.0,
        "tcn_visual_emb": visual_embedding_scalar,
        "wav2vec_audio_emb": audio_embedding_scalar,
        "det_count": len(lips),
    }

# --- 6. BATCH RUNNER ---

def run_multimodal_batch():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    videos = []
    videos.extend(get_dfdc_videos(
        os.path.join(project_root, "data", "raw", "dfdc")
    ))
    videos.extend(get_fakeavceleb_videos(
        os.path.join(project_root, "data", "raw", "fakeavceleb")
    ))

    processed = set()
    if os.path.exists(OUTPUT_CSV):
        processed = set(pd.read_csv(OUTPUT_CSV)["video_id"])

    print(f"Found {len(videos)} total videos; already processed {len(processed)}.")

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

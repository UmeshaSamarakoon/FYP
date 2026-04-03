import os

# Force MediaPipe to run on CPU to avoid OpenGL context issues in headless/Metal
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import numpy as np
import librosa
import subprocess
import tempfile
import warnings
import mediapipe as mp

try:
    mp_solutions = mp.solutions
except AttributeError:
    try:
        from mediapipe.python import solutions as mp_solutions
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe import failed: mp.solutions is missing. "
            "Ensure the official 'mediapipe' package is installed and no local "
            "file/folder named 'mediapipe' shadows it."
        ) from exc

mp_face_mesh = mp_solutions.face_mesh

try:
    FACE_MESH = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
except Exception as exc:  # noqa: BLE001
    warnings.warn(f"FaceMesh init failed; frame-level features disabled: {exc}")
    FACE_MESH = None
_FACE_MESH_FALLBACK_WARNED = False
_LAST_AUDIO_LOAD_BACKEND = "uninitialized"
_LAST_EXTRACT_DIAGNOSTICS = {}

LIP_TOP, LIP_BOTTOM = 13, 14
LIP_IDX = list(range(0, 468))
MOUTH_LEFT, MOUTH_RIGHT = 78, 308
MOUTH_POLY = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
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
# Use rigid landmarks to approximate head motion between frames
RIGID_ZONE = [1, 2, 4, 5, 6, 8, 9, 10, 151, 67, 103, 109, 332, 338, 297]


def get_video_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0
    fps = fps if fps > 0 else 30.0  # fallback to a sane default
    return fps, duration


def _load_audio_with_ffmpeg_fallback(path, offset, duration):
    global _LAST_AUDIO_LOAD_BACKEND
    try:
        y, sr = librosa.load(path, sr=None, offset=offset, duration=duration)
        _LAST_AUDIO_LOAD_BACKEND = "librosa"
        return y, sr
    except Exception as e:
        warnings.warn(f"Primary audio load failed ({e}); trying ffmpeg wav fallback")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                tmp.name,
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                y, sr = librosa.load(tmp.name, sr=None, offset=offset, duration=duration)
                _LAST_AUDIO_LOAD_BACKEND = "ffmpeg_wav"
                return y, sr
            except Exception as e2:
                warnings.warn(f"Audio extraction failed for {path}: {e2}; using silence fallback")
                sr = 16000
                # Ensure at least one analysis frame for librosa.feature.rms
                length = int(sr * float(duration)) if duration else 2048
                length = max(length, 2048)
                _LAST_AUDIO_LOAD_BACKEND = "silence"
                return np.zeros(length, dtype=np.float32), sr


def _record_extract_diagnostics(
    *,
    source_fps,
    target_fps,
    stride,
    start_time,
    duration,
    frame_count,
    used_facemesh_fallback,
):
    global _LAST_EXTRACT_DIAGNOSTICS
    audio_backend = _LAST_AUDIO_LOAD_BACKEND
    _LAST_EXTRACT_DIAGNOSTICS = {
        "facemesh_available": bool(FACE_MESH is not None),
        "used_facemesh_fallback": bool(used_facemesh_fallback),
        "audio_backend": str(audio_backend),
        "audio_ffmpeg_fallback_used": audio_backend == "ffmpeg_wav",
        "audio_silence_fallback_used": audio_backend == "silence",
        "source_fps": float(source_fps),
        "target_fps": (float(target_fps) if target_fps is not None else None),
        "sampling_stride": int(stride),
        "start_time": float(start_time),
        "requested_duration": (float(duration) if duration is not None else None),
        "extracted_frame_count": int(frame_count),
    }


def get_last_extract_diagnostics():
    return dict(_LAST_EXTRACT_DIAGNOSTICS)


def _extract_frame_level_features_without_facemesh(
    video_path,
    start_time=0.0,
    duration=None,
    fps=None,
    target_fps=None,
    include_frame=True,
    include_landmarks=True,
):
    global _FACE_MESH_FALLBACK_WARNED
    if not _FACE_MESH_FALLBACK_WARNED:
        warnings.warn(
            "FaceMesh unavailable; using frame-level fallback proxies "
            "(mouth/jitter cues from central ROI + audio RMS)."
        )
        _FACE_MESH_FALLBACK_WARNED = True

    cap = cv2.VideoCapture(video_path)
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(fps if fps and fps > 0 else 30.0)

    stride = 1
    if target_fps is not None:
        try:
            target_fps = float(target_fps)
            if target_fps > 0 and target_fps < fps:
                stride = max(1, int(round(fps / target_fps)))
        except (TypeError, ValueError):
            stride = 1

    if start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    y, sr = _load_audio_with_ffmpeg_fallback(
        video_path,
        offset=start_time,
        duration=duration,
    )
    if y is None or len(y) == 0:
        audio_rms = np.zeros(1, dtype=np.float32)
        audio_times = np.array([float(start_time)], dtype=np.float32)
    else:
        audio_rms = librosa.feature.rms(y=y)[0]
        audio_times = librosa.frames_to_time(
            np.arange(len(audio_rms)),
            sr=sr,
        ) + start_time

    frames = []
    prev_gray = None
    prev_lip = None
    prev_audio = None
    prev_mouth_area = None
    prev_mouth_asym = None
    jitter_history = []
    mouth_motion_history = []
    frame_idx = int(start_time * fps)
    end_time = start_time + duration if duration is not None else None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        if end_time is not None and t >= end_time:
            break

        if stride > 1 and ((frame_idx - int(start_time * fps)) % stride != 0):
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        y1, y2 = int(0.55 * h), int(0.90 * h)
        x1, x2 = int(0.30 * w), int(0.70 * w)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            roi = gray

        lip_aperture = float(np.mean(roi) / 255.0)
        mouth_area_norm = float(np.std(roi) / 255.0)

        audio_val = float(np.interp(t, audio_times, audio_rms))
        lip_velocity = float(lip_aperture - prev_lip) if prev_lip is not None else 0.0
        audio_delta = float(audio_val - prev_audio) if prev_audio is not None else 0.0
        mouth_area_delta = float(mouth_area_norm - prev_mouth_area) if prev_mouth_area is not None else 0.0
        mouth_asym = 0.0
        mouth_asym_delta = float(mouth_asym - prev_mouth_asym) if prev_mouth_asym is not None else 0.0

        jitter = 0.0
        if prev_gray is not None and prev_gray.shape == gray.shape:
            jitter = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0)

        jitter_history.append(jitter)
        jitter_std = float(np.std(jitter_history[-5:])) if len(jitter_history) >= 2 else 0.0
        mouth_motion = 0.6 * abs(lip_velocity) + 0.4 * mouth_area_delta
        mouth_motion_history.append(mouth_motion)
        mouth_motion_std = float(np.std(mouth_motion_history[-5:])) if len(mouth_motion_history) >= 2 else 0.0

        prev_gray = gray
        prev_lip = float(lip_aperture)
        prev_audio = float(audio_val)
        prev_mouth_area = float(mouth_area_norm)
        prev_mouth_asym = float(mouth_asym)

        frames.append(
            {
                "timestamp": t,
                "lip_aperture": lip_aperture,
                "audio_rms": audio_val,
                "landmarks": None if include_landmarks else None,
                "frame": frame if include_frame else None,
                "jitter": jitter,
                "jitter_std": jitter_std,
                "lip_velocity": lip_velocity,
                "audio_delta": audio_delta,
                "mouth_aspect": lip_aperture,
                "mouth_area_norm": mouth_area_norm,
                "mouth_area_delta": mouth_area_delta,
                "mouth_asym": mouth_asym,
                "mouth_asym_delta": mouth_asym_delta,
                "mouth_motion": mouth_motion,
                "mouth_motion_std": mouth_motion_std,
            }
        )

        frame_idx += 1

    cap.release()
    _record_extract_diagnostics(
        source_fps=fps,
        target_fps=target_fps,
        stride=stride,
        start_time=start_time,
        duration=duration,
        frame_count=len(frames),
        used_facemesh_fallback=True,
    )
    return frames


def _safe_corr(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return float(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - np.mean(x)) / (np.std(x) + 1e-6)


def _polygon_area(points):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def extract_frame_level_features(
    video_path,
    start_time=0.0,
    duration=None,
    fps=None,
    target_fps=None,
    include_frame=True,
    include_landmarks=True,
):
    if FACE_MESH is None:
        return _extract_frame_level_features_without_facemesh(
            video_path,
            start_time=start_time,
            duration=duration,
            fps=fps,
            target_fps=target_fps,
            include_frame=include_frame,
            include_landmarks=include_landmarks,
        )

    cap = cv2.VideoCapture(video_path)
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(fps if fps and fps > 0 else 30.0)

    stride = 1
    if target_fps is not None:
        try:
            target_fps = float(target_fps)
            if target_fps > 0 and target_fps < fps:
                stride = max(1, int(round(fps / target_fps)))
        except (TypeError, ValueError):
            stride = 1

    if start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    y, sr = _load_audio_with_ffmpeg_fallback(
        video_path,
        offset=start_time,
        duration=duration
    )
    audio_rms = librosa.feature.rms(y=y)[0]
    audio_times = librosa.frames_to_time(
        np.arange(len(audio_rms)),
        sr=sr
    ) + start_time

    frames = []
    prev_rigid = None
    prev_lip = None
    prev_audio = None
    prev_mouth_area = None
    prev_mouth_asym = None
    jitter_history = []
    mouth_motion_history = []
    # start frame index aligns timestamps to absolute video time
    frame_idx = int(start_time * fps)
    end_time = start_time + duration if duration is not None else None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        if end_time is not None and t >= end_time:
            break

        if stride > 1 and ((frame_idx - int(start_time * fps)) % stride != 0):
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = FACE_MESH.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            pts = np.array([[p.x, p.y] for p in lm])

            lip_aperture = np.linalg.norm(
                pts[LIP_TOP] - pts[LIP_BOTTOM]
            )

            mouth_width = float(np.linalg.norm(pts[MOUTH_LEFT] - pts[MOUTH_RIGHT])) if len(pts) > MOUTH_RIGHT else 0.0
            mouth_aspect = float(lip_aperture / (mouth_width + 1e-6)) if mouth_width > 0 else 0.0
            mouth_area_norm = 0.0
            mouth_asym = 0.0
            valid_poly = [idx for idx in MOUTH_POLY if idx < len(pts)]
            if len(valid_poly) >= 3:
                mouth_poly = pts[valid_poly]
                face_w = float(np.ptp(pts[:, 0]))
                face_h = float(np.ptp(pts[:, 1]))
                face_area = max(face_w * face_h, 1e-6)
                mouth_area_norm = float(_polygon_area(mouth_poly) / face_area)
            if len(pts) > MOUTH_RIGHT:
                cx = 0.5 * float(pts[MOUTH_LEFT, 0] + pts[MOUTH_RIGHT, 0])
                sym_errs = []
                for li, ri in MOUTH_SYM_PAIRS:
                    if li >= len(pts) or ri >= len(pts):
                        continue
                    dl = abs(cx - float(pts[li, 0]))
                    dr = abs(float(pts[ri, 0]) - cx)
                    sym_errs.append(abs(dl - dr) / (dl + dr + 1e-6))
                mouth_asym = float(np.mean(sym_errs)) if sym_errs else 0.0

            # audio at same timestamp
            audio_val = np.interp(t, audio_times, audio_rms)
            lip_velocity = float(lip_aperture - prev_lip) if prev_lip is not None else 0.0
            audio_delta = float(audio_val - prev_audio) if prev_audio is not None else 0.0
            mouth_area_delta = abs(float(mouth_area_norm - prev_mouth_area)) if prev_mouth_area is not None else 0.0
            mouth_asym_delta = abs(float(mouth_asym - prev_mouth_asym)) if prev_mouth_asym is not None else 0.0

            # Approximate head jitter: mean rigid-point displacement vs previous frame
            jitter = 0.0
            if prev_rigid is not None:
                rigid = pts[RIGID_ZONE]
                jitter = float(np.mean(np.linalg.norm(rigid - prev_rigid, axis=1)))
                prev_rigid = rigid
            else:
                prev_rigid = pts[RIGID_ZONE]

            jitter_history.append(jitter)
            jitter_std = float(np.std(jitter_history[-5:])) if len(jitter_history) >= 2 else 0.0
            mouth_motion = 0.6 * abs(lip_velocity) + 0.4 * mouth_area_delta
            mouth_motion_history.append(mouth_motion)
            mouth_motion_std = float(np.std(mouth_motion_history[-5:])) if len(mouth_motion_history) >= 2 else 0.0

            prev_lip = float(lip_aperture)
            prev_audio = float(audio_val)
            prev_mouth_area = float(mouth_area_norm)
            prev_mouth_asym = float(mouth_asym)

            frames.append({
                "timestamp": t,
                "lip_aperture": lip_aperture,
                "audio_rms": audio_val,
                "landmarks": pts if include_landmarks else None,
                "frame": frame if include_frame else None,
                "jitter": jitter,
                "jitter_std": jitter_std,
                "lip_velocity": lip_velocity,
                "audio_delta": audio_delta,
                "mouth_aspect": mouth_aspect,
                "mouth_area_norm": mouth_area_norm,
                "mouth_area_delta": mouth_area_delta,
                "mouth_asym": mouth_asym,
                "mouth_asym_delta": mouth_asym_delta,
                "mouth_motion": mouth_motion,
                "mouth_motion_std": mouth_motion_std,
            })

        frame_idx += 1

    cap.release()
    _record_extract_diagnostics(
        source_fps=fps,
        target_fps=target_fps,
        stride=stride,
        start_time=start_time,
        duration=duration,
        frame_count=len(frames),
        used_facemesh_fallback=False,
    )
    return frames

def compute_av_mismatch(frames, window=5):
    return compute_av_sync_signals(frames, window=window)["mismatch"]


def compute_av_sync_signals(frames, window=5):
    """
    Per-frame AV sync diagnostics for stronger AV alignment modeling.

    Returns arrays for:
      - mismatch: 1 - local correlation
      - local_corr / local_corr_std
      - local_lag (frames), peak_corr, peak_prominence
      - onset_corr: correlation of |lip velocity| and |audio delta|
    """
    n = len(frames)
    zeros = np.zeros(n, dtype=np.float32)
    if n == 0:
        return {
            "mismatch": zeros,
            "local_corr": zeros,
            "local_corr_std": zeros,
            "local_lag": zeros,
            "peak_corr": zeros,
            "peak_prominence": zeros,
            "onset_corr": zeros,
        }

    lips = np.array([f.get("lip_aperture", 0.0) for f in frames], dtype=np.float32)
    audio = np.array([f.get("audio_rms", 0.0) for f in frames], dtype=np.float32)
    lip_vel = np.array([f.get("lip_velocity", 0.0) for f in frames], dtype=np.float32)
    aud_vel = np.array([f.get("audio_delta", 0.0) for f in frames], dtype=np.float32)

    mismatch = np.zeros(n, dtype=np.float32)
    local_corr = np.zeros(n, dtype=np.float32)
    local_corr_std = np.zeros(n, dtype=np.float32)
    local_lag = np.zeros(n, dtype=np.float32)
    peak_corr = np.zeros(n, dtype=np.float32)
    peak_prom = np.zeros(n, dtype=np.float32)
    onset_corr = np.zeros(n, dtype=np.float32)

    window = int(max(window, 1))
    for i in range(n):
        l = max(0, i - window)
        r = min(n, i + window + 1)
        ls = lips[l:r]
        au = audio[l:r]
        corr = _safe_corr(ls, au)
        local_corr[i] = corr
        mismatch[i] = float(1.0 - corr)
        local_corr_std[i] = float(np.std(_norm(ls) - _norm(au))) if len(ls) >= 2 else 0.0

        lv = np.abs(lip_vel[l:r])
        av = np.abs(aud_vel[l:r])
        onset_corr[i] = _safe_corr(lv, av)

        if len(ls) >= 3:
            ls0 = ls - np.mean(ls)
            au0 = au - np.mean(au)
            corr_full = np.correlate(ls0, au0, mode="full")
            denom = float(np.linalg.norm(ls0) * np.linalg.norm(au0) + 1e-6)
            corr_full = corr_full / denom
            pk_idx = int(np.argmax(corr_full))
            zero_idx = len(ls) - 1
            pk = float(corr_full[pk_idx])
            zc = float(corr_full[zero_idx])
            peak_corr[i] = pk
            peak_prom[i] = float(abs(pk - zc))
            local_lag[i] = float(pk_idx - zero_idx)

    return {
        "mismatch": mismatch,
        "local_corr": local_corr,
        "local_corr_std": local_corr_std,
        "local_lag": local_lag,
        "peak_corr": peak_corr,
        "peak_prominence": peak_prom,
        "onset_corr": onset_corr,
    }

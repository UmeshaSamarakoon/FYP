import cv2
import numpy as np
import librosa
import subprocess
import tempfile
import warnings
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

FACE_MESH = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

LIP_TOP, LIP_BOTTOM = 13, 14
LIP_IDX = list(range(0, 468))


def get_video_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0
    fps = fps if fps > 0 else 30.0  # fallback to a sane default
    return fps, duration


def extract_frame_level_features(
    video_path,
    start_time=0.0,
    duration=None,
    fps=None
):
    cap = cv2.VideoCapture(video_path)
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    # Limit audio read to the current chunk, with an ffmpeg fallback for mp4
    def _load_audio(path, offset, duration):
        try:
            return librosa.load(path, sr=None, offset=offset, duration=duration)
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
                    return librosa.load(tmp.name, sr=None, offset=offset, duration=duration)
                except Exception as e2:
                    raise RuntimeError(f"Audio extraction failed for {path}: {e2}") from e

    y, sr = _load_audio(
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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = FACE_MESH.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            pts = np.array([[p.x, p.y] for p in lm])

            lip_aperture = np.linalg.norm(
                pts[LIP_TOP] - pts[LIP_BOTTOM]
            )

            # audio at same timestamp
            audio_val = np.interp(t, audio_times, audio_rms)

            frames.append({
                "timestamp": t,
                "lip_aperture": lip_aperture,
                "audio_rms": audio_val,
                "landmarks": pts,
                "frame": frame
            })

        frame_idx += 1

    cap.release()
    return frames

def compute_av_mismatch(frames, window=5):
    lips = np.array([f["lip_aperture"] for f in frames])
    audio = np.array([f["audio_rms"] for f in frames])

    lips = (lips - lips.mean()) / (lips.std() + 1e-6)
    audio = (audio - audio.mean()) / (audio.std() + 1e-6)

    scores = []
    for i in range(len(frames)):
        l = max(0, i - window)
        r = min(len(frames), i + window)

        corr = np.corrcoef(lips[l:r], audio[l:r])[0, 1]
        mismatch = 1 - np.nan_to_num(corr)

        scores.append(mismatch)

    return scores

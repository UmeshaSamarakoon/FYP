import cv2
import numpy as np

# Import mediapipe solutions directly; fail fast with a clear error.
try:
    from mediapipe import solutions as mp_solutions
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "MediaPipe import failed. Ensure the official 'mediapipe' package is installed "
        "and no local file/folder named 'mediapipe' shadows it."
    ) from exc

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

# Initialize MediaPipe Face Detection ONCE at module load
mp_face_detection = mp_solutions.face_detection

_face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
    310, 311, 312, 13, 82, 81, 80, 191
]


def mouth_bbox_from_landmarks(landmarks, frame_shape, padding=0.05):
    """
    Returns a mouth-region bbox [x1, y1, x2, y2] using normalized landmarks.
    """
    if landmarks is None or frame_shape is None:
        return None

    h, w, _ = frame_shape
    if len(landmarks) == 0:
        return None

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

    return [x1, y1, x2, y2]


def detect_face_bbox(frame):
    """
    Returns a mouth-region bbox [x1, y1, x2, y2] or None.
    Uses face detection and derives a mouth ROI from the lower face region.
    """
    if frame is None:
        return None

    if not hasattr(frame, "shape"):
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = _face_detector.process(rgb)

    if not results.detections:
        return None

    h, w, _ = frame.shape
    box = results.detections[0].location_data.relative_bounding_box

    x1 = int(box.xmin * w)
    y1 = int(box.ymin * h)
    x2 = int((box.xmin + box.width) * w)
    y2 = int((box.ymin + box.height) * h)

    face_w = max(0, x2 - x1)
    face_h = max(0, y2 - y1)

    mouth_x1 = int(x1 + face_w * 0.2)
    mouth_x2 = int(x1 + face_w * 0.8)
    mouth_y1 = int(y1 + face_h * 0.6)
    mouth_y2 = int(y1 + face_h * 0.95)

    mouth_x1 = max(0, min(mouth_x1, w - 1))
    mouth_y1 = max(0, min(mouth_y1, h - 1))
    mouth_x2 = max(0, min(mouth_x2, w - 1))
    mouth_y2 = max(0, min(mouth_y2, h - 1))

    if mouth_x2 <= mouth_x1 or mouth_y2 <= mouth_y1:
        return None

    return [mouth_x1, mouth_y1, mouth_x2, mouth_y2]

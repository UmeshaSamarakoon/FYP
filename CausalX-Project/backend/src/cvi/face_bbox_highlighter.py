import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection ONCE at module load
mp_face_detection = mp.solutions.face_detection

_face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


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

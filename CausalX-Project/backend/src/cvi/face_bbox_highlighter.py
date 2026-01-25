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
    Returns [x1, y1, x2, y2] or None
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

    return [x1, y1, x2, y2]

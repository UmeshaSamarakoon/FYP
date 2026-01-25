import sys, os
import cv2

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.cvi.face_bbox_highlighter import detect_face_bbox

VIDEO = "data/raw/fakeavceleb/FakeVideo-FakeAudio/African/men/id00076/00109_2_id00166_wavtolip.mp4"

cap = cv2.VideoCapture(VIDEO)
ret, frame = cap.read()
cap.release()

bbox = detect_face_bbox(frame)

print("Bounding box:", bbox)

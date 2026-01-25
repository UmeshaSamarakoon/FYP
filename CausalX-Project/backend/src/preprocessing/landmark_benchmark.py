import cv2
import mediapipe as mp
import dlib
import time
import pandas as pd

class LandmarkBenchmarker:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        
        # Initialize Dlib (Ensure you have shape_predictor_68_face_landmarks.dat in /models)
        self.detector = dlib.get_frontal_face_detector()
        # self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    def benchmark_mediapipe(self, frame):
        start = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_frame)
        latency = (time.time() - start) * 1000 # ms
        return latency

    def benchmark_opencv_haar(self, frame):
        # OpenCV Haar is usually the "baseline" but less accurate
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        start = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        latency = (time.time() - start) * 1000
        return latency

# --- Logic to run the test ---
def run_comparison(video_path):
    bench = LandmarkBenchmarker()
    cap = cv2.VideoCapture(video_path)
    data = []

    for _ in range(100): # Test over 100 frames
        ret, frame = cap.read()
        if not ret: break
        
        mp_time = bench.benchmark_mediapipe(frame)
        cv_time = bench.benchmark_opencv_haar(frame)
        
        data.append({"Method": "MediaPipe", "Latency (ms)": mp_time})
        data.append({"Method": "OpenCV Haar", "Latency (ms)": cv_time})

    cap.release()
    df = pd.DataFrame(data)
    print(df.groupby("Method").mean())

# run_comparison("data/raw/sample_video.mp4")
import sys
import os
import cv2
import numpy as np
import mediapipe as mp

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

DATA_RAW_DIR = os.path.join(project_root, "data", "raw", "train_sample_videos")

from src.utils.metadata_parser import load_metadata

# RIGID INDICES: Forehead and Nose Bridge (Parts that don't move during speech)
RIGID_ZONE = [1, 2, 4, 5, 6, 8, 9, 10, 151, 9, 67, 103, 109, 332, 338, 297]

def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def align_landmarks_advanced(landmarks):
    """
    FULL PROCRUSTES ALIGNMENT:
    Removes Translation (Slide), Rotation (Tilt), and Scale (Leaning).
    """
    NOSE = 1
    L_EYE = 33
    R_EYE = 263

    # 1. Translate (Center on Nose)
    nose_anchor = landmarks[NOSE]
    centered = landmarks - nose_anchor

    # 2. Rotate (Level the eyes)
    left_eye, right_eye = centered[L_EYE], centered[R_EYE]
    angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = np.dot(centered, rotation_matrix.T)

    # 3. Scale (Normalize based on eye distance)
    eye_dist = np.linalg.norm(rotated[R_EYE] - rotated[L_EYE])
    return rotated / eye_dist if eye_dist > 0 else rotated

def extract_jitter_logic(video_path, label="Video"):
    print(f"🎬 Analyzing {label}: {os.path.basename(video_path)}...")
    mp_face_mesh = mp.solutions.face_mesh
    
    # --- CRITICAL FIX: Add confidence thresholds back ---
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        refine_landmarks=False, # Set to False for better stability on M2
        min_detection_confidence=0.3, # This was missing!
        min_tracking_confidence=0.3   # This was missing!
    )
    
    cap = cv2.VideoCapture(video_path)
    prev_rigid_points = None
    frame_jitters = []
    detection_count = 0
    total_frames = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        total_frames += 1
        
        h, w = frame.shape[:2]
        new_h = 480
        new_w = int(w * (new_h / h))
        frame = cv2.resize(frame, (new_w, new_h))
        
        # 1. Boost contrast
        enhanced = apply_clahe(frame)
        
        # 2. Prepare frame for MediaPipe (Optimization for Mac)
        rgb_frame = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if results.multi_face_landmarks:
            detection_count += 1
            raw_pts = np.array([[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark])
            
            # 3. Apply the Advanced Alignment
            aligned_pts = align_landmarks_advanced(raw_pts)
            rigid_pts = aligned_pts[RIGID_ZONE]
            
            if prev_rigid_points is not None:
                dist = np.linalg.norm(rigid_pts - prev_rigid_points, axis=1)
                frame_jitters.append(np.mean(dist))
            
            prev_rigid_points = rigid_pts

            # Visual Debug: Draw the Rigid Zone points
            for idx in RIGID_ZONE:
                pt = raw_pts[idx]
                cv2.circle(frame, (int(pt[0]*new_w), int(pt[1]*new_h)), 2, (0, 255, 0), -1)

        # Show live processing window
        cv2.imshow(f'CausalX Debug: {label}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    
    avg_jitter = np.mean(frame_jitters) if frame_jitters else 0.0
    return avg_jitter, detection_count, total_frames

def run_benchmark():
    metadata = load_metadata(DATA_RAW_DIR)
    pair = next(((f, info['original']) for f, info in metadata.items() 
                 if info['label'] == 'FAKE' and 'original' in info and 
                 os.path.exists(os.path.join(DATA_RAW_DIR, f)) and 
                 os.path.exists(os.path.join(DATA_RAW_DIR, info['original']))), None)

    if not pair: return

    real_j, r_c, r_t = extract_jitter_logic(os.path.join(DATA_RAW_DIR, pair[1]), "REAL")
    fake_j, f_c, f_t = extract_jitter_logic(os.path.join(DATA_RAW_DIR, pair[0]), "FAKE")

    print(f"\n" + "="*45)
    print(f"📊 STABILIZED RIGID-ZONE RESULTS")
    print(f"="*45)
    print(f"REAL Jitter: {real_j:.8f} ({r_c}/{r_t} frames)")
    print(f"FAKE Jitter: {fake_j:.8f} ({f_c}/{f_t} frames)")
    print(f"Causal Gap:  {fake_j - real_j:.8f}")
    print("="*45)

if __name__ == "__main__":
    run_benchmark()
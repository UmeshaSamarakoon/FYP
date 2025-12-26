import sys
import os
import cv2
import numpy as np

# --- 1. ROBUST MEDIAPIPE IMPORT ---
try:
    import mediapipe as mp
    # Test if solutions is actually accessible
    _test = mp.solutions.face_mesh
    print("✅ MediaPipe solutions found!")
except (AttributeError, ImportError):
    print("⚠️ Standard import failed, trying internal fallback...")
    try:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        import mediapipe.python.solutions.drawing_utils as mp_drawing
        print("✅ Fallback internal import successful!")
    except ImportError:
        print("❌ CRITICAL: MediaPipe is still broken. Check Python version (Must be < 3.12)")
        sys.exit(1)

# --- 2. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from src.utils.metadata_parser import load_metadata
DATA_RAW_DIR = os.path.join(project_root, "data", "raw", "train_sample_videos")

def extract_jitter_logic(video_path):
    """Processes a video and returns frames with landmarks and a jitter score."""
    # Initialize inside the function for stability
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    
    cap = cv2.VideoCapture(video_path)
    prev_landmarks = None
    frame_jitters = []
    processed_frames = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.resize(frame, (480, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = np.array([[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark])
            if prev_landmarks is not None:
                movement = np.linalg.norm(landmarks - prev_landmarks, axis=1)
                frame_jitters.append(np.mean(movement))
            prev_landmarks = landmarks

            for lm in landmarks:
                x, y = int(lm[0] * 480), int(lm[1] * 480)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        processed_frames.append(frame)
    
    cap.release()
    face_mesh.close() # Clean up memory
    return processed_frames, frame_jitters

def run_benchmark():
    print(f"📂 Searching data in: {DATA_RAW_DIR}")
    try:
        metadata = load_metadata(DATA_RAW_DIR)
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return

    # Find a Pair
   # Find a Pair that actually exists on disk
    real_video_name, fake_video_name = None, None
    
    print("🔍 Searching metadata for a pair where BOTH files exist...")
    for filename, info in metadata.items():
        # Check if this entry is a FAKE video
        if info['label'] == 'FAKE' and 'original' in info:
            temp_fake = filename
            temp_real = info['original']
            
            # Construct the full paths
            path_fake = os.path.join(DATA_RAW_DIR, temp_fake)
            path_real = os.path.join(DATA_RAW_DIR, temp_real)
            
            # DEBUG: See what is happening
            # print(f"Checking: {temp_fake} (exists: {os.path.exists(path_fake)}) | {temp_real} (exists: {os.path.exists(path_real)})")

            # ONLY pick this pair if both files are physically there
            if os.path.exists(path_fake) and os.path.exists(path_real):
                fake_video_name = temp_fake
                real_video_name = temp_real
                print(f"✅ Found valid pair on disk: REAL({real_video_name}) vs FAKE({fake_video_name})")
                break # We found one! Stop looking.

    if not real_video_name:
        print("❌ No valid causal pair found in metadata.json")
        return

    # Process both videos
    print("⌛ Processing videos (Side-by-side benchmark)...")
    real_path = os.path.join(DATA_RAW_DIR, real_video_name)
    fake_path = os.path.join(DATA_RAW_DIR, fake_video_name)
    
    real_frames, real_jitters = extract_jitter_logic(real_path)
    fake_frames, fake_jitters = extract_jitter_logic(fake_path)

    print(f"\n--- CAUSAL JITTER RESULTS ---")
    print(f"REAL Avg Jitter: {np.mean(real_jitters):.6f}")
    print(f"FAKE Avg Jitter: {np.mean(fake_jitters):.6f}")
    print(f"Difference: {abs(np.mean(fake_jitters) - np.mean(real_jitters)):.6f}")

    # Show Side-by-Side
    print("Press 'q' to close the video window.")
    for r, f in zip(real_frames, fake_frames):
        combined = np.hstack((r, f))
        cv2.imshow('Side-by-Side: REAL (Left) vs FAKE (Right)', combined)
        if cv2.waitKey(30) & 0xFF == ord('q'): break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_benchmark()
import mediapipe as mp
try:
    print(f"MediaPipe Version: {mp.__version__}")
    mesh = mp.solutions.face_mesh
    print("✅ Success! 'solutions' attribute found.")
except AttributeError:
    print("❌ Failure: Still cannot find 'solutions'.")
except Exception as e:
    print(f"❌ An error occurred: {e}")
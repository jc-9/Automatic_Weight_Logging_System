import os
import cv2
import time
from face_engine import FaceEngine


def run_vision_test():
    print("==================================================")
    print("      FACIAL RECOGNITION ENGINE TEST SCRIPT       ")
    print("==================================================")

    # 1. Initialize the Face Engine
    # This will crawl the 'known_users' directory and generate facial encodings
    try:
        engine = FaceEngine(known_users_dir = "known_users")
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize FaceEngine: {e}")
        print("Make sure 'face_recognition', 'dlib', and 'numpy' are installed.")
        return

    if not engine.known_names:
        print("\n[WARNING] No profiles found in the 'known_users' directory!")
        print("Please place a clear '.jpg' or '.png' image of a face in the folder")
        print("and name it 'yourname.jpg' before running this test.")
        return

    print("\n[INFO] Starting camera tracking loop...")
    print("[INFO] Looking for a match... (Timeout set to 10 seconds)")
    print("Please look directly into your camera now.")
    print("--------------------------------------------------")

    # 2. Run the recognition sequence
    # We increase the timeout slightly for testing purposes
    start_time = time.time()
    matched_identity = engine.recognize_user(timeout = 10)
    elapsed_time = time.time() - start_time

    # 3. Output Results
    print("--------------------------------------------------")
    print(f"Test Finished in: {elapsed_time:.2f} seconds")

    if matched_identity != "Unknown":
        print(f"🎯 MATCH FOUND! Identity resolved as: ** {matched_identity} **")
    else:
        print("❌ NO MATCH FOUND. Engine returned: 'Unknown'")
        print("\nTroubleshooting tips:")
        print("1. Ensure your camera lens is clean and has adequate lighting.")
        print("2. Ensure the registration photo in 'known_users' is front-facing and clear.")
        print("3. Check if your system's default camera index is '0'.")
    print("==================================================")


if __name__ == "__main__":
    run_vision_test()
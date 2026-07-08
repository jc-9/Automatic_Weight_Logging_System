import os
import cv2
import face_recognition
import numpy as np
import time


class FaceEngine:
    def __init__(self, known_users_dir = "known_users"):
        self.known_encodings = []
        self.known_names = []
        self.known_users_dir = known_users_dir
        self.load_profiles()

    def load_profiles(self):
        print("[VISION] Initializing facial library matrix...")
        if not os.path.exists(self.known_users_dir):
            os.makedirs(self.known_users_dir)
            return

        for file in os.listdir(self.known_users_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(self.known_users_dir, file)
                try:
                    img = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(os.path.splitext(file)[0])
                        print(f"[VISION] Loaded profile: {os.path.splitext(file)[0]}")
                except Exception as e:
                    print(f"[VISION] Error loading {file}: {e}")

    def recognize_user(self, timeout = 5):
        # Open connection to the Raspberry Pi Camera / USB Web Cam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[VISION] Error: Camera could not be initialized.")
            return "Unknown"

        start_time = time.time()
        matched_name = "Unknown"

        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret:
                continue

            # Downscale frame to 1/4 size for fast processing on Raspberry Pi CPU
            small_frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect and encode faces found in current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance = 0.5)
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)

                if face_distances.size > 0:
                    best_match_idx = np.argmin(face_distances)
                    if matches[best_match_idx]:
                        matched_name = self.known_names[best_match_idx]
                        cap.release()
                        return matched_name

        cap.release()
        return matched_name
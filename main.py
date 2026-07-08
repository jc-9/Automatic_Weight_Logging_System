import os
import cv2
import face_recognition
import numpy as np
import time
from datetime import datetime
import docker
import re
import threading
from influxdb_client import InfluxDBClient, Point, WriteOptions

# =====================================================================
# SYSTEM LOGGER HELPER
# =====================================================================
def log(message: str):
    """Prints a message prefixed with a high-precision local timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

# =====================================================================
# FACIAL RECOGNITION ENGINE (EVENT-DRIVEN CAMERA)
# =====================================================================
class FaceEngine:
    def __init__(self, known_users_dir="known_users"):
        self.known_encodings = []
        self.known_names = []
        self.known_users_dir = known_users_dir
        self.load_profiles()
        
        # Identity states
        self.active_user = "Unknown"
        self.user_lock_time = 0
        self.is_scanning = False

    def load_profiles(self):
        log("[VISION] Initializing facial library matrix...")
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
                        log(f"[VISION] Loaded profile: {os.path.splitext(file)[0]}")
                except Exception as e:
                    log(f"[VISION] Error loading {file}: {e}")

    def _threaded_scan(self):
        """Streams video dynamically for up to 10 seconds, forcing valid frame capture blocks."""
        log("📷 [CAMERA TRIGGER] Scale detected step-on event. Initializing high-intensity tracking loop...")
        cap = None
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not cap.isOpened():
                log("⚠️ [CAMERA ERROR] Hardware bus failed to open.")
                self.is_scanning = False
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # V4L2 BUFFER FLUSH
            time.sleep(0.4)
            for _ in range(6):
                cap.read()

            start_time = time.time()
            scan_duration = 10.0
            frames_processed = 0

            log("🧬 [VISION] Analyzing live video pipeline streams...")
            while time.time() - start_time < scan_duration:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                frames_processed += 1
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                if not face_locations:
                    continue

                log(f"👤 [VISION] Face detected in frame {frames_processed}! Comparing features...")
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.58)
                    face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)

                    if face_distances.size > 0:
                        best_match_idx = np.argmin(face_distances)
                        if matches[best_match_idx]:
                            self.active_user = self.known_names[best_match_idx]
                            self.user_lock_time = time.time()
                            log(f"🎯 [IDENTITY MATCH] Confirmed user '{self.active_user}' after evaluating {frames_processed} frames!")
                            return

                time.sleep(0.02)
            
            log(f"👤 [VISION] Scan window expired. Evaluated {frames_processed} total frames. Identity: Unknown.")
        except Exception as e:
            log(f"⚠️ [CAMERA THREAD CRASH] Error: {e}")
        finally:
            if cap is not None:
                cap.release()
            self.is_scanning = False

    def trigger_scan(self):
        if self.is_scanning:
            return
        self.is_scanning = True
        self.active_user = "Unknown"
        threading.Thread(target=self._threaded_scan, daemon=True).start()

# =====================================================================
# SYSTEM STREAM PARSER (SINGLE BACKBONE THREAD WORKER)
# =====================================================================
def log_stream_worker(engine, current_metrics, write_api, ORG, BUCKET):
    docker_client = docker.from_env()
    log_stream = None

    while True:
        if log_stream is None:
            try:
                container = docker_client.containers.get("bles")
                current_epoch = int(time.time())
                log_stream = container.logs(stream=True, follow=True, since=current_epoch)
                log("🔄 [DOCKER LINK] Background thread successfully locked onto 'bles' stream.")
            except Exception as e:
                log(f"⏳ [DOCKER LINK] Worker waiting for container 'bles' to recover... ({e})")
                time.sleep(2.0)
                continue

        try:
            for chunk in log_stream:
                line = chunk.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                # ⚡ THE REACTION TRIGGER
                if "Found device:" in line or "Renpho-Scale" in line:
                    log(f"📶 [SCALE EVENT] Step-on detected! Captured line -> {line}")
                    engine.trigger_scan()

                # ⚖️ PARSE WEIGHTS & IMPEDANCE
                elif "Measurement received:" in line:
                    match = re.search(r"Measurement received:\s+([0-9.]+)\s+lbs\s+/\s+([0-9.]+)\s+Ohm", line)
                    if match:
                        current_metrics["weight"] = float(match.group(1))
                        current_metrics["impedance"] = float(match.group(2))
                        log(f"⚖️ Intercepted Payload: {current_metrics['weight']} lbs.")
                        
                        # Trigger an instant conditional write attempt if user identity is already resolved
                        if engine.active_user != "Unknown":
                            commit_metrics_to_db(engine, current_metrics, write_api, ORG, BUCKET)

                # 🩸 PARSE BODY FAT
                elif "bodyFatPercent:" in line:
                    match = re.search(r"bodyFatPercent:\s+([0-9.]+)", line)
                    if match:
                        current_metrics["body_fat"] = float(match.group(1))
                        log(f"🩸 Intercepted Body Fat: {current_metrics['body_fat']}%")
                        
                        # Trigger an instant conditional write attempt if user identity is already resolved
                        if engine.active_user != "Unknown":
                            commit_metrics_to_db(engine, current_metrics, write_api, ORG, BUCKET)

                # 🟢 TRANSACTION END SIGNALS
                elif "No exporters configured" in line or "measurement processed" in line or "Session boundary" in line:
                    if "weight" in current_metrics and engine.active_user != "Unknown":
                        # Catch out-of-order updates that missed immediate evaluation hooks
                        commit_metrics_to_db(engine, current_metrics, write_api, ORG, BUCKET)
                    else:
                        # Safe fallback window: give parsing thread a minor delay block before dumping cache
                        time.sleep(0.5)
                        if "weight" in current_metrics and engine.active_user != "Unknown":
                            commit_metrics_to_db(engine, current_metrics, write_api, ORG, BUCKET)
                        else:
                            current_metrics.clear()
                            engine.active_user = "Unknown"
                            log("🟢 Session boundary clean up. Returning worker to default idle state.")

            log("⚠️ [DOCKER ALERT] Log stream ended cleanly. Resetting handle...")
            log_stream = None

        except Exception as e:
            log(f"⚠️ [STREAM WORKER ERROR] Connection issue: {e}")
            log_stream = None
            time.sleep(1.0)

# =====================================================================
# MODULAR DATABASE COMMIT DRIVER
# =====================================================================
def commit_metrics_to_db(engine, current_metrics, write_api, ORG, BUCKET):
    """Safely constructs data points and flushes metrics directly to InfluxDB."""
    if "weight" not in current_metrics or engine.active_user == "Unknown":
        return

    try:
        point = Point("weight_metrics") \
            .tag("user", engine.active_user) \
            .field("weight", current_metrics["weight"]) \
            .field("impedance", current_metrics["impedance"])
        
        if "body_fat" in current_metrics:
            point.field("body_fat", current_metrics["body_fat"])

        write_api.write(bucket=BUCKET, org=ORG, record=point)
        log(f"🚀 Success: Instantly synced metrics for '{engine.active_user}' to InfluxDB Dashboard.")
        
        # Clear specific variables following database flush
        current_metrics.clear()
        engine.active_user = "Unknown"
    except Exception as e:
        log(f"❌ InfluxDB Synchronous Write Error: {e}")

# =====================================================================
# SYSTEM MAIN ENGINE START
# =====================================================================
def start_reactive_system():
    NAS_IP = os.environ.get("NAS_IP")
    INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN")
    ORG = os.environ.get("ORG", "Home")
    BUCKET = os.environ.get("BUCKET", "scale")
    
    log(f"🔍 Debug: Loaded NAS_IP={NAS_IP} from environment.")
    engine = FaceEngine()
    
    try:
        influx_client = InfluxDBClient(url=f"http://{NAS_IP}:8086", token=INFLUX_TOKEN, org=ORG)
        write_api = influx_client.write_api(write_options=WriteOptions(batch_size=1))
        log("🔌 Connected to InfluxDB on NAS.")
    except Exception as e:
        log(f"❌ InfluxDB connection error: {e}")
        return

    current_metrics = {}

    stream_thread = threading.Thread(
        target=log_stream_worker, 
        args=(engine, current_metrics, write_api, ORG, BUCKET), 
        daemon=True
    )
    stream_thread.start()
    log("🟢 System armed. Dedicated background parser thread active.")

    last_heartbeat_time = 0
    HEARTBEAT_INTERVAL = 10.0

    while True:
        current_time = time.time()
        
        if current_time - last_heartbeat_time >= HEARTBEAT_INTERVAL:
            log("💓 [HEARTBEAT] Monitoring loop idling at minimal CPU utilization.")
            last_heartbeat_time = current_time

        if engine.active_user != "Unknown" and (current_time - engine.user_lock_time > 60):
            log("⏱️ [TIMEOUT] Identity cache expired without weights. Resetting.")
            engine.active_user = "Unknown"
            current_metrics.clear()

        time.sleep(1.0)

if __name__ == '__main__':
    start_reactive_system()

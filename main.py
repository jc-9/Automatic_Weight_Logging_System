import os
import cv2
import face_recognition
import numpy as np
import time
from datetime import datetime
import docker
import re
import threading
import socket
from influxdb_client import InfluxDBClient, Point, WriteOptions

# =====================================================================
# OLED DISPLAY ENGINE INITIALIZATION
# =====================================================================
try:
    from luma.core.interface.serial import i2c
    from luma.oled.device import sh1106
    from luma.core.render import canvas
    from PIL import ImageFont
    
    serial = i2c(port=1, address=0x3C)
    OLED_DEVICE = sh1106(serial)
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()
    print("📺 [OLED] Display successfully initialized on I2C bus.")
except Exception as e:
    print(f"⚠️ [OLED] Initialization Failed: {e}")
    OLED_DEVICE = None
    font_large = None
    font_small = None

# Global flag to track scale health for the screen
SCALE_READY = False

def check_wifi():
    """Checks if the local network is reachable."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        NAS_IP = os.environ.get("NAS_IP", "192.168.1.1")
        s.connect((NAS_IP, 80)) 
        s.close()
        return True
    except Exception:
        return False

def update_hmi(state, username="", weight=""):
    """Renders the screen state instantly."""
    if not OLED_DEVICE:
        return

    wifi_alive = check_wifi()

    with canvas(OLED_DEVICE) as draw:
        # Top Status Bar
        wifi_txt = "WiFi: OK" if wifi_alive else "WiFi: OFF"
        scale_txt = "SCALE: READY" if SCALE_READY else "SCALE: ERR"
        
        draw.text((0, 0), wifi_txt, fill="white", font=font_small)
        draw.text((70, 0), scale_txt, fill="white", font=font_small)
        draw.line((0, 14, 128, 14), fill="white")
        
        # Main HMI Logic
        if not wifi_alive:
            draw.text((10, 25), "SYSTEM OFFLINE", fill="white", font=font_large)
            draw.text((15, 42), "Check Router/WiFi", fill="white", font=font_small)
            
        elif not SCALE_READY:
            draw.text((15, 25), "SCALE OFFLINE", fill="white", font=font_large)
            draw.text((10, 42), "Check Docker Logs", fill="white", font=font_small)

        elif state == 'IDLE':
            draw.text((25, 25), "SCALE READY", fill="white", font=font_large)
            draw.text((12, 42), "Step on to begin...", fill="white", font=font_small)
            
        elif state == 'SCANNING_FACE':
            draw.text((15, 25), "LOOK AT CAMERA", fill="white", font=font_large)
            draw.text((20, 42), "Identifying user...", fill="white", font=font_small)
            
        elif state == 'FACE_MATCHED':
            draw.text((5, 25), f"HELLO {username.upper()}!", fill="white", font=font_large)
            draw.text((10, 42), "Hold still on scale", fill="white", font=font_small)
            
        elif state == 'LOGGING_DATA':
            draw.text((20, 25), "SAVING METRICS", fill="white", font=font_large)
            draw.text((15, 42), "Syncing to NAS...", fill="white", font=font_small)
            
        elif state == 'SUCCESS':
            draw.text((25, 20), "WEIGHT LOGGED!", fill="white", font=font_large)
            if weight:
                draw.text((35, 38), f"{weight} lbs", fill="white", font=font_large)
            draw.text((30, 52), "Done. Step off.", fill="white", font=font_small)

def show_success_screen(weight):
    """Flashes success for 5 seconds without blocking background parsing."""
    update_hmi('SUCCESS', weight=str(weight))
    time.sleep(5)
    update_hmi('IDLE')

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
                            update_hmi('FACE_MATCHED', username=self.active_user)
                            return

                time.sleep(0.02)
            
            log(f"👤 [VISION] Scan window expired. Evaluated {frames_processed} total frames. Identity: Unknown.")
            update_hmi('IDLE')
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
        update_hmi('SCANNING_FACE')
        threading.Thread(target=self._threaded_scan, daemon=True).start()

# =====================================================================
# SYSTEM STREAM PARSER (SINGLE BACKBONE THREAD WORKER)
# =====================================================================
def log_stream_worker(engine, current_metrics, write_api, ORG, BUCKET):
    global SCALE_READY
    docker_client = docker.from_env()
    log_stream = None

    while True:
        if log_stream is None:
            try:
                container = docker_client.containers.get("bles")
                current_epoch = int(time.time())
                log_stream = container.logs(stream=True, follow=True, since=current_epoch)
                SCALE_READY = True
                update_hmi('IDLE')
                log("🔄 [DOCKER LINK] Background thread successfully locked onto 'bles' stream.")
            except Exception as e:
                SCALE_READY = False
                update_hmi('IDLE')
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
            SCALE_READY = False

        except Exception as e:
            log(f"⚠️ [STREAM WORKER ERROR] Connection issue: {e}")
            log_stream = None
            SCALE_READY = False
            time.sleep(1.0)

# =====================================================================
# MODULAR DATABASE COMMIT DRIVER
# =====================================================================
def commit_metrics_to_db(engine, current_metrics, write_api, ORG, BUCKET):
    """Safely constructs data points and flushes metrics directly to InfluxDB."""
    if "weight" not in current_metrics or engine.active_user == "Unknown":
        return

    try:
        update_hmi('LOGGING_DATA')
        
        point = Point("weight_metrics") \
            .tag("user", engine.active_user) \
            .field("weight", current_metrics["weight"]) \
            .field("impedance", current_metrics["impedance"])
        
        if "body_fat" in current_metrics:
            point.field("body_fat", current_metrics["body_fat"])

        write_api.write(bucket=BUCKET, org=ORG, record=point)
        log(f"🚀 Success: Instantly synced metrics for '{engine.active_user}' to InfluxDB Dashboard.")
        
        # Flash the success screen in a background thread so we don't block
        threading.Thread(target=show_success_screen, args=(current_metrics["weight"],), daemon=True).start()
        
        # Clear specific variables following database flush
        current_metrics.clear()
        engine.active_user = "Unknown"
    except Exception as e:
        log(f"❌ InfluxDB Synchronous Write Error: {e}")
        update_hmi('IDLE')

# =====================================================================
# SYSTEM MAIN ENGINE START
# =====================================================================
def start_reactive_system():
    NAS_IP = os.environ.get("NAS_IP")
    INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN")
    ORG = os.environ.get("ORG", "Home")
    BUCKET = os.environ.get("BUCKET", "scale")
    
    log(f"🔍 Debug: Loaded NAS_IP={NAS_IP} from environment.")
    
    # Init Display
    update_hmi('IDLE')
    
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
            update_hmi('IDLE')

        time.sleep(1.0)

if __name__ == '__main__':
    start_reactive_system()

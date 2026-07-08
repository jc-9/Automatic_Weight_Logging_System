import asyncio
import threading
import time
from ble_sniffer import ScaleSniffer
from face_engine import FaceEngine
from hmi_display import HmiDisplay
from db_handler import DbHandler

# Instantiating edge systems
display = HmiDisplay()
db = DbHandler()
vision = FaceEngine()

last_process_time = 0
COOLDOWN_PERIOD = 7  # Prevents duplicate entries within 7 seconds of weighing


def handle_scale_trigger(weight):
    global last_process_time
    current_time = time.time()

    # Simple gate to ignore rolling scale data updates while user is standing still
    if current_time - last_process_time < COOLDOWN_PERIOD:
        return

    last_process_time = current_time
    print(f"\n[SYSTEM EVENT] Scale stable: {weight} kg detected.")

    # 1. Update HMI screen
    display.show_processing(weight)

    # 2. Fire biometric facial vision scan
    user_identity = vision.recognize_user(timeout = 5)
    print(f"[SYSTEM EVENT] Identity resolved: {user_identity}")

    # 3. Write payload down to internal database engine
    success = db.log_event(user_identity, weight)

    # 4. Route feedback loop back out to display matrix
    if success:
        display.show_success(user_identity, weight)
    else:
        display.show_error("DB Write Error")


async def main():
    # Pass our handler function down to the BLE thread hook
    sniffer = ScaleSniffer(callback = handle_scale_trigger)
    await sniffer.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSystem shutting down cleanly.")
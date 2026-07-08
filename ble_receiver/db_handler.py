import requests
from datetime import datetime


class DbHandler:
    def __init__(self):
        # REPLACE THIS with your server's internal IP address and target port
        self.server_url = "http://192.168.1.100:5000/api/weight"

    def log_event(self, user_id, weight_kg):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Format the data into a standardized JSON payload
            payload = {
                "user_id": user_id,
                "weight_kg": weight_kg,
                "timestamp": timestamp
            }

            # Ship the data across your home network to the server
            response = requests.post(self.server_url, json = payload, timeout = 3)

            if response.status_code == 200:
                print(f"[NETWORK] Successfully sent {weight_kg}kg for {user_id} to server.")
                return True
            else:
                print(f"[NETWORK] Server rejected data with status: {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"[NETWORK] Failed to connect to server: {e}")
            return False
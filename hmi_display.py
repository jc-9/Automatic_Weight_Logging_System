from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306
import time

class HmiDisplay:
    def __init__(self):
        try:
            # Connect using the Pi's hardware I2C port 1 (Pins 3 and 5)
            serial = i2c(port=1, address=0x3C)
            self.device = ssd1306(serial)
            self.show_idle()
        except Exception as e:
            print(f"[HMI] Failed to clear screen: {e}")
            self.device = None

    def show_idle(self):
        if not self.device: return
        with canvas(self.device) as draw:
            draw.text((10, 20), "SYSTEM READY", fill="white")
            draw.text((10, 35), "Step on scale...", fill="white")

    def show_processing(self, weight):
        if not self.device: return
        with canvas(self.device) as draw:
            draw.text((10, 10), f"Weight: {weight} kg", fill="white")
            draw.text((10, 30), "Scanning face...", fill="white")

    def show_success(self, name, weight):
        if not self.device: return
        with canvas(self.device) as draw:
            draw.text((10, 5), "LOGGED SUCCESS!", fill="white")
            draw.text((10, 25), f"User: {name}", fill="white")
            draw.text((10, 45), f"Weight: {weight} kg", fill="white")
        time.sleep(4)
        self.show_idle()

    def show_error(self, msg):
        if not self.device: return
        with canvas(self.device) as draw:
            draw.text((10, 20), "ERROR LOGGING", fill="white")
            draw.text((10, 35), msg, fill="white")
        time.sleep(3)
        self.show_idle()
import time
from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
from luma.core.render import canvas

try:
    # Forces i2c port 1 and hardware address 0x3c
    serial = i2c(port=1, address=0x3c)
    # Explicitly defines the SH1106 driver
    device = sh1106(serial)
    
    with canvas(device) as draw:
        draw.rectangle(device.bounding_box, outline="white", fill="black")
        draw.text((20, 25), "HARDWARE ALIVE!", fill="white")
    
    print("Test command sent! Check the screen.")
    time.sleep(5)
except Exception as e:
    print(f"Error: {e}")

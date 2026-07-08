# Automated Biometric Weight Tracking System (IoT)

An automated, hands-free weight logging system built using a Raspberry Pi 4, a low-cost Bluetooth Low Energy (BLE) smart scale, a camera module, and an I2C OLED display. The system seamlessly identifies users via facial recognition the moment they step on the scale and automatically commits their weight events directly to a private timeseries database.

## 🚀 Key Features

* **Zero-Effort UX:** No apps to open, no buttons to press. Just step on the scale, look at the camera, and step off.
* **Local Biometric Auth:** 99.38% accurate facial recognition running completely locally and offline on a Raspberry Pi 4 using the `face_recognition` deep learning library.
* **Passive BLE Sniffing:** Intercepts unencrypted Bluetooth advertisement packets broadcasted by budget smart scales (e.g., Renpho, Xiaomi, Wyze).
* **Real-Time HMI Feedback:** Clear user verification, weight readout, and database confirmation status using a cheap 0.96" SSD1306 OLED display.
* **Timeseries Architecture:** Stores data cleanly inside a chronological structured database schema for streamlined telemetry tracking.

## 📐 System Architecture

[ User Steps on Scale ]
          │
          ├──(Bluetooth BLE Broadcast)──> [ Bluetooth Packet Sniffer (Bleak) ]
          │                                           │ (Raw Weight Data)
          └──(Camera Sight Line)        ──> [ Facial Recognition Node (dlib) ]
                                                      │ (Matched User ID)
                                                      ▼
                                           [ Central Logic Broker (Python) ]
                                                      │
                                                      ▼
                                           [ Database Connection Layer ]

## 🛠️ Hardware Requirements

* **Raspberry Pi 4** (or any modern Linux platform with BLE capabilities and >=2GB RAM)
* **Raspberry Pi Camera Module** (or standard USB Webcam mounted at eye level)
* **0.96" SSD1306 I2C OLED Display** (128x64 resolution, monochrome)
* **Cheap Bluetooth Smart Scale** (e.g., Renpho, Wyze Scale X, Xiaomi Mi Body Composition 2)
* 4x Female-to-Female Jumper Wires

## 🔌 Hardware Wiring (OLED to Raspberry Pi 4)

Connect the I2C OLED display directly to the Pi's dedicated hardware I2C pins:

| OLED Pin | RPi4 GPIO Pin | Physical Pin Number |
| :--- | :--- | :--- |
| **VCC** | 3.3V Power | Pin 1 |
| **GND** | Ground | Pin 6 |
| **SDA** | GPIO 2 (SDA) | Pin 3 |
| **SCL** | GPIO 3 (SCL) | Pin 5 |

## 💻 Software Prerequisites

```bash
# Update package lists
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies for facial recognition (dlib compilation)
sudo apt-get install -y cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev

# Enable I2C interface via config tool
sudo raspi-config # Navigate to Interface Options -> I2C -> Enable


pip install opencv-python face_recognition bleak luma.oled numpy

scale-biometric-logger/
├── README.md               # Project documentation
├── main.py                 # Master execution thread orchestration
├── face_engine.py          # Facial detection & encoding logic
├── ble_sniffer.py          # BLE packet capture and parsing loop
├── hmi_display.py          # OLED drawing and UI state machine
├── db_handler.py           # Database ingestion connection scripts
└── known_users/            # Directory containing registration photos
    ├── alex.jpg            # Dead-center neutral lighting registration photo
    └── sarah.jpg
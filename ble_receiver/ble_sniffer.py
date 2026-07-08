import asyncio
from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData


class ScaleSniffer:
    def __init__(self, callback):
        self.callback = callback
        self.scanning = True

    def _detection_callback(self, device: BLEDevice, advertisement_data: AdvertisementData):
        # Look for manufacturing data or specific scale names (e.g., "KiwiScale", "iScale", "Taic")
        if advertisement_data.manufacturer_data:
            for manuf_id, data in advertisement_data.manufacturer_data.items():
                # Passive check: Weight packets are usually 10-20 bytes long
                if len(data) >= 6:
                    try:
                        # Common parsing structure: weight is often stored in bytes 4 and 5
                        # multiplied by a scale factor (e.g., 0.1 or 0.01)
                        raw_weight = (data[4] << 8) + data[5]
                        weight_kg = raw_weight / 10.0

                        # Validate that it's a realistic human weight (e.g., 30kg to 200kg)
                        if 30.0 < weight_kg < 200.0:
                            self.callback(weight_kg)
                    except Exception:
                        pass

    async def run(self):
        print("[BLE] Starting passive scale sniffer loop...")
        scanner = BleakScanner(detection_callback = self._detection_callback)
        await scanner.start()
        while self.scanning:
            await asyncio.sleep(1)
        await scanner.stop()
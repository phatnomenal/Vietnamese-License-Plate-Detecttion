# Raspberry Pi Deployment Guide

Deploy the Vietnamese License Plate Detection system on Raspberry Pi with the two pre-trained models.

---

## 📋 Requirements

### Hardware
- **Raspberry Pi 4** (4GB RAM minimum, 8GB recommended) or **Raspberry Pi 5**
- **SD Card**: 32GB or larger (Class 10 recommended)
- **Power Supply**: 5V 3A USB-C (Pi 4) or 5V 5A USB-C (Pi 5)
- **USB Camera**: Standard USB webcam or Pi Camera Module v2+
- **Heat Sink & Fan** (optional but recommended for sustained inference)

### Software
- **OS**: Raspberry Pi OS (Bullseye or Bookworm) - 64-bit recommended
- **Python**: 3.8+ (Python 3.10/3.11 recommended)
- **Git**: For cloning repositories

---

## 🚀 Installation Steps

### Step 1: Prepare Raspberry Pi OS

1. Download and flash **Raspberry Pi OS 64-bit** to SD card using Raspberry Pi Imager
2. Boot the Pi and run initial setup:
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

3. Install system dependencies:
   ```bash
   sudo apt install -y python3-pip python3-dev python3-venv
   sudo apt install -y libatlas-base-dev libjasper-dev libtiff5 libjasper-dev 
   sudo apt install -y libharfbuzz0b libwebp6 libtiff5 libjasper-dev
   sudo apt install -y libopenjp2-7 libtiff5 libjasper-dev libharfbuzz0b libwebp6
   ```

4. Install OpenCV dependencies:
   ```bash
   sudo apt install -y libjasper-dev libatlas-base-dev libtiff5 libjasper-dev \
       libharfbuzz0b libwebp6 libopenjp2-7 libatlas-base-dev
   ```

### Step 2: Clone Project

```bash
cd ~
git clone https://github.com/phatnomenal/Vietnamese-License-Plate-Detecttion.git
cd Vietnamese-License-Plate-Detecttion
```

### Step 3: Create Virtual Environment

```bash
python3 -m venv lp-env
source lp-env/bin/activate  # On Pi, use source instead of Activate.ps1
```

### Step 4: Install Dependencies for Raspberry Pi

Create a `requirements-pi.txt` file with Pi-optimized packages:

```bash
# Download the Pi-specific requirements file
# Or create it manually with the following:
```

**requirements-pi.txt**:
```
numpy==1.24.3
opencv-python==4.8.1.78
torch==2.0.1
torchvision==0.15.2
ultralytics==8.3.29
Pillow>=9.5
matplotlib>=3.7
pandas>=2.1
seaborn>=0.12
pyyaml>=6.0
requests>=2.31
tqdm>=4.66
```

Install packages:
```bash
pip install -r requirements-pi.txt
```

**Note**: This may take 30-60 minutes. Consider using pre-built wheels or running overnight.

### Step 5: Copy Models

Copy the two trained models to the Pi:

```bash
# Method 1: Using SCP (from your PC)
scp models/LP_best.pt pi@raspberrypi.local:~/Vietnamese-License-Plate-Detecttion/models/
scp models/Letters_detection.pt pi@raspberrypi.local:~/Vietnamese-License-Plate-Detecttion/models/

# Method 2: Manual copy via USB or network
# Copy LP_best.pt and Letters_detection.pt to ~/models/
```

### Step 6: Create Raspberry Pi Script

Create `rpi_detect.py` on the Pi:

```python
import cv2
import torch
import numpy as np
import time
import argparse
from ultralytics import YOLO

class LicensePlateDetector:
    def __init__(self, lp_model_path="models/LP_best.pt", 
                 ocr_model_path="models/Letters_detection.pt"):
        """Initialize detector with models"""
        print("[INFO] Loading LP detection model...")
        self.lp_detector = YOLO(lp_model_path)
        
        print("[INFO] Loading OCR model...")
        self.ocr_model = YOLO(ocr_model_path)
        
        # Character mapping
        self.char_map = {
            0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",
            10:"A",11:"B",12:"C",13:"D",14:"E",15:"F",16:"G",17:"H",
            18:"I",19:"J",20:"K",21:"L",22:"M",23:"N",24:"O",25:"P",
            26:"Q",27:"R",28:"S",29:"T",30:"U",31:"V",32:"W",33:"X",
            34:"Y",35:"Z"
        }
        
    def deskew(self, image, cc, ct):
        """Deskew image: 4-direction rotation"""
        if cc == 1:
            image = cv2.rotate(image, cv2.ROTATE_180)
        if ct == 1:
            image = cv2.flip(image, 1)
        return image
    
    def format_plate(self, lines):
        """Format plate from lines"""
        if len(lines) == 1:
            return lines[0]
        line1, line2 = lines
        return f"{line1}{line2}"
    
    def ocr_read(self, crop_img):
        """Read characters from plate crop"""
        result = self.ocr_model(crop_img, conf=0.4)[0]
        boxes = result.boxes
        
        chars = []
        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            if conf < 0.4:
                continue
            chars.append(((x1, y1), self.char_map.get(cls, "?")))
        
        if len(chars) == 0:
            return "unknown"
        
        # Sort by X coordinate
        chars = sorted(chars, key=lambda c: c[0][0])
        
        # Check for 1 or 2 lines
        ys = [c[0][1] for c in chars]
        y_range = max(ys) - min(ys)
        
        if y_range < 15:
            # 1 line
            final = "".join([c[1] for c in chars])
        else:
            # 2 lines
            threshold = np.mean(ys)
            line1 = [c for c in chars if c[0][1] < threshold]
            line2 = [c for c in chars if c[0][1] >= threshold]
            
            line1 = sorted(line1, key=lambda c: c[0][0])
            line2 = sorted(line2, key=lambda c: c[0][0])
            
            text1 = "".join([c[1] for c in line1])
            text2 = "".join([c[1] for c in line2])
            final = self.format_plate([text for text in [text1, text2] if text != ""])
        
        return final
    
    def process_frame(self, frame):
        """Process single frame"""
        plates_detected = []
        
        results = self.lp_detector(frame, conf=0.5)[0]
        boxes = results.boxes
        
        h, w = frame.shape[:2]
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop_img = frame[y1:y2, x1:x2]
            
            # Try 4 deskewing orientations
            lp_text = "unknown"
            found = False
            for cc in range(2):
                for ct in range(2):
                    rotated_crop = self.deskew(crop_img, cc, ct)
                    lp_text = self.ocr_read(rotated_crop)
                    if lp_text != "unknown":
                        plates_detected.append(lp_text)
                        cv2.putText(frame, lp_text, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        found = True
                        break
                if found:
                    break
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,225), 2)
        
        return plates_detected, frame
    
    def run_camera(self, camera_index=0, resolution=(640, 480)):
        """Run detection on camera stream"""
        print(f"[INFO] Starting camera (index={camera_index})...")
        cap = cv2.VideoCapture(camera_index)
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        prev_time = 0
        detected_plates = set()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame")
                    break
                
                # Resize for faster processing
                frame = cv2.resize(frame, resolution)
                
                # Detect plates
                plates, output_frame = self.process_frame(frame)
                detected_plates.update(plates)
                
                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
                prev_time = curr_time
                
                cv2.putText(output_frame, f"FPS: {int(fps)}", (7,70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,0), 2)
                cv2.putText(output_frame, f"Plates: {len(detected_plates)}", (7,110),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,0), 2)
                
                cv2.imshow('License Plate Detection', output_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Quitting...")
                    break
        
        except KeyboardInterrupt:
            print("[INFO] Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"[INFO] Detected plates: {detected_plates}")

def main():
    parser = argparse.ArgumentParser(description="RPi License Plate Detection")
    parser.add_argument("--lp-model", default="models/LP_best.pt", 
                       help="Path to LP detection model")
    parser.add_argument("--ocr-model", default="models/Letters_detection.pt",
                       help="Path to OCR model")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                       help="Camera resolution (width height)")
    
    args = parser.parse_args()
    
    detector = LicensePlateDetector(args.lp_model, args.ocr_model)
    detector.run_camera(args.camera, tuple(args.resolution))

if __name__ == "__main__":
    main()
```

Save as `rpi_detect.py`

### Step 7: Test on Raspberry Pi

```bash
# Connect USB camera and run:
python rpi_detect.py

# With custom paths:
python rpi_detect.py --lp-model models/LP_best.pt --ocr-model models/Letters_detection.pt --camera 0
```

---

## ⚙️ Performance Optimization for Raspberry Pi

### 1. **Reduce Model Inference Size**

Create quantized models for faster inference:

```python
# Convert PyTorch models to TFLite (optional, requires additional setup)
# This significantly reduces model size and improves inference speed
```

### 2. **Lower Resolution**

Modify `rpi_detect.py` to use lower resolution:

```python
detector.run_camera(camera_index=0, resolution=(320, 240))  # Lower resolution
```

### 3. **Reduce Confidence Threshold**

```python
results = self.lp_detector(frame, conf=0.3)[0]  # Lower threshold for faster processing
```

### 4. **Enable GPU/TPU** (if available)

```python
# For Raspberry Pi with TPU:
# Install Coral TPU runtime and use EdgeTPU-optimized models
```

---

## 📊 Expected Performance

| Metric | Pi 4 (4GB) | Pi 5 (8GB) |
|--------|-----------|-----------|
| FPS (640×480) | 2-4 | 4-6 |
| FPS (320×240) | 6-8 | 8-12 |
| Avg Inference Time | 250-500ms | 150-250ms |
| Memory Usage | ~800MB | ~1.2GB |

---

## 🔧 Troubleshooting

### Issue: "Out of Memory" Error
**Solution**:
- Use lower resolution: `resolution=(320, 240)`
- Reduce batch size
- Use swap memory:
  ```bash
  sudo dphys-swapfile swapoff
  sudo nano /etc/dphys-swapfile  # Increase CONF_SWAPSIZE
  sudo dphys-swapfile swapon
  ```

### Issue: Very Slow Inference
**Solution**:
- Lower resolution
- Reduce model confidence threshold
- Use Pi 5 instead of Pi 4
- Consider quantized models

### Issue: Camera Not Detected
**Solution**:
```bash
# List USB cameras
ls -l /dev/video*

# Test camera
v4l2-ctl --list-devices

# Update camera index if needed
python rpi_detect.py --camera 1
```

### Issue: Models Not Found
**Solution**:
```bash
# Verify model paths
ls -l models/LP_best.pt models/Letters_detection.pt

# Use absolute paths if needed
python rpi_detect.py --lp-model /home/pi/models/LP_best.pt --ocr-model /home/pi/models/Letters_detection.pt
```

---

## 📁 Final Pi Directory Structure

```
~/Vietnamese-License-Plate-Detecttion/
├── rpi_detect.py              # Main detection script for Pi
├── requirements-pi.txt        # Pi-specific dependencies
├── models/
│   ├── LP_best.pt            # License plate detector
│   └── Letters_detection.pt   # Character OCR model
├── README.md
└── lp-env/                    # Virtual environment
```

---

## 🚀 Running as a Service (Optional)

Create a systemd service to auto-start on boot:

**Create `/etc/systemd/system/lp-detection.service`**:

```ini
[Unit]
Description=License Plate Detection Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Vietnamese-License-Plate-Detecttion
ExecStart=/home/pi/Vietnamese-License-Plate-Detecttion/lp-env/bin/python rpi_detect.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable lp-detection
sudo systemctl start lp-detection
sudo systemctl status lp-detection
```

---

## 📝 Notes

- **First run may be slow**: Model loading takes time initially
- **USB power**: Ensure adequate power supply for consistent performance
- **Cooling**: Add heatsink/fan for sustained inference
- **Storage**: Models take ~200-300MB total
- **Network**: Consider SSH for remote monitoring

---

**Last Updated**: November 2025
**Version**: 1.0.0

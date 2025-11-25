# Vietnamese License Plate Detection

An advanced **Automatic Number Plate Recognition (ANPR)** system for Vietnamese license plates using YOLOv8. This project combines license plate detection and character recognition (OCR) to extract plate numbers from images and video streams.

## 📋 Overview

This project implements a two-stage detection pipeline:
1. **License Plate Detection**: Locates and crops license plates in images/video
2. **Character Recognition (OCR)**: Recognizes individual characters on the detected plates (digits 0-9 and letters A-Z)

The system supports both single-line and two-line Vietnamese license plate formats with robust character orientation handling.

---

## ✨ Key Features

- **YOLOv8-based Detection**: State-of-the-art object detection for license plates
- **Character OCR**: Recognizes 36 classes (0-9, A-Z)
- **Multi-line Support**: Handles both single and two-line license plate formats
- **Deskewing**: Automatic rotation correction (4-direction deskewing)
- **Real-time Processing**: Camera/video stream support with FPS display
- **Static Image Support**: Process individual images
- **Confidence Filtering**: Filters low-confidence character predictions

---

## 🗂️ Project Structure

```
LP_Detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── webcam.py                          # Real-time camera processing
│
├── data/                              # Training data
│   ├── lp/                            # License plate detection dataset
│   │   ├── lp.yaml                    # LP detection config (1 class)
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   └── ocr/                           # Character OCR dataset
│       ├── ocr.yaml                   # OCR config (36 classes: 0-9, A-Z)
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
│
├── models/                            # Pre-trained models
│   ├── LP_best.pt                     # LP detector (nano/small model)
│   ├── Letters_detection.pt           # Character OCR model
│   ├── implement.py                   # Model implementation utilities
│   └── implement_upgrade.py           # Model upgrade scripts
│
├── training/                          # Jupyter notebooks
│   ├── LP_Detection.ipynb             # LP detection training notebook
│   ├── Letters_Detection.ipynb        # Character OCR training notebook
│   ├── LP_recognition.ipynb           # Recognition pipeline notebook
│   ├── yolov8n.pt                     # Pre-trained YOLOv8 nano weights
│   └── runs/                          # Training results
│       └── train/
│
├── scripts/                           # Utility scripts
│   └── prepare_ocr_labels.py          # OCR label preparation
│
├── src/                               # Source code modules
│   ├── app/                           # Application modules
│   ├── detector/                      # Detection logic
│   ├── ocr/                           # OCR processing
│   └── utils/                         # Utility functions
│
├── experiments/                       # Experiment results
├── runs/                              # Training and inference runs
│   └── train/
│       ├── LP_detection/
│       └── Letters_detection/
│
└── lp-env/                            # Python virtual environment
```

---

## Installation
### Data folder
This repo uses 2 sets of data for 2 stage of license plate recognition problem:

- [License Plate Detection Dataset](https://drive.google.com/file/d/1xchPXf7a1r466ngow_W_9bittRqQEf_T/view?usp=sharing)
- [Character Detection Dataset](https://drive.google.com/file/d/1bPux9J0e1mz-_Jssx4XX1-wPGamaS8mI/view?usp=sharing)
Thanks Mì Ai and winter2897 for sharing a part in this dataset.

### Prerequisites
- Python 3.8+
- CUDA 11.7 (optional, for GPU acceleration)
- Git

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/phatnomenal/Vietnamese-License-Plate-Detecttion.git
   cd LP_Detection
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv lp-env
   .\lp-env\Scripts\Activate.ps1  # Windows PowerShell
   # or
   lp-env\Scripts\activate.bat    # Windows CMD
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: The `requirements.txt` specifies:
   - `torch==2.0.1+cu117` - PyTorch with CUDA 11.7
   - `ultralytics>=8.3.229` - YOLOv8 framework
   - `opencv-python>=4.8` - Computer vision library
   - `matplotlib`, `pandas`, `seaborn` - Data visualization and analysis

---

## 🎯 Usage

### 1. Real-time Camera Processing

Run live detection from your webcam:

```python
python webcam.py
```

**Controls**:
- Press `q` to quit

**Features**:
- Real-time FPS display
- Bounding boxes around detected plates
- Character recognition results displayed on screen

### 2. Process Static Images

Process a single image:

```python
from webcam import process_image

plates = process_image("path/to/image.jpg")
print("Detected plates:", plates)
```

### 3. Model Configuration

Edit model paths in `webcam.py`:
```python
yolo_LP_detect = YOLO("path/to/LP_detector.pt")      # LP detector
yolo_license_plate = YOLO("path/to/OCR_model.pt")    # Character OCR
```

---

## Character Mapping

The OCR model recognizes 36 classes:

| Index | Class | Index | Class | Index | Class |
|-------|-------|-------|-------|-------|-------|
| 0-9   | Digits 0-9 | 10-19 | A-J | 20-29 | K-T |
| 30-35 | U-Z   | -     | -     | -     | -     |

Vietnamese license plates typically contain:
- 2 characters for province code
- 1-2 line formats
- Alphanumeric characters only

---

## Model Details

### License Plate Detection Model
- **Architecture**: YOLOv8 Nano
- **Input Size**: 640×640
- **Output**: Bounding boxes for license plates
- **Classes**: 1 (license_plate)
- **Model File**: `models/LP_best.pt`

### Character OCR Model
- **Architecture**: YOLOv8 Nano
- **Input Size**: 640×640
- **Output**: Character bounding boxes with class predictions
- **Classes**: 36 (0-9, A-Z)
- **Model File**: `models/Letters_detection.pt`

---

## Processing Pipeline

```
Input Image/Frame
       ↓
1. License Plate Detection (YOLOv8)
       ↓
   Extract Plate Crop
       ↓
2. Deskewing (4-direction rotation check)
       ↓
3. Character Detection & Recognition (YOLOv8 OCR)
       ↓
4. Character Sorting & Line Separation
       ↓
5. Format Output (1-line or 2-line format)
       ↓
   Final Plate Number
```

---

## Advanced Features

### Deskewing
Handles rotated license plates by trying 4 orientations:
- `cc=0, ct=0`: Original
- `cc=1, ct=0`: Rotated 180°
- `cc=0, ct=1`: Flipped horizontal
- `cc=1, ct=1`: Rotated + Flipped

### Confidence Filtering
- Character predictions below 0.4 confidence are discarded
- Multi-directional deskewing improves recognition accuracy

### Multi-line Support
Automatically detects if plate has 1 or 2 lines based on Y-coordinate distribution (threshold: 15 pixels)

---

## Training

To train the models on your dataset:

1. **License Plate Detection**:
   ```bash
   jupyter notebook training/LP_Detection.ipynb
   ```

2. **Character OCR**:
   ```bash
   jupyter notebook training/Letters_Detection.ipynb
   ```

3. **Full Recognition Pipeline**:
   ```bash
   jupyter notebook training/LP_recognition.ipynb
   ```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.0.1+cu117 | Deep learning framework |
| torchvision | 0.15.2+cu117 | Computer vision models |
| ultralytics | ≥8.3.229 | YOLOv8 framework |
| opencv-python | ≥4.8 | Image processing |
| numpy | <2 | Numerical computing |
| matplotlib | ≥3.7 | Visualization |
| Pillow | ≥9.5 | Image library |
| pandas | ≥2.1 | Data analysis |
| seaborn | ≥0.12 | Statistical visualization |

---

## System Requirements

- **OS**: Windows, macOS, Linux
- **Python**: 3.8+
- **RAM**: 4GB minimum (8GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional, for acceleration)
- **Storage**: 2GB for models and datasets

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'ultralytics'"
**Solution**: Reinstall requirements
```bash
pip install -r requirements.txt
```

### Issue: Model file not found
**Solution**: Ensure model paths in `webcam.py` are correct:
```python
# Check paths are relative to script location or use absolute paths
yolo_LP_detect = YOLO("models/LP_best.pt")
```

### Issue: Camera not detected
**Solution**: Try different camera indices:
```python
process_camera(camera_index=0)  # Try 0, 1, 2, etc.
```

### Issue: Low recognition accuracy
**Solution**: 
- Ensure good lighting conditions
- Check that license plate is clearly visible
- Try different deskewing orientations
- Adjust confidence threshold in `ocr_read()` function

---

## Notes

- Vietnamese license plates have specific formatting rules
- The system handles both old and new plate formats
- Character recognition works best with clear, well-lit images
- GPU acceleration significantly improves processing speed

---

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests
- Improve documentation

---

## License

This project is part of the Vietnamese License Plate Detection initiative.

---

## Contact

**Repository**: [GitHub - Vietnamese-License-Plate-Detecttion](https://github.com/phatnomenal/Vietnamese-License-Plate-Detecttion)

---

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Official Guide](https://pytorch.org/docs/stable/index.html)
- [OpenCV Documentation](https://docs.opencv.org/)

---

**Last Updated**: November 2025
**Version**: 1.0.0




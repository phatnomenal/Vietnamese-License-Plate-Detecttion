"""
Raspberry Pi License Plate Detection Script
Optimized for Raspberry Pi 4/5 with USB camera

Usage:
    python rpi_detect.py                                    # Default camera, 640x480
    python rpi_detect.py --camera 1 --resolution 320 240   # Camera index 1, 320x240
    python rpi_detect.py --lp-model path/to/model.pt       # Custom model paths
"""

import cv2
import torch
import numpy as np
import time
import argparse
from pathlib import Path
from ultralytics import YOLO


class LicensePlateDetector:
    """License Plate Detection and OCR for Raspberry Pi"""
    
    def __init__(self, lp_model_path="models/LP_best.pt", 
                 ocr_model_path="models/Letters_detection.pt"):
        """
        Initialize detector with models
        
        Args:
            lp_model_path: Path to license plate detection model
            ocr_model_path: Path to character OCR model
        """
        print("[INFO] Loading LP detection model...")
        self.lp_detector = YOLO(lp_model_path)
        
        print("[INFO] Loading OCR model...")
        self.ocr_model = YOLO(ocr_model_path)
        
        # Character mapping: 0-9 (indices 0-9), A-Z (indices 10-35)
        self.char_map = {
            0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9",
            10:"A", 11:"B", 12:"C", 13:"D", 14:"E", 15:"F", 16:"G", 17:"H",
            18:"I", 19:"J", 20:"K", 21:"L", 22:"M", 23:"N", 24:"O", 25:"P",
            26:"Q", 27:"R", 28:"S", 29:"T", 30:"U", 31:"V", 32:"W", 33:"X",
            34:"Y", 35:"Z"
        }
        print("[INFO] Models loaded successfully!")
    
    def deskew(self, image, cc, ct):
        """
        Deskew image by trying 4 orientations
        
        Args:
            image: Input image
            cc: Rotation flag (0=original, 1=180 degrees)
            ct: Flip flag (0=no flip, 1=horizontal flip)
        
        Returns:
            Transformed image
        """
        if cc == 1:
            image = cv2.rotate(image, cv2.ROTATE_180)
        if ct == 1:
            image = cv2.flip(image, 1)
        return image
    
    def format_plate(self, lines):
        """
        Format detected text lines into plate number
        
        Args:
            lines: List of text strings
        
        Returns:
            Formatted plate number
        """
        if len(lines) == 1:
            return lines[0]
        line1, line2 = lines
        return f"{line1}{line2}"
    
    def ocr_read(self, crop_img, conf_threshold=0.4):
        """
        Read characters from license plate crop
        
        Args:
            crop_img: Cropped license plate image
            conf_threshold: Confidence threshold for character detection
        
        Returns:
            Recognized plate text
        """
        result = self.ocr_model(crop_img, conf=conf_threshold)[0]
        boxes = result.boxes
        
        if len(boxes) == 0:
            return "unknown"
        
        chars = []
        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            
            if conf < conf_threshold:
                continue
            
            chars.append(((x1, y1), self.char_map.get(cls, "?")))
        
        if len(chars) == 0:
            return "unknown"
        
        # Sort characters by X coordinate (left to right)
        chars = sorted(chars, key=lambda c: c[0][0])
        
        # Determine if 1 line or 2 lines based on Y distribution
        ys = [c[0][1] for c in chars]
        y_range = max(ys) - min(ys)
        y_threshold = 15  # Pixel threshold for line separation
        
        if y_range < y_threshold:
            # Single line plate
            final = "".join([c[1] for c in chars])
        else:
            # Multi-line plate (typically Vietnamese 2-line format)
            y_mean = np.mean(ys)
            line1 = [c for c in chars if c[0][1] < y_mean]
            line2 = [c for c in chars if c[0][1] >= y_mean]
            
            # Sort each line left to right
            line1 = sorted(line1, key=lambda c: c[0][0])
            line2 = sorted(line2, key=lambda c: c[0][0])
            
            text1 = "".join([c[1] for c in line1])
            text2 = "".join([c[1] for c in line2])
            final = self.format_plate([text for text in [text1, text2] if text != ""])
        
        return final
    
    def process_frame(self, frame, conf_lp=0.5, conf_ocr=0.4):
        """
        Process single frame for license plate detection
        
        Args:
            frame: Input image frame
            conf_lp: Confidence threshold for plate detection
            conf_ocr: Confidence threshold for character detection
        
        Returns:
            Tuple of (detected_plates, annotated_frame)
        """
        plates_detected = []
        
        # Detect license plates
        results = self.lp_detector(frame, conf=conf_lp)[0]
        boxes = results.boxes
        
        h, w = frame.shape[:2]
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop_img = frame[y1:y2, x1:x2]
            
            if crop_img.size == 0:
                continue
            
            # Try 4 deskewing orientations
            lp_text = "unknown"
            found = False
            
            for cc in range(2):
                for ct in range(2):
                    rotated_crop = self.deskew(crop_img, cc, ct)
                    lp_text = self.ocr_read(rotated_crop, conf_ocr)
                    
                    if lp_text != "unknown":
                        plates_detected.append(lp_text)
                        # Draw recognized text on frame
                        cv2.putText(frame, lp_text, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        found = True
                        break
                
                if found:
                    break
            
            # Draw bounding box around detected plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 225), 2)
        
        return plates_detected, frame
    
    def run_camera(self, camera_index=0, resolution=(640, 480), conf_lp=0.5, conf_ocr=0.4):
        """
        Run license plate detection on camera stream
        
        Args:
            camera_index: Index of camera device
            resolution: Tuple of (width, height)
            conf_lp: Confidence threshold for plate detection
            conf_ocr: Confidence threshold for character detection
        """
        print(f"[INFO] Starting camera (index={camera_index}, resolution={resolution})...")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("[ERROR] Failed to open camera!")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Optional: Set FPS
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        prev_time = 0
        detected_plates = set()
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame")
                    break
                
                frame_count += 1
                
                # Resize frame to target resolution
                frame = cv2.resize(frame, resolution)
                
                # Detect plates and OCR
                plates, output_frame = self.process_frame(frame, conf_lp, conf_ocr)
                detected_plates.update(plates)
                
                # Calculate and display FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
                prev_time = curr_time
                
                # Display FPS
                cv2.putText(output_frame, f"FPS: {int(fps)}", (7, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)
                
                # Display unique plates count
                cv2.putText(output_frame, f"Plates Found: {len(detected_plates)}", (7, 110),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)
                
                # Display frame count
                cv2.putText(output_frame, f"Frames: {frame_count}", (7, 150),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)
                
                # Show window
                cv2.imshow('License Plate Detection (Press Q to Quit)', output_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Quitting...")
                    break
        
        except KeyboardInterrupt:
            print("[INFO] Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n[INFO] Detection Summary:")
            print(f"  - Total frames: {frame_count}")
            print(f"  - Unique plates detected: {len(detected_plates)}")
            print(f"  - Detected plates: {detected_plates}")


def main():
    parser = argparse.ArgumentParser(
        description="License Plate Detection for Raspberry Pi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rpi_detect.py
  python rpi_detect.py --camera 1 --resolution 320 240
  python rpi_detect.py --lp-model custom_lp_model.pt --ocr-model custom_ocr_model.pt
        """
    )
    
    parser.add_argument("--lp-model", type=str, default="models/LP_best.pt", 
                       help="Path to license plate detection model (default: models/LP_best.pt)")
    parser.add_argument("--ocr-model", type=str, default="models/Letters_detection.pt",
                       help="Path to character OCR model (default: models/Letters_detection.pt)")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index (default: 0)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                       metavar=('WIDTH', 'HEIGHT'),
                       help="Camera resolution in pixels (default: 640 480)")
    parser.add_argument("--conf-lp", type=float, default=0.5,
                       help="Confidence threshold for plate detection (default: 0.5)")
    parser.add_argument("--conf-ocr", type=float, default=0.4,
                       help="Confidence threshold for character detection (default: 0.4)")
    
    args = parser.parse_args()
    
    # Validate model paths
    lp_path = Path(args.lp_model)
    ocr_path = Path(args.ocr_model)
    
    if not lp_path.exists():
        print(f"[ERROR] LP model not found: {args.lp_model}")
        return
    if not ocr_path.exists():
        print(f"[ERROR] OCR model not found: {args.ocr_model}")
        return
    
    # Initialize detector
    detector = LicensePlateDetector(args.lp_model, args.ocr_model)
    
    # Run detection
    detector.run_camera(
        camera_index=args.camera,
        resolution=tuple(args.resolution),
        conf_lp=args.conf_lp,
        conf_ocr=args.conf_ocr
    )


if __name__ == "__main__":
    main()

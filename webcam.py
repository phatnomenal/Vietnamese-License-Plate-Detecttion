import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO

# ------------------------------
# LOAD YOLOv8 MODELS
# ------------------------------
yolo_LP_detect = YOLO("model/LP_detector_nano_61.pt")      # YOLOv8 LP detector
yolo_license_plate = YOLO("model/LP_ocr_nano_62.pt")       # YOLOv8 OCR

CHAR_MAP = {
    0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",
    10:"A",11:"B",12:"C",13:"D",14:"E",15:"F",16:"G",17:"H",
    18:"I",19:"J",20:"K",21:"L",22:"M",23:"N",24:"O",25:"P",
    26:"Q",27:"R",28:"S",29:"T",30:"U",31:"V",32:"W",33:"X",
    34:"Y",35:"Z"
}

# ------------------------------
# HELPER: deskew (xoay 4 hướng)
# ------------------------------
def deskew(image, cc, ct):
    """
    cc, ct: 0 hoặc 1
    - 0: không xoay, 1: xoay 180
    """
    if cc == 1:
        image = cv2.rotate(image, cv2.ROTATE_180)
    if ct == 1:
        image = cv2.flip(image, 1)
    return image

# ------------------------------
# HELPER: format plate
# ------------------------------
def format_plate(lines):
    if len(lines) == 1:
        return lines[0]
    line1, line2 = lines
    return f"{line1}{line2}"

# ------------------------------
# OCR READ (linh hoạt 1 hoặc 2 dòng)
# ------------------------------
def ocr_read(ocr_model, crop_img):
    # YOLOv8 OCR inference
    result = ocr_model(crop_img)[0]
    boxes = result.boxes

    chars = []
    for box in boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        if conf < 0.4:
            continue
        chars.append(((x1, y1), CHAR_MAP.get(cls, "?")))

    if len(chars) == 0:
        return "unknown"

    # sort theo X
    chars = sorted(chars, key=lambda c: c[0][0])

    # check Y range để quyết định 1 hay 2 dòng
    ys = [c[0][1] for c in chars]
    y_range = max(ys) - min(ys)
    threshold_y = 15

    if y_range < threshold_y:
        # 1 dòng
        final = "".join([c[1] for c in chars])
    else:
        # 2 dòng
        threshold = np.mean(ys)
        line1 = [c for c in chars if c[0][1] < threshold]
        line2 = [c for c in chars if c[0][1] >= threshold]

        line1 = sorted(line1, key=lambda c: c[0][0])
        line2 = sorted(line2, key=lambda c: c[0][0])

        # đảm bảo dòng trên có Y nhỏ hơn dòng dưới
        if line1 and line2:
            if np.mean([c[0][1] for c in line1]) > np.mean([c[0][1] for c in line2]):
                line1, line2 = line2, line1

        text1 = "".join([c[1] for c in line1])
        text2 = "".join([c[1] for c in line2])
        final = format_plate([text for text in [text1, text2] if text != ""])

    return final

# ------------------------------
# XỬ LÝ 1 FRAME
# ------------------------------
def process_frame(frame):
    list_read_plates = set()

    results = yolo_LP_detect(frame)[0]
    plates = results.boxes

    h, w = frame.shape[:2]

    for p in plates:
        x1, y1, x2, y2 = map(int, p.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop_img = frame[y1:y2, x1:x2]

        # Deskew + OCR 4 hướng
        lp_text = "unknown"
        flag = False
        for cc in range(2):
            for ct in range(2):
                rotated_crop = deskew(crop_img, cc, ct)
                lp_text = ocr_read(yolo_license_plate, rotated_crop)
                if lp_text != "unknown":
                    list_read_plates.add(lp_text)
                    cv2.putText(frame, lp_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    flag = True
                    break
            if flag:
                break

        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,225), 2)

    return list_read_plates, frame

# ------------------------------
# XỬ LÝ CAMERA
# ------------------------------
def process_camera(camera_index=0):
    vid = cv2.VideoCapture(camera_index)
    prev_frame_time = 0

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        plates, out_frame = process_frame(frame)

        # FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        cv2.putText(out_frame, f"FPS: {int(fps)}", (7,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100,255,0), 3, cv2.LINE_AA)

        cv2.imshow('License Plate Detection', out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

# ------------------------------
# XỬ LÝ ẢNH TĨNH
# ------------------------------
def process_image(img_path):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Không đọc được ảnh: {img_path}")
        return
    plates, out_frame = process_frame(frame)
    print("Biển số nhận diện:", plates)
    cv2.imshow('Result', out_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return plates

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    # Test camera
    # process_camera(camera_index=1)

    # Test ảnh tĩnh
    test_image = "test.jpg"
    process_image(test_image)

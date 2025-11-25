import cv2
import torch
import time
import function.utils_rotate as utils_rotate
import function.helper as helper

# ------------------------------
# LOAD YOLOv8 MODEL
# ------------------------------
from ultralytics import YOLO

yolo_LP_detect = YOLO("model/LP_best.pt")      # YOLOv8 LP detector
yolo_license_plate = YOLO("model/Letters_detection.pt")       # YOLOv8 OCR

# ------------------------------
# HÀM XỬ LÝ 1 FRAME
# ------------------------------
def process_frame(frame):
    list_read_plates = set()

    # Phát hiện biển số
    results = yolo_LP_detect(frame)[0]      # YOLOv8 trả về list
    plates = results.boxes

    h, w = frame.shape[:2]

    for p in plates:
        x1, y1, x2, y2 = map(int, p.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop_img = frame[y1:y2, x1:x2]

        # Deskew + OCR
        lp_text = ""
        flag = False
        for cc in range(2):
            for ct in range(2):
                rotated_crop = utils_rotate.deskew(crop_img, cc, ct)
                lp_text = helper.read_plate(yolo_license_plate, rotated_crop)
                if lp_text != "unknown":
                    list_read_plates.add(lp_text)
                    cv2.putText(frame, lp_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    flag = True
                    break
            if flag:
                break

        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0,0,225), thickness=2)

    return list_read_plates, frame

# ------------------------------
# HÀM XỬ LÝ CAMERA
# ------------------------------
def process_camera(camera_index=0):
    vid = cv2.VideoCapture(camera_index)
    prev_frame_time = 0

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        plates, out_frame = process_frame(frame)

        # Tính FPS
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
# HÀM XỬ LÝ ẢNH TĨNH
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
# MAIN TEST
# ------------------------------
if __name__ == "__main__":
    # Test camera
    # process_camera(camera_index=1)

    # Hoặc test với ảnh tĩnh
    test_image = "557.png"
    process_image(test_image)

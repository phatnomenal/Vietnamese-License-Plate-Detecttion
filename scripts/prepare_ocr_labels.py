import os
import csv

def create_csv(data_dir, output_file):
    images = []
    for split in ['train', 'val']:
        img_dir = os.path.join(data_dir, 'images', split)
        lbl_dir = os.path.join(data_dir, 'labels', split)
        for img_name in os.listdir(img_dir):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(img_dir, img_name)
                label_name = os.path.splitext(img_name)[0] + '.txt'
                label_path = os.path.join(lbl_dir, label_name)
                if os.path.exists(label_path):
                    with open(label_path, 'r', encoding='utf-8') as f:
                        label_text = f.read().strip()
                    images.append([img_path.replace("\\", "/"), label_text])

    # Write CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label'])
        writer.writerows(images)
    print(f"CSV saved to {output_file}")

if __name__ == "__main__":
    create_csv("../data/ocr", "../data/ocr/labels.csv")

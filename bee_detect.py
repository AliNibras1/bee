import torch
import os
import cv2
from pathlib import Path

# Paths
YOLO_PATH = r"C:\Users\QASR-PC BK Z\20226205\yolov5"
MODEL_PATH = r"C:\Users\QASR-PC BK Z\20226205\best.pt"
IMAGE_PATH = r"C:\Users\QASR-PC BK Z\20226205\bee_detected_image.jpg"
# Check if image exists
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Image not found: {IMAGE_PATH}")
    exit()

# Load model
model = torch.hub.load(YOLO_PATH, 'custom', path=MODEL_PATH, source='local')
model.conf = 0.2  

# Run detection
results = model(IMAGE_PATH)
detections = results.pandas().xywh[0]

# Show columns
print("\n🧾 Columns:", detections.columns.tolist())

# Show all detection results
print("\n🔍 Detection Results:")
print(detections[['xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name']])

# Filter bees by class name
bees_detected = detections[detections['name'] == 'bee']

if bees_detected.empty:
    print("❌ No bees detected.")
else:
    num_bees = len(bees_detected)
    avg_conf = bees_detected['confidence'].mean() * 100

    print(f"\n🐝 Bee Detection Summary:")
    print(f" - Total Bees: {num_bees}")
    print(f" - Avg Confidence: {avg_conf:.2f}%")

    # Draw count on image
    img = cv2.imread(IMAGE_PATH)
    cv2.putText(img, f"Total Bees: {num_bees}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Save image
    output_dir = "runs/detect_with_count"
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = Path(output_dir) / Path(IMAGE_PATH).name
    cv2.imwrite(str(output_image_path), img)

    results.show()
    print(f"\n✔️ Saved image with count: {output_image_path}")

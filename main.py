import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("model.pt")

with open("labels.txt", "r") as f:
    classLabels = [line.strip() for line in f.readlines()]

# Images
#img = cv2.imread("crowd.jpg")
#img = cv2.imread("fire.jpg")
#img = cv2.imread("gun.jpg")
#img = cv2.imread("heads.jpg")
img = cv2.imread("person_with_knife.jpg")
#img = cv2.imread("unconsciousness.jpg")

results = model(img)[0]

confidence = results.boxes.conf.cpu().numpy()
class_ids = results.boxes.cls.cpu().numpy().astype(int)
bbox = results.boxes.xyxy.cpu().numpy().astype(int)


img_with_boxes = img.copy()
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

for cls_id, conf, box in zip(class_ids, confidence, bbox):
    x1, y1, x2, y2 = box
    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{classLabels[cls_id]} {conf:.2f}"
    cv2.putText(img_with_boxes, label, (x1 + 10, y1 + 40),
                font, fontScale=font_scale, color=(0, 0, 255), thickness=2)


plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
plt.title("Detected Objects")
plt.axis("off")
plt.show()

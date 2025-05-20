import cv2
from ultralytics import YOLO

model = YOLO("model.pt")

with open("labels.txt", "r") as f:
    classLabels = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No camera frame, the end.")
        break

    results = model(frame)[0]

    confidence = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    bbox = results.boxes.xyxy.cpu().numpy().astype(int)

    for cls_id, conf, box in zip(class_ids, confidence, bbox):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{classLabels[cls_id]} {conf:.2f}"
        cv2.putText(frame, label, (x1 + 10, y1 + 40),
                    cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(0, 0, 255), thickness=2)

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

model = YOLO("model.pt")

with open("labels.txt", "r") as f:
    classLabels = [line.strip() for line in f.readlines()]

video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video file.")
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

    cv2.imshow("YOLO Detection on Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

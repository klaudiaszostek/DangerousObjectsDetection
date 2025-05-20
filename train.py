from ultralytics import YOLO

model = YOLO("yolo11n.pt")

data_path = r"C:\Users\klaud\Downloads\dangerous.v4i.yolov8\data.yaml"

train_results = model.train(
    data=data_path,
    epochs=10,
    imgsz=640,
    device="cpu"
)
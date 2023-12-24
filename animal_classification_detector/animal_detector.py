from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("best.pt")  # загрузите предварительно обученную модель YOLOv8n
    model.predict(source="*.jpg")
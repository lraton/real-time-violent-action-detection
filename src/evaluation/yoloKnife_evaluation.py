from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("../../models/knife/run2/weights/best.pt")

    metrics = model.val(data="../../datasets/knife-dataset/data.yaml", split="test")

    print(metrics)

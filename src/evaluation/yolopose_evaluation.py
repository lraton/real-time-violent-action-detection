from ultralytics import YOLO
from ultralytics.utils.downloads import download

if __name__ == '__main__':
    # URL for COCO8-Pose tiny 8-image dataset
    dataset_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip"

    # Download and unzip if you don't have it
    download(dataset_url, dir="../../datasets")

    model = YOLO("../../models/yolo11n-pose.pt")

    # validate on the tiny dataset first
    metrics = model.val(data="coco8-pose.yaml", split="val")

    print(f"mAP50-95: {metrics.pose.map}")
    print(f"mAP50: {metrics.pose.map50}")

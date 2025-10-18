from yolo_gui_app import YOLOCameraApp

app = YOLOCameraApp("models/knife/runs/detect/train3/weights/best.pt", "models/yolo11n-pose.pt") #(object detect, pose detect) yolo11n.pt for object 
app.run()

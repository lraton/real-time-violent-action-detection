from yolo_gui_app import YOLOCameraApp

app = YOLOCameraApp("models/knife/weights/best.pt", "models/pose/weights/yolo11x-pose.pt") #(object detect, pose detect) yolo11n.pt for object 
app.run()

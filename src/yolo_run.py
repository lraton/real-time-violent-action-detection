from yolo_gui_app import YOLOCameraApp

app = YOLOCameraApp("yolo11n.pt", "yolo11n-pose.pt") #(object detect, pose detect) Change object with knife dectetion
app.run()

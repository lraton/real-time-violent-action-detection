from roboflow import Roboflow
rf = Roboflow(api_key="L734pXjCCEwZGBPUFknp")
project = rf.workspace("tesi-c7kla").project("knife-fwpuw-rrrcw")
version = project.version(1)
dataset = version.download("yolov11")
                
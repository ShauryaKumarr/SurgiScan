from roboflow import Roboflow
rf = Roboflow(api_key="xM5wu5m3Nogv0OhU9exS")
project = rf.workspace("surgiscan").project("utensil-detection")
version = project.version(1)
dataset = version.download("yolov5")
                
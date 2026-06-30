from lib import *

model = YOLO('yolov8n.pt')

model.train(
    data= 'yolo/lettuce.v1i.yolov8/data.yaml',
    epochs = 100,
    imgsz = 640,
    batch = 32,
    device = 'mps'
)


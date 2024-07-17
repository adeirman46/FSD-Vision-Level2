import sys
sys.path.insert(0, "/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics")

from ultralytics import YOLO
import cv2

img = cv2.imread('/home/irman/Documents/FSD-Level-1/vision_control/videos/jalan_tol_new.png')


number = 3 #input how many tasks in your work
model = YOLO('/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics/v4s.onnx', task='multi')  # Validate the model
results = model.predict(source=img, imgsz=(384,672), device=[0],name='v4_daytime', save=True, conf=0.25, iou=0.45, show_labels=False, stream=True, save_txt=True)


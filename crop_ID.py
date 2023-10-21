from ultralytics import YOLO
import cv2
import sys
import shutil

folder_path = "./runs/predict"

def crop_cin(img_path):
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        print(f"Error deleteing folder: {e}")
    # Load Model
    model = YOLO("yolov8n.pt")
    img = cv2.imread(img_path)
    results = model.predict(source=img)
    for result in results:
        result.save_crop("runs/predict")

def crop_cins(fold_path):
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        print(f"Error deleteing folder: {e}")
    model = YOLO("yolov8n.pt")
    results = model.predict(source=fold_path)
    for result in results:
        result.save_crop("runs/predict")


crop_cins(sys.argv[1])
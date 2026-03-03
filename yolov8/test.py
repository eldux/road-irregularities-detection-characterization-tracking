from ultralytics import YOLO

# Load a model
model = YOLO("/home/tlkc/Research/Speedbump-detection/yolov8/runs/detect/train2/weights/best.pt")  # build a new model from YAML

# Train the model
results = model.val(data="/home/tlkc/Research/Non-regularities-detection/non-regularities-detection/data/raw/data_local.yaml", imgsz=640, half=True, batch=1, device='cuda')

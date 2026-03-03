from ultralytics import YOLO

# Load a model
model = YOLO("/home/tlkc/Research/Non-regularities-detection/non_regularities_detector3/run_non_regularities_detector33/weights/best.pt")  # build a new model from YAML

# Train the model
results = model.val(data="/home/tlkc/Research/Non-regularities-detection/non-regularities-detection/data/raw/data_local.yaml", imgsz=640, batch=32, device='cuda')
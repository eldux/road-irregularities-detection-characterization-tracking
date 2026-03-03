from ultralytics import YOLO

# Load a model
model = YOLO("yolo12s.pt")  # build a new model from YAML

# Train the model
results = model.train(data="/home/tlkc/Research/Non-regularities-detection/non-regularities-detection/data/raw/data_local.yaml", epochs=300, imgsz=640, batch=8, device='cuda')

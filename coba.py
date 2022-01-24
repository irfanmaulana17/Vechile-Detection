import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = '.\data\images\malioboro.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.save()  # or .show(), .save(), .crop(), .pandas(), etc.

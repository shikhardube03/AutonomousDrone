import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

model = torch.hub.load('models/yolov5m.yaml', 'yolov5m')

img = "https://ultralytics.com/images/zidane.jpg"

results = model(img)

results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
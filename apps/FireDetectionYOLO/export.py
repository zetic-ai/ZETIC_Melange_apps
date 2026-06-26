from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import numpy as np
import os

print('Downloading model...')
path = hf_hub_download(repo_id='leeyunjai/yolo11-firedetect', filename='firedetect-11s.pt')
print(f'Downloaded to: {path}')

model = YOLO(path)
print('Model loaded')

result = model.export(format='onnx', imgsz=640, opset=12, simplify=True, dynamic=False, half=False)
print(f'Export result: {result}')

sample = np.random.rand(1, 3, 640, 640).astype(np.float32)
np.save('sample_input.npy', sample)
print('sample_input.npy saved') 
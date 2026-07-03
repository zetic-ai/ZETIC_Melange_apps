"""SafetyPPEYOLO — Stage-0 export recipe (YOLO family, PyroGuard recipe).

Winner: ayushgupta7777/safetyvision-yolov8, v2/best.pt (YOLOv8s, 13 PPE classes).
Produces safetyppe-8s.onnx (float32[1,3,640,640] -> float32[1,17,8400], static, opset 12)
and sample_input.npy.
"""
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import numpy as np
import shutil

print('Downloading model...')
path = hf_hub_download(repo_id='ayushgupta7777/safetyvision-yolov8', filename='v2/best.pt')
print(f'Downloaded to: {path}')

model = YOLO(path)
print('Model loaded')
print('Classes:', model.names)

result = model.export(format='onnx', imgsz=640, opset=12, simplify=True, dynamic=False, half=False)
print(f'Export result: {result}')
shutil.copy(result, 'safetyppe-8s.onnx')
print('Saved as safetyppe-8s.onnx')

sample = np.random.rand(1, 3, 640, 640).astype(np.float32)
np.save('sample_input.npy', sample)
print('sample_input.npy saved')

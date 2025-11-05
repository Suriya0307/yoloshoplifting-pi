import sys
print('python executable:', sys.executable)
try:
    import cv2
    print('cv2', cv2.__version__)
except Exception as e:
    print('cv2 import failed:', e)
try:
    import torch
    print('torch', torch.__version__)
except Exception as e:
    print('torch import failed:', e)
try:
    from ultralytics import YOLO
    print('ultralytics OK')
except Exception as e:
    print('ultralytics import failed:', e)

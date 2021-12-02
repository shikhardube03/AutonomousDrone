import argparse
import os
import sys
from pathlib import Path
import onnxruntime
import cv2
import numpy as np
import torch
from pathlib import Path
import sys
import torch.backends.cudnn as cudnn
import os
weights = 'yolov5s.onnx'
w = str(weights[0] if isinstance(weights, list) else weights)
session = onnxruntime.InferenceSession(w, None)
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def check_imshow():
    # Check if environment supports image displays
    try:
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False

img0 = cv2.imread('/Users/karoliskrakys/PycharmProjects/Group52-FA21/Project/Resources/Images/IMG_0776.jpg')  # BGR
img0 = cv2.resize(img0, (640,640))
img = letterbox(img0, (640,640), stride=32, auto=True)[0]
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)
img = img.astype('float32')
img /= 255.0
if len(img.shape) == 3:
    img = img[None]
# print(session.get_inputs()[0].name)
print('-'*50)
print(session.get_outputs()[0].name)
pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

print(pred)
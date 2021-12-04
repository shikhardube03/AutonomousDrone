from yolov5m.detect2 import run, parse_opt
import djitellopy as tello
import cv2
import numpy as np
import time

# Intitialize drone
me = tello.Tello()
me.connect()
print(me.get_battery())

global img

# Initiate video stream
me.streamon()

# Stream the video stream with object detection on screen
while True:
    img = me.get_frame_read().frame
    print(img.shape)
    h, w, ch = img.shape
    #1, ch, h, w
    img = np.array(img, (h, w, ch))
    np.reshape(img)
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    #print(img)
    opt = parse_opt()
    img_detect, dist = run(im=img)
    print(dist)
    #img_show = img_detect.render()
    #v2.imshow("Image", img_show[0])
    cv2.waitKey(1)
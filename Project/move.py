from droneStream import Tello as tello
import cv2



me = tello()
me.connect()
print(me.get_battery())
me.streamon()
me.takeoff()
me.rotate_counter_clockwise(40)
img = me.get_frame_read()
cv2.imshow("Output", img)

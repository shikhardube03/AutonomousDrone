from djitellopy import tello
from time import sleep

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
me.takeoff()
me.rotate_counter_clockwise(40)

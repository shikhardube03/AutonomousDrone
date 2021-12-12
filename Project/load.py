from djitellopy import tello
from time import sleep
global me
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
#sleep(15)
while True:
    sleep(0.05)

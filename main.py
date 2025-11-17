import time
from adafruit_servokit import ServoKit

kit = ServoKit(channels=8)
# setup 
kit.servo[0].angle = 0
kit.servo[1].angle = 0
time.sleep(5)
# set home
kit.servo[0].angle = 90
kit.servo[1].angle = 90


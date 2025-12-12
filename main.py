import time
from adafruit_servokit import ServoKit

kit = ServoKit(channels=8)

# setup 
kit.servo[0].angle = 0
# kit.continuous_servo[1].throttle = 0
time.sleep(.5)
# set home
# kit.servo[0].angle = 90
# kit.continuous_servo[1].throttle = 1


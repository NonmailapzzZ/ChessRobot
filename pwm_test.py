# Channel
import time
import board
import busio
from adafruit_pca9685 import PCA9685

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50


SERVO_CH = 0

# ===== ????????????? calibrate ??? =====
PWM_MIN_US = 500      
PWM_MAX_US = 2500     
ANGLE_MAX = 300       
# ======================================

def us_to_duty(us):
    return int(us * 65535 / 20000)

def angle_to_duty(angle):
    angle = max(0, min(ANGLE_MAX, angle))
    us = PWM_MIN_US + (angle / ANGLE_MAX) * (PWM_MAX_US - PWM_MIN_US)
    return us_to_duty(us)


def move_slow(start, target, step=1, delay=0.02):
    for angle in range(int(start), target + 1, step):
        pca.channels[SERVO_CH].duty_cycle = angle_to_duty(angle)
        time.sleep(delay)
    
pca.channels[SERVO_CH].duty_cycle = 500
time.sleep(5)
move_slow(0, 90)

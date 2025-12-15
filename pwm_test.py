import time
from adafruit_pca9685 import PCA9685
import busio
import board


# I2C
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c) 
pca.frequency = 50

# Channel
SERVO_CH = 0

PWM_MIN_US = 500
PWM_MAX_US = 2500
ANGLE_MAX = 300

def us_to_duty(us):
    return int(us * (2**12) / 20000)

def angle_to_duty(angle):
    us = PWM_MIN_US + (angle / ANGLE_MAX) * (PWM_MAX_US - PWM_MIN_US)
    return us_to_duty(us)

def move_servo(angle):
    pca.channels[SERVO_CH].duty_cycle = angle_to_duty(angle)
    
    
    
move_servo(90)
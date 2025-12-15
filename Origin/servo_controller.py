# = = = = = = = Const. = = = = = = = =
import time
from adafruit_servokit import ServoKit
from numpy import abs
import time
from adafruit_pca9685 import PCA9685
import busio
import board

kit = ServoKit(channels=16)
# I2C
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c) 
pca.frequency = 50
SERVO_CH = 0

position = 0

PWM_MIN_US = 500      
PWM_MAX_US = 2500     
ANGLE_MAX = 300       
# = = = = = = = = = = = = = = = = = = = = 
 
 
# = = = = = = = Calculate. = = = = = = = =
 
def duty_to_us(duty):
    return duty * 20000 / 65535

def us_to_duty(us):
    return int(us * 65535 / 20000)

def angle_to_duty(angle):
    angle = max(0, min(ANGLE_MAX, angle))
    us = PWM_MIN_US + (angle / ANGLE_MAX) * (PWM_MAX_US - PWM_MIN_US)
    return us_to_duty(us)

def duty_to_angle(duty):
    us = duty_to_us(duty)
    angle = (us - PWM_MIN_US) * ANGLE_MAX / (PWM_MAX_US - PWM_MIN_US)
    return max(0, min(ANGLE_MAX, angle))
# = = = = = = = = = = = = = = = = = = = = 

def move_slow_link1(target, step=1, delay=0.04):
    target = int(180 - target + 89)
    SERVO_CH = 0
    current = pca.channels[SERVO_CH].duty_cycle
    current = duty_to_angle(current)
    
    if current < target:
        for angle in range(int(current), target + 5, step):
            pca.channels[SERVO_CH].duty_cycle = angle_to_duty(angle) - 44
            time.sleep(delay)
    else:
        for angle in range(int(current), target - 5, -step):
            pca.channels[SERVO_CH].duty_cycle = angle_to_duty(angle) - 88
            time.sleep(delay)

class move_slow_link2 :
    def __init__(self, kit=kit, channel=2, deg_per_sec=360/7.39, forward_servo_speed = 0.2, backward_servo_speed = -0.31):
        self.kit = kit
        self.channel = channel
        self.deg_per_sec = deg_per_sec
        self.pos = 0
        self.forward_speed = forward_servo_speed
        self.backward_speed = backward_servo_speed
    
    def move(self, target_deg):
        delta = int(target_deg) - self.pos
        duration = abs(delta) / self.deg_per_sec
        self.kit.continuous_servo[self.channel].throttle = self.backward_speed if delta < 0 else self.forward_speed
        time.sleep(duration)
        self.kit.continuous_servo[self.channel].throttle = 0
        # update pos
        self.pos = target_deg
    
def move_slow_slider(target, step=1, delay=0.04):
    target = int(target)
    channel = 4
    current = kit.servo[channel].angle
    if current < target:
        for angle in range(int(current), target + 1, step):
            kit.servo[channel].angle = angle
            time.sleep(delay)
    else:
        for angle in range(int(current), target - 1, -step):
            kit.servo[channel].angle = angle
            time.sleep(delay)

def calibration_servo() :
    kit.servo["link1"].angle = 180
    kit.servo["slider"].angle = 70

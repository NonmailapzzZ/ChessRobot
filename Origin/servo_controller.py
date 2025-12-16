# = = = = = = = Const. = = = = = = = =
import time
from numpy import abs
import RPi.GPIO as GPIO

# ============ GPIO SETUP ==============
GPIO.setmode(GPIO.BCM)
servo_pins = [12, 13, 18, 19]
GPIO.setup(servo_pins, GPIO.OUT)

# ============ PWM 50 HZ ================
servos = [GPIO.PWM(pin, 50) for pin in servo_pins]
 
 
# ====== CONFIG SERVO ======
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 300
SERVO_MIN_DUTY  = 2.5
SERVO_MAX_DUTY  = 12.5


# = = = = = = = Calculate. = = = = = = = =
def angle_to_duty(angle):
    angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))
    duty = SERVO_MIN_DUTY + (angle / SERVO_MAX_ANGLE) * (SERVO_MAX_DUTY - SERVO_MIN_DUTY)
    return duty

# = = = = = = = SERVO ROTATE = = = = = = = = = 
def move_slow_link1(target, step=1, delay=0.02):
    target = int(180 - target + 85)
    servo_index = 0
    
    if 'current_angle' not in move_slow_link1.__dict__:
        move_slow_link1.current_angle = [150]
        
    current = move_slow_link1.current_angle[0]
    step = 1 if current < target else -1
    
    for angle in range(int(current), target + step, step):
        servos[servo_index].ChangeDutyCycle(angle_to_duty(angle))
        time.sleep(delay)
    
    move_slow_link1.current_angle[servo_index] = target

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
  
def calibration_servo() :
    kit.servo["link1"].angle = 180
    kit.servo["slider"].angle = 70

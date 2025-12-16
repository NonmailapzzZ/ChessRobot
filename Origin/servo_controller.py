# = = = = = = = Const. = = = = = = = =
import time
from numpy import abs
import RPi.GPIO as GPIO

# ============ GPIO SETUP ==============
GPIO.setmode(GPIO.BCM)
servo_pins = [12, 13, 18, 19]
GPIO.setup(servo_pins, GPIO.OUT)

# ~ servo_link1 = GPIO12
# ~ servo_link2 = GPIO13
# ~ servo_vertical = GPI18
# ~ servo_gripper = GPIO19

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
    
# ============ START DUTY =============
servos[0].start(angle_to_duty(270))
time.sleep(0.3)
# ~ servos[1].start(angle_to_duty(0))
# ~ time.sleep(0.3)
servos[2].start(angle_to_duty(155))
time.sleep(0.3)
servos[3].start(angle_to_duty(0))
time.sleep(0.3)
    

# = = = = = = = SERVO ROTATE = = = = = = = = = 
def move_slow_link1(target, step = 1, delay=0.04):
    target = int(180 - target + 90)
    servo_index = 0
    
    if 'current_angle' not in move_slow_link1.__dict__:
        move_slow_link1.current_angle = [270]
        
    current = move_slow_link1.current_angle[0]
    if current < target:
        for angle in range(int(current), target + step + 4, step):
            servos[servo_index].ChangeDutyCycle(angle_to_duty(angle))
            time.sleep(delay)
    
    else:
        for angle in range(int(current), target - step - 4, -step):
            servos[servo_index].ChangeDutyCycle(angle_to_duty(angle))
            time.sleep(delay)
    
    move_slow_link1.current_angle[servo_index] = target



# ~ class move_slow_link2 :
    # ~ def __init__(self, kit=kit, channel=2, deg_per_sec=360/7.39, forward_servo_speed = 0.2, backward_servo_speed = -0.31):
        # ~ self.kit = kit
        # ~ self.channel = channel
        # ~ self.deg_per_sec = deg_per_sec
        # ~ self.pos = 0
        # ~ self.forward_speed = forward_servo_speed
        # ~ self.backward_speed = backward_servo_speed
    
    # ~ def move(self, target_deg):
        # ~ delta = int(target_deg) - self.pos
        # ~ duration = abs(delta) / self.deg_per_sec
        # ~ self.kit.continuous_servo[self.channel].throttle = self.backward_speed if delta < 0 else self.forward_speed
        # ~ time.sleep(duration)
        # ~ self.kit.continuous_servo[self.channel].throttle = 0
        # ~ # update pos
        # ~ self.pos = target_deg
  
  
    
def move_slow_slider(status, step = 1, delay=0.03):
    servo_index = 2
    
    if 'current_angle' not in move_slow_slider.__dict__:
        move_slow_slider.current_angle = [155]
        
    current = move_slow_slider.current_angle[0]
    
    if status == 'DOWN':
        for angle in range(int(current), 60 - step, -step):
            servos[servo_index].ChangeDutyCycle(angle_to_duty(angle))
            time.sleep(delay)
        move_slow_slider.current_angle[servo_index] = 60
        
    else:
        for angle in range(int(current), 155 + step, step):
            servos[servo_index].ChangeDutyCycle(angle_to_duty(angle))
            time.sleep(delay)
        move_slow_slider.current_angle[servo_index] = 155



def move_slow_grippper(status, step = 1, delay = 0.02):
    servo_index = 3
    
    if 'current_angle' not in move_slow_gripper.__dict__:
        move_slow_gripper.current_angle = [0]
        
    current = move_slow_gripper.current_angle[0]
    
    if status == 'ON':
        for angle in range(int(current), 70 + step, step):
            servos[servo_index].ChangeDutyCycle(angle_to_duty(angle))
            time.sleep(delay)
        move_slow_grippper,current_angle[servo_index] = 70
    else:
        for angle in range(int(current), 0 - step, -step):
            servos[servo_index].ChangeDutyCycle(angle_to_duty(angle))
            time.sleep(delay)
        move_slow_grippper,current_angle[servo_index] = 0



import time
from adafruit_servokit import ServoKit
from numpy import abs

kit = ServoKit(channels=8)


position = 0

def move_slow_link1(target, step=1, delay=0.02):
    channel = 0
    current = kit.servo[channel].angle

    if current is None:
        current = 0
        kit.servo[channel].angle = current
        time.sleep(.5)

    if current < target:
        for angle in range(int(current), target + 1, step):
            kit.servo[channel].angle = angle
            time.sleep(delay)
    else:
        for angle in range(int(current), target - 1, -step):
            kit.servo[channel].angle = angle
            time.sleep(delay)

def move_slow_link2(degrees, throttle, position): 
    #- - - const. - - -
    channel = 2
    forward_servo_speed = 0.2
    backward_servo_speed = -0.305
    rev_servo_speed = 360/7.39
    duration = abs(degrees-position) / rev_servo_speed
    
    #- - - - - - - - - -
    if position < degrees:
        kit.continuous_servo[channel].throttle = backward_servo_speed
        time.sleep(duration)
        kit.continuous_servo[channel].throttle = 0
        time.sleep(.5)
    else :
        kit.continuous_servo[channel].throttle = forward_servo_speed
        time.sleep(duration)
        kit.continuous_servo[channel].throttle = 0
        time.sleep(.5)
        
        
        
    kit.continuous_servo[channel].throttle = throttle
    time.sleep(duration)
    kit.continuous_servo[channel].throttle = 0
    
def move_slow_slider(target, step=1, delay=0.02):
    channel = 4
    current = kit.servo[channel].angle

    if current is None:
        current = 70
        kit.servo[channel].angle = current
        time.sleep(.5)

    if current < target:
        for angle in range(int(current), target + 1, step):
            kit.servo[channel].angle = angle
            time.sleep(delay)
    else:
        for angle in range(int(current), target - 1, -step):
            kit.servo[channel].angle = angle
            time.sleep(delay)

def calibration_servo() :
    kit.servo["link1"].angle = 0
    kit.servo["slider"].angle = 70


# for grippper maximum for grab
# move_slow(0, 0)

# for gripprt maximum for lifting gripper
# move_slow(0, 70)
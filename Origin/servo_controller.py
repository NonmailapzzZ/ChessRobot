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
        for angle in range(int(current), 180 - target + 1, step):
            kit.servo[channel].angle = angle
            time.sleep(delay)
    else:
        for angle in range(int(current), 180 - target - 1, -step):
            kit.servo[channel].angle = angle
            time.sleep(delay)

class move_slow_link2 :
    
    def __init__(self, kit, channel=2, deg_per_sec=360/7.39, forward_servo_speed = 0.2, backward_servo_speed = -0.305):
        self.kit = kit
        self.channel = channel
        self.deg_per_sec = deg_per_sec
        self.pos = 0
        self.forward_speed = forward_servo_speed
        self.backward_speed = backward_servo_speed
    
    def move(self, target_deg):
        delta = target_deg - self.pos
        duration = abs(delta) / self.deg_per_sec

        self.kit.continuous_servo[self.channel].throttle = self.backward_speed if delta > 0 else self.forward_speed
        time.sleep(duration)
        self.kit.continuous_servo[self.channel].throttle = 0
        
        # update pos
        self.pos = target_deg
    
def move_slow_slider(target, step=1, delay=0.02):
    channel = 4
    current = kit.servo[channel].angle

    if current is None:
        current = 70
        kit.servo[channel].angle = current
        time.sleep(.5)

    if current < target:
        for angle in range(int(current), 180 - target + 1, step):
            kit.servo[channel].angle = angle
            time.sleep(delay)
    else:
        for angle in range(int(current), 180 - target - 1, -step):
            kit.servo[channel].angle = angle
            time.sleep(delay)

def calibration_servo() :
    kit.servo["link1"].angle = 0
    kit.servo["slider"].angle = 70


# for grippper maximum for grab
# move_slow(0, 0)

# for gripprt maximum for lifting gripper
# move_slow(0, 70)
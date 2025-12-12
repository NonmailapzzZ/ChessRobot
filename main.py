import time
from adafruit_servokit import ServoKit

kit = ServoKit(channels=8)

def move_slow(channel, target, step=1, delay=0.02):
    current = kit.servo[channel].angle

    # ถ้ายังไม่มีค่า ให้ตั้งค่าเริ่มต้นเป็น 90
    if current is None:
        current = 90
        kit.servo[channel].angle = current

    if current < target:
        for angle in range(int(current), target + 1, step):
            kit.servo[channel].angle = angle
            time.sleep(delay)
    else:
        for angle in range(int(current), target - 1, -step):
            kit.servo[channel].angle = angle
            time.sleep(delay)



def rotate_degrees(channel, degrees, throttle): #
    #- - - const. - - -
    forward_servo_speed = 0.2
    backward_servo_speed = -0.305
    rev_servo_speed = 360/7.39
    duration = degrees / rev_servo_speed
    #- - - - - - - - - -
    
    kit.continuous_servo[channel].throttle = throttle
    time.sleep(duration)
    kit.continuous_servo[channel].throttle = 0
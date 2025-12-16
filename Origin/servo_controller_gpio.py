# = = = = = = = Const. = = = = = = = =
import time
from numpy import abs
import pigpio

# pigpio init
pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("pigpio daemon not running")

# ================= GPIO mapping =================
# แทน PCA9685 channel เดิม
SERVO_CH = 0          # link1
SERVO_PIN = 18        # GPIO18 (PWM)
LINK2_PIN = 19        # continuous servo
SLIDER_PIN = 20       # slider servo

# ================= Servo params =================
PWM_MIN_US = 500
PWM_MAX_US = 2500
ANGLE_MAX = 300

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

# = = = = = = = Internal helper (GPIO) = = = = = = =

def _angle_to_us(angle):
    angle = max(0, min(ANGLE_MAX, angle))
    return int(PWM_MIN_US + (angle / ANGLE_MAX) * (PWM_MAX_US - PWM_MIN_US))


def _us_to_angle(us):
    angle = (us - PWM_MIN_US) * ANGLE_MAX / (PWM_MAX_US - PWM_MIN_US)
    return max(0, min(ANGLE_MAX, angle))

# = = = = = = = Functions (ชื่อเดิม) = = = = = = =

def move_slow_link1(target, step=1, delay=0.04):
    target = int(180 - target + 89)

    current_us = pi.get_servo_pulsewidth(SERVO_PIN)
    if current_us == 0:
        current_us = _angle_to_us(0)

    current = _us_to_angle(current_us)

    if current < target:
        for angle in range(int(current), target + 5, step):
            pi.set_servo_pulsewidth(SERVO_PIN, _angle_to_us(angle) - 44)
            time.sleep(delay)
    else:
        for angle in range(int(current), target - 5, -step):
            pi.set_servo_pulsewidth(SERVO_PIN, _angle_to_us(angle) - 88)
            time.sleep(delay)


class move_slow_link2:
    def __init__(self, kit=None, channel=2, deg_per_sec=360/7.39,
                 forward_servo_speed=0.2, backward_servo_speed=-0.31):
        self.channel = channel
        self.deg_per_sec = deg_per_sec
        self.pos = 0

        # map throttle → pulsewidth
        self.forward_us = 1500 + int(forward_servo_speed * 400)
        self.backward_us = 1500 + int(backward_servo_speed * 400)

    def move(self, target_deg):
        delta = int(target_deg) - self.pos
        duration = abs(delta) / self.deg_per_sec

        if delta < 0:
            pi.set_servo_pulsewidth(LINK2_PIN, self.backward_us)
        else:
            pi.set_servo_pulsewidth(LINK2_PIN, self.forward_us)

        time.sleep(duration)
        pi.set_servo_pulsewidth(LINK2_PIN, 1500)
        self.pos = target_deg


def move_slow_slider(target, step=1, delay=0.04):
    target = int(target)

    current_us = pi.get_servo_pulsewidth(SLIDER_PIN)
    if current_us == 0:
        current_us = _angle_to_us(0)

    current = _us_to_angle(current_us)

    if current < target:
        for angle in range(int(current), target + 1, step):
            pi.set_servo_pulsewidth(SLIDER_PIN, _angle_to_us(angle))
            time.sleep(delay)
    else:
        for angle in range(int(current), target - 1, -step):
            pi.set_servo_pulsewidth(SLIDER_PIN, _angle_to_us(angle))
            time.sleep(delay)


def calibration_servo():
    pi.set_servo_pulsewidth(SERVO_PIN, _angle_to_us(180))
    pi.set_servo_pulsewidth(SLIDER_PIN, _angle_to_us(70))


# = = = = = = = Cleanup = = = = = = = =

def cleanup():
    pi.set_servo_pulsewidth(SERVO_PIN, 0)
    pi.set_servo_pulsewidth(LINK2_PIN, 0)
    pi.set_servo_pulsewidth(SLIDER_PIN, 0)
    pi.stop()

# = = = = = = = Const. = = = = = = = =
# GPIOZero ONLY version (ไม่ใช้ pigpio / I2C / PCA9685)
# คงชื่อฟังก์ชันและ class เดิมทั้งหมด

import time
from numpy import abs
from gpiozero import Servo, AngularServo

# ================= GPIO mapping =================
# map channel เดิม → GPIO pin
SERVO_CH = 0          # คงไว้เพื่อ compatibility
LINK1_PIN = 18        # servo 300°
LINK2_PIN = 19        # continuous servo
SLIDER_PIN = 20       # servo 180°

# ================= Servo params =================
PWM_MIN_US = 500
PWM_MAX_US = 2500
ANGLE_MAX = 300

# ================= Servo objects =================
# link1 : 300° servo
link1 = AngularServo(
    LINK1_PIN,
    min_angle=0,
    max_angle=ANGLE_MAX,
    min_pulse_width=PWM_MIN_US / 1_000_000,
    max_pulse_width=PWM_MAX_US / 1_000_000,
)

# slider : 180° servo
slider = AngularServo(
    SLIDER_PIN,
    min_angle=0,
    max_angle=180,
    min_pulse_width=0.5 / 1000,
    max_pulse_width=2.5 / 1000,
)

# continuous servo
link2 = Servo(
    LINK2_PIN,
    min_pulse_width=1.3 / 1000,
    max_pulse_width=1.7 / 1000,
)

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

# = = = = = = = Functions (ชื่อเดิม) = = = = = = =

def move_slow_link1(target, step=1, delay=0.04):
    target = int(target)

    current = link1.angle if link1.angle is None else 0

    if current < target:
        for angle in range(int(current), target + 5, step):
            link1.angle = angle
            time.sleep(delay)
    else:
        for angle in range(int(current), target - 5, -step):
            link1.angle = angle
            time.sleep(delay)


class move_slow_link2:
    def __init__(self, deg_per_sec=360/7.39,
                 forward_servo_speed=0.2, backward_servo_speed=-0.31):
        self.deg_per_sec = deg_per_sec
        self.pos = 0
        self.forward_speed = forward_servo_speed
        self.backward_speed = backward_servo_speed

    def move(self, target_deg):
        delta = int(target_deg) - self.pos
        duration = abs(delta) / self.deg_per_sec

        link2.value = self.backward_speed if delta < 0 else self.forward_speed
        time.sleep(duration)
        link2.value = 0
        self.pos = target_deg


def move_slow_slider(target, step=1, delay=0.04):
    target = int(target)
    current = slider.angle or 0

    if current < target:
        for angle in range(int(current), target + 1, step):
            slider.angle = angle
            time.sleep(delay)
    else:
        for angle in range(int(current), target - 1, -step):
            slider.angle = angle
            time.sleep(delay)


def calibration_servo():
    link1.angle = 180
    slider.angle = 70


# = = = = = = = Cleanup = = = = = = = =

def cleanup():
    link1.detach()
    slider.detach()
    link2.stop()

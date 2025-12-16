# = = = = = = = Const. = = = = = = = =
import time
from numpy import abs
import lgpio

# ================= lgpio init =================
# เปิด gpiochip0
chip = lgpio.gpiochip_open(0)

# ================= GPIO mapping =================
# แทน PCA9685 channel เดิม
SERVO_CH = 0          # link1 (คงไว้เพื่อ compatibility)
SERVO_PIN = 18        # GPIO18 (PWM)
LINK2_PIN = 19        # continuous servo
SLIDER_PIN = 20       # slider servo

# ================= Servo params =================
PWM_MIN_US = 500
PWM_MAX_US = 2500
ANGLE_MAX = 300
PWM_FREQ = 50         # 50Hz servo
PERIOD_US = 20000     # 20ms

# ขอสิทธิ์ควบคุม GPIO
for pin in (SERVO_PIN, LINK2_PIN, SLIDER_PIN):
    lgpio.gpio_claim_output(chip, pin)

# = = = = = = = Calculate. = = = = = = = =

def duty_to_us(duty):
    return duty * PERIOD_US / 65535

def us_to_duty(us):
    return int(us * 65535 / PERIOD_US)

def angle_to_duty(angle):
    angle = max(0, min(ANGLE_MAX, angle))
    us = PWM_MIN_US + (angle / ANGLE_MAX) * (PWM_MAX_US - PWM_MIN_US)
    return us_to_duty(us)

def duty_to_angle(duty):
    us = duty_to_us(duty)
    angle = (us - PWM_MIN_US) * ANGLE_MAX / (PWM_MAX_US - PWM_MIN_US)
    return max(0, min(ANGLE_MAX, angle))

# = = = = = = = Internal helper (lgpio PWM) = = = = = = =

def _angle_to_us(angle):
    angle = max(0, min(ANGLE_MAX, angle))
    return int(PWM_MIN_US + (angle / ANGLE_MAX) * (PWM_MAX_US - PWM_MIN_US))


def _us_to_angle(us):
    angle = (us - PWM_MIN_US) * ANGLE_MAX / (PWM_MAX_US - PWM_MIN_US)
    return max(0, min(ANGLE_MAX, angle))


def _set_servo_us(pin, us):
    """ตั้ง PWM ด้วย lgpio (us → duty%)"""
    duty_percent = (us / PERIOD_US) * 100.0
    lgpio.tx_pwm(chip, pin, PWM_FREQ, duty_percent)

# = = = = = = = Functions (ชื่อเดิม) = = = = = = =

def move_slow_link1(target, step=1, delay=0.04):
    target = int(180 - target + 89)

    # ไม่มี readback PWM ใน lgpio → สมมติเริ่มที่ 0
    current = 0

    if current < target:
        for angle in range(int(current), target + 5, step):
            _set_servo_us(SERVO_PIN, _angle_to_us(angle) - 44)
            time.sleep(delay)
    else:
        for angle in range(int(current), target - 5, -step):
            _set_servo_us(SERVO_PIN, _angle_to_us(angle) - 88)
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
            _set_servo_us(LINK2_PIN, self.backward_us)
        else:
            _set_servo_us(LINK2_PIN, self.forward_us)

        time.sleep(duration)
        _set_servo_us(LINK2_PIN, 1500)  # stop
        self.pos = target_deg


def move_slow_slider(target, step=1, delay=0.04):
    target = int(target)
    current = 0

    if current < target:
        for angle in range(int(current), target + 1, step):
            _set_servo_us(SLIDER_PIN, _angle_to_us(angle))
            time.sleep(delay)
    else:
        for angle in range(int(current), target - 1, -step):
            _set_servo_us(SLIDER_PIN, _angle_to_us(angle))
            time.sleep(delay)


def calibration_servo():
    _set_servo_us(SERVO_PIN, _angle_to_us(180))
    _set_servo_us(SLIDER_PIN, _angle_to_us(70))


# = = = = = = = Cleanup = = = = = = = =

def cleanup():
    for pin in (SERVO_PIN, LINK2_PIN, SLIDER_PIN):
        lgpio.tx_pwm(chip, pin, 0, 0)
    lgpio.gpiochip_close(chip)

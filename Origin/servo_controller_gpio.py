import gpiod
from gpiod import line
import time

global period
period = 0.02 # 50 Hz

# ================= GPIO SETUP =================
chip = gpiod.Chip("/dev/gpiochip4")

servo_link1 = 12
servo_link2 = 13
servo_vertical = 18
servo_gripper = 19

lines = chip.request_lines(
    consumer="servo_control",
    config={
        12: gpiod.LineSettings(direction=line.Direction.OUTPUT, output_value=line.Value.INACTIVE),
        13: gpiod.LineSettings(direction=line.Direction.OUTPUT, output_value=line.Value.INACTIVE),
        18: gpiod.LineSettings(direction=line.Direction.OUTPUT, output_value=line.Value.INACTIVE),
        19: gpiod.LineSettings(direction=line.Direction.OUTPUT, output_value=line.Value.INACTIVE)
    }
)

# ================= SERVO CONFIG =================
MIN_ANGLE = 0.0
MAX_ANGLE = 300.0

def angle_to_pulse_ms(angle):
    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
    return 0.5 + (angle / 300.0) * 2.0   # 0.5 to 2.5 ms


# ================= CALIBREATION =================
def calibration_servo() :
    # SERVO_LINK1
    pulse_ms = angle_to_pulse_ms(274)
    lines.set_value(12, line.Value.ACTIVE)
    time.sleep(pulse_ms / 1000)
    lines.set_value(12, line.Value.INACTIVE)
    time.sleep(period - pulse_ms / 1000)


# ================= PWM THREAD =================
current_link1 = 274
def move_slow_link1(target, step=0.5):
    target = int(180 - target + 85)
    
    pulse_ms = angle_to_pulse_ms(target)
    lines.set_value(12, line.Value.ACTIVE)
    time.sleep(pulse_ms / 1000)
    lines.set_value(12, line.Value.INACTIVE)
    time.sleep(period - pulse_ms / 1000)
        
    current = target
    
    

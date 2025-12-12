# main.py
# Motor driver API for SCARA (simple wrapper around Adafruit ServoKit)
# Provides init(), move_theta1(), move_theta2(), move_d4(), set_joints(), cleanup()
# Falls back to simulation logging if hardware libs are missing.

import time

HW_AVAILABLE = True
try:
    from adafruit_servokit import ServoKit
except Exception as e:
    # hardware library not available — fall back to simulation mode
    ServoKit = None
    HW_AVAILABLE = False

# Configuration: which channels map to which joint (adjust to your wiring)
CHANNEL_TH1 = 0   # servo channel for theta1 (servo angle)
CHANNEL_TH2 = 1   # servo or continuous for theta2
CHANNEL_D4  = 2   # servo for vertical (or placeholder)

kit = None
_sim_state = {
    "theta1": 0.0,
    "theta2": 0.0,
    "d4": 0.0,
    "inited": False
}

def init(channels=8):
    """Initialize hardware (call once). Returns True if hw available and initialized."""
    global kit, _sim_state
    if HW_AVAILABLE and ServoKit is not None:
        try:
            kit = ServoKit(channels=channels)
            # optional: safe initial values
            try:
                # set initial safe positions; use try/except in case some channels not used
                if hasattr(kit, 'servo'):
                    try:
                        kit.servo[CHANNEL_TH1].angle = 90
                    except Exception:
                        pass
                    try:
                        kit.servo[CHANNEL_D4].angle = 90
                    except Exception:
                        pass
                if hasattr(kit, 'continuous_servo'):
                    try:
                        kit.continuous_servo[CHANNEL_TH2].throttle = 0
                    except Exception:
                        pass
            except Exception:
                pass
            _sim_state["inited"] = True
            print("[main] Hardware ServoKit initialized.")
            return True
        except Exception as e:
            print(f"[main] Failed to init ServoKit: {e}")
            kit = None
            _sim_state["inited"] = True
            return False
    else:
        # simulation mode
        _sim_state["inited"] = True
        print("[main] ServoKit not available — running in SIMULATION mode.")
        return False

def _clamp_angle(a):
    # clamp angle to 0..180 for standard servos
    try:
        v = float(a)
    except Exception:
        v = 0.0
    if v < 0.0: return 0.0
    if v > 180.0: return 180.0
    return v

def move_theta1(angle_deg):
    """Move joint 1 (theta1). angle_deg is in degrees (robot frame)."""
    # mapping: robot angle -> servo angle
    # default mapping: servo angle = angle_deg + 90  (so robot 0 -> servo 90)
    servo_angle = _clamp_angle(angle_deg + 90.0)
    if kit and hasattr(kit, 'servo') and kit.servo is not None:
        try:
            kit.servo[CHANNEL_TH1].angle = servo_angle
            print(f"[main] HW: move_theta1 -> servo ch{CHANNEL_TH1} angle {servo_angle}")
        except Exception as e:
            print(f"[main] HW error move_theta1: {e}")
    else:
        # simulation
        _sim_state['theta1'] = angle_deg
        print(f"[main] SIM: move_theta1 -> {angle_deg}° (mapped servo {servo_angle})")

def move_theta2(angle_deg):
    """Move joint 2 (theta2). Accepts degrees. If channel is continuous servo, sets throttle."""
    # we'll try to set servo angle first; if continuous, set throttle
    servo_angle = _clamp_angle(angle_deg + 90.0)
    if kit:
        # prefer normal servo if available
        try:
            if hasattr(kit, 'servo') and kit.servo is not None:
                kit.servo[CHANNEL_TH2].angle = servo_angle
                print(f"[main] HW: move_theta2 -> servo ch{CHANNEL_TH2} angle {servo_angle}")
                return
        except Exception:
            pass
        try:
            if hasattr(kit, 'continuous_servo') and kit.continuous_servo is not None:
                # map angle to throttle -1..1 (this mapping is arbitrary; adjust to your hardware)
                throttle = max(-1.0, min(1.0, (angle_deg / 180.0)))
                kit.continuous_servo[CHANNEL_TH2].throttle = throttle
                print(f"[main] HW: move_theta2 -> continuous ch{CHANNEL_TH2} throttle {throttle}")
                return
        except Exception as e:
            print(f"[main] HW error move_theta2: {e}")
    # fallback simulation
    _sim_state['theta2'] = angle_deg
    print(f"[main] SIM: move_theta2 -> {angle_deg}° (mapped servo {servo_angle})")

def move_d4(mm):
    """Move prismatic joint d4 (mm). Default: map mm to servo angle or ignore if not used."""
    # map mm range 0..200 -> servo angle 0..180 as example
    try:
        v = float(mm)
    except Exception:
        v = 0.0
    servo_angle = (v / 200.0) * 180.0
    servo_angle = _clamp_angle(servo_angle)
    if kit and hasattr(kit, 'servo') and kit.servo is not None:
        try:
            kit.servo[CHANNEL_D4].angle = servo_angle
            print(f"[main] HW: move_d4 -> servo ch{CHANNEL_D4} angle {servo_angle}")
            return
        except Exception as e:
            print(f"[main] HW error move_d4: {e}")
    _sim_state['d4'] = v
    print(f"[main] SIM: move_d4 -> {v} mm (mapped servo {servo_angle})")

def set_joints(theta1, theta2, d4):
    """Convenience: set all three joints (non-blocking)."""
    # You may want to add safety checks/limits here
    print(f"[main] set_joints: theta1={theta1}, theta2={theta2}, d4={d4}")
    move_theta1(theta1)
    move_theta2(theta2)
    move_d4(d4)

def cleanup():
    """Cleanup hardware resources (call before exit)."""
    try:
        if kit:
            # if continuous servo present, stop throttle
            try:
                if hasattr(kit, 'continuous_servo'):
                    kit.continuous_servo[CHANNEL_TH2].throttle = 0
            except Exception:
                pass
        print("[main] cleanup done.")
    except Exception as e:
        print(f"[main] cleanup error: {e}")

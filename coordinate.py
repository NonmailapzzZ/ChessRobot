"""
coordinate.py

Utility module for SCARA kinematics and simple coordinate transforms.
Designed to be imported and called from the UI (scara_camera_final.py) or a controller.

Provides:
- fk_scara(theta1_deg, theta2_deg, d4_mm) -> (x_mm, y_mm, z_mm, yaw_deg)
- ik_scara(x_mm, y_mm, z_mm) -> list of solutions [{"theta1":deg,...}, ...] or [] if none
- clamp_joint_limits(joints, limits) -> dict
- units: inputs/outputs use degrees for angles and mm for distances (matches UI)

Default link lengths L1,L2 set to 200,150 to match the UI file's constants.

Contains a small test block when run as __main__ to sanity-check forward/inverse round-trip.
"""

import math
from typing import List, Dict, Optional, Tuple

# Default robot geometry (match scara_ui defaults)
L1 = 200.0  # mm
L2 = 150.0  # mm

# Joint limits (degrees for angles, mm for d4) - same convention as UI
DEFAULT_JOINT_LIMITS = {
    'theta1': (-180.0, 180.0),
    'theta2': (-180.0, 180.0),
    'd4': (0.0, 200.0),
}

# -------------------- Utility helpers --------------------

def _deg(rad: float) -> float:
    return math.degrees(rad)


def _rad(deg: float) -> float:
    return math.radians(deg)


# -------------------- Forward Kinematics --------------------

def fk_scara(theta1_deg: float, theta2_deg: float, d4_mm: float,
             l1: float = L1, l2: float = L2) -> Tuple[float, float, float, float]:
    """
    Forward kinematics for a planar 2-link SCARA + vertical slide.

    Inputs:
        theta1_deg, theta2_deg : joint angles in degrees
        d4_mm                : vertical slide displacement in mm (positive downwards)
        l1, l2               : link lengths in mm

    Returns:
        x_mm, y_mm, z_mm, yaw_deg
        - z_mm is negative for positive d4 (same sign convention used in UI)
        - yaw_deg = theta1 + theta2 normalized to [-180,180]
    """
    t1 = _rad(theta1_deg)
    t2 = _rad(theta2_deg)

    x = l1 * math.cos(t1) + l2 * math.cos(t1 + t2)
    y = l1 * math.sin(t1) + l2 * math.sin(t1 + t2)
    z = -d4_mm

    yaw = (theta1_deg + theta2_deg) % 360.0
    if yaw > 180.0:
        yaw -= 360.0

    return x, y, z, yaw


# -------------------- Inverse Kinematics --------------------

def ik_scara(x_mm: float, y_mm: float, z_mm: float,
             l1: float = L1, l2: float = L2,
             allow_elbow: bool = True) -> List[Dict[str, float]]:
    """
    Inverse kinematics for planar 2-link SCARA (theta1, theta2) + vertical d4.

    Inputs:
        x_mm, y_mm : desired end-effector planar position in mm
        z_mm       : desired z (matches fk output) — note fk returns z = -d4
        l1, l2     : link lengths in mm
        allow_elbow: if True, return both elbow-up and elbow-down solutions when available

    Returns:
        List of solution dicts like { 'theta1': deg, 'theta2': deg, 'd4': mm }
        Returns empty list if point is out of reach.

    Notes:
        - Uses standard closed-form solution for 2-link planar manipulator.
        - Angle convention: results in degrees.
        - d4 is computed as -z (inverse of fk's sign convention).
    """
    solutions: List[Dict[str, float]] = []

    # planar distance
    r2 = x_mm * x_mm + y_mm * y_mm
    r = math.sqrt(r2)

    # check reachability
    denom = 2.0 * l1 * l2
    if denom == 0:
        return []

    cos_theta2 = (r2 - l1 * l1 - l2 * l2) / denom

    # numerical tolerance
    if cos_theta2 > 1.0 + 1e-9 or cos_theta2 < -1.0 - 1e-9:
        return []

    # clamp
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))

    # two possible theta2 solutions (elbow up/down)
    theta2_a = math.acos(cos_theta2)
    theta2_b = -theta2_a

    def _compute_theta1(theta2_rad: float) -> float:
        k1 = l1 + l2 * math.cos(theta2_rad)
        k2 = l2 * math.sin(theta2_rad)
        phi = math.atan2(y_mm, x_mm)
        psi = math.atan2(k2, k1)
        theta1_rad = phi - psi
        return theta1_rad

    # solution A
    th1a = _compute_theta1(theta2_a)
    th2a = theta2_a
    solutions.append({'theta1': _deg(th1a), 'theta2': _deg(th2a), 'd4': -z_mm})

    # solution B (if allowed and distinct)
    if allow_elbow:
        th1b = _compute_theta1(theta2_b)
        th2b = theta2_b
        # if solutions essentially different, add
        if abs(_deg(th1b) - _deg(th1a)) > 1e-6 or abs(_deg(th2b) - _deg(th2a)) > 1e-6:
            solutions.append({'theta1': _deg(th1b), 'theta2': _deg(th2b), 'd4': -z_mm})

    # normalize angles to [-180,180]
    for s in solutions:
        s['theta1'] = ((s['theta1'] + 180.0) % 360.0) - 180.0
        s['theta2'] = ((s['theta2'] + 180.0) % 360.0) - 180.0

    return solutions


# -------------------- Helpers for integration --------------------

def clamp_joint_limits(joints: Dict[str, float],
                       limits: Dict[str, Tuple[float, float]] = DEFAULT_JOINT_LIMITS) -> Dict[str, float]:
    """
    Clamp a joints dict to provided limits. Returns a new dict.
    Example input joints: { 'theta1': 10.0, 'theta2': -5.0, 'd4': 20.0 }
    """
    out = {}
    for k, v in joints.items():
        if k in limits:
            lo, hi = limits[k]
            out[k] = max(lo, min(hi, float(v)))
        else:
            out[k] = float(v)
    return out


def format_solution(sol: Dict[str, float]) -> str:
    return f"theta1={sol['theta1']:.2f}°, theta2={sol['theta2']:.2f}°, d4={sol['d4']:.2f} mm"


# -------------------- Simple tests / example usage --------------------
if __name__ == '__main__':
    print('coordinate.py quick test (forward <-> inverse roundtrip)')

    test_cases = [
        {'theta1': 0.0, 'theta2': 0.0, 'd4': 0.0},
        {'theta1': 30.0, 'theta2': -45.0, 'd4': 50.0},
        {'theta1': -90.0, 'theta2': 45.0, 'd4': 20.0},
    ]

    for t in test_cases:
        x, y, z, yaw = fk_scara(t['theta1'], t['theta2'], t['d4'])
        print(f"FK -> theta1={t['theta1']}, theta2={t['theta2']}, d4={t['d4']} -> x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}")
        sols = ik_scara(x, y, z)
        if not sols:
            print('  IK: no solution (unexpected)')
        else:
            for i, s in enumerate(sols):
                print(f"  IK sol {i}: {format_solution(s)}")

    # Reachability test: out of reach
    xr = L1 + L2 + 10
    sol = ik_scara(xr, 0.0, 0.0)
    print(f"Out-of-reach test at x={xr}: solutions found = {len(sol)}")

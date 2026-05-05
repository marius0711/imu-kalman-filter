"""
test_complementary_filter.py — Tests for the complementary filter.

Uses synthetic IMU data: known constant rotation → verify angle tracking.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from complementary_filter import run, accel_to_angles


def make_static_imu(duration: float = 10.0, rate: float = 200.0,
                    roll_deg: float = 0.0, pitch_deg: float = 0.0) -> dict:
    """
    Static IMU: no motion, gravity rotated to given roll/pitch.
    Gyro = zero + tiny noise. Accel = rotated gravity vector + tiny noise.
    """
    N = int(duration * rate)
    dt = 1.0 / rate
    t = np.arange(N) * dt
    rng = np.random.default_rng(0)

    roll_r  = np.radians(roll_deg)
    pitch_r = np.radians(pitch_deg)

    # Gravity in body frame for given roll/pitch
    g = 9.81
    ax = -g * np.sin(pitch_r)
    ay =  g * np.sin(roll_r) * np.cos(pitch_r)
    az =  g * np.cos(roll_r) * np.cos(pitch_r)

    accel = np.tile([ax, ay, az], (N, 1)) + rng.normal(0, 0.01, (N, 3))
    gyro  = rng.normal(0, 1e-4, (N, 3))

    return {"t": t, "gyro": gyro, "accel": accel}


def make_rotating_imu(omega_x: float = 0.1, duration: float = 10.0,
                      rate: float = 200.0) -> dict:
    """
    Constant rotation around X axis (pure roll).
    Gyro = [omega_x, 0, 0]. Accel = gravity rotated by current angle.
    """
    N = int(duration * rate)
    dt = 1.0 / rate
    t = np.arange(N) * dt
    rng = np.random.default_rng(1)

    g = 9.81
    roll = omega_x * t  # true roll angle [rad]

    gyro  = np.column_stack([
        np.full(N, omega_x) + rng.normal(0, 1e-4, N),
        rng.normal(0, 1e-4, N),
        rng.normal(0, 1e-4, N),
    ])
    accel = np.column_stack([
        rng.normal(0, 0.01, N),
        g * np.sin(roll) + rng.normal(0, 0.01, N),
        g * np.cos(roll) + rng.normal(0, 0.01, N),
    ])
    return {"t": t, "gyro": gyro, "accel": accel, "roll_true_deg": np.degrees(roll)}


# ---------------------------------------------------------------------------

def test_output_keys():
    imu = make_static_imu()
    result = run(imu)
    expected = {"t", "roll_cf", "pitch_cf", "roll_accel", "pitch_accel",
                "roll_gyro", "pitch_gyro"}
    assert set(result.keys()) == expected


def test_output_shapes():
    imu = make_static_imu(duration=5.0)
    N = len(imu["t"])
    result = run(imu)
    for key in result:
        assert result[key].shape == (N,), f"Wrong shape for {key}"


def test_static_roll_zero():
    """Static sensor flat on table: roll and pitch should be ~0°."""
    imu = make_static_imu(roll_deg=0.0, pitch_deg=0.0)
    result = run(imu, alpha=0.98)
    # After settling, mean should be close to 0
    assert abs(np.mean(result["roll_cf"][-100:])) < 1.0
    assert abs(np.mean(result["pitch_cf"][-100:])) < 1.0


def test_static_known_tilt():
    """Static sensor tilted 20° roll: CF should converge to ~20°."""
    imu = make_static_imu(roll_deg=20.0, pitch_deg=0.0, duration=15.0)
    result = run(imu, alpha=0.98)
    assert abs(np.mean(result["roll_cf"][-100:]) - 20.0) < 2.0


def test_cf_less_drift_than_gyro():
    """Complementary filter corrects gyro drift caused by constant bias."""
    # Inject a constant gyro bias: sensor at rest but gyro reads 0.05 rad/s
    imu = make_static_imu(duration=30.0)
    imu["gyro"][:, 0] += 0.05  # constant roll-rate bias

    result = run(imu, alpha=0.98)
    # Gyro-only: integrates bias → large drift after 30s
    gyro_end_drift = abs(result["roll_gyro"][-1] - result["roll_gyro"][0])
    # CF: accel reference corrects bias → stays near 0°
    cf_end_drift = abs(result["roll_cf"][-1] - result["roll_cf"][0])
    assert cf_end_drift < gyro_end_drift


def test_alpha_effect():
    """Lower alpha → faster accel correction → less steady-state error on static."""
    imu = make_static_imu(roll_deg=15.0, duration=20.0)
    r_high = run(imu, alpha=0.99)
    r_low  = run(imu, alpha=0.80)
    # Lower alpha converges faster; error after 20s should be smaller
    err_high = abs(np.mean(r_high["roll_cf"][-50:]) - 15.0)
    err_low  = abs(np.mean(r_low["roll_cf"][-50:])  - 15.0)
    assert err_low <= err_high + 0.5  # allow small tolerance


def test_accel_to_angles_gravity_aligned():
    """Pure gravity on Z: roll and pitch should be ~0°."""
    accel = np.array([[0.0, 0.0, 9.81]])
    angles = accel_to_angles(accel)
    assert abs(angles[0, 0]) < 0.1  # roll
    assert abs(angles[0, 1]) < 0.1  # pitch


def test_accel_to_angles_pure_roll():
    """Gravity on Y axis → ~90° roll."""
    accel = np.array([[0.0, 9.81, 0.0]])
    angles = accel_to_angles(accel)
    assert abs(angles[0, 0] - 90.0) < 1.0

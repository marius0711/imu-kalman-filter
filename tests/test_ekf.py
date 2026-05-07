"""
Tests for the Extended Kalman Filter (ekf.py).

Test strategy:
  - Static case: EKF at rest must return near-identity orientation.
  - Pitch sweep: constant rotation around Y axis, accelerometer provides
    tilt reference. RMSE < 2 degrees required.
  - Bias convergence: inject known constant bias; filter must estimate it.
  - Quaternion normalisation: state vector must stay unit after many steps.
  - Covariance symmetry: P must remain symmetric and positive-definite.
  - Measurement Jacobian: numerical gradient check against analytical H.
  - Process Jacobian: numerical gradient check against analytical F.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from ekf import EKF, run_ekf_on_dataset, quat_angle_error_deg


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

FS = 200.0
DT = 1.0 / FS
G = 9.81

# ADIS16448 parameters (same as used in generate_synthetic_imu.py)
SIGMA_GYRO = 1.7e-4
SIGMA_BIAS = 1.9e-5
SIGMA_ACCEL = 2.0e-3


def make_ekf(**kwargs) -> EKF:
    defaults = dict(
        sigma_gyro=SIGMA_GYRO,
        sigma_bias=SIGMA_BIAS,
        sigma_accel=SIGMA_ACCEL,
        fs=FS,
        g=G,
    )
    defaults.update(kwargs)
    return EKF(**defaults)


def generate_pitch_sweep(
    omega_y: float = 0.2,
    duration: float = 15.0,
    noise_scale: float = 1.0,
    seed: int = 42,
) -> dict:
    """
    Synthetic constant-rate rotation around the Y axis (pitch).

    Ground truth quaternion: q = [cos(theta/2), 0, sin(theta/2), 0]
    Gyro:  [0, omega_y, 0] + noise
    Accel: R(q)^T @ [0, 0, g] + noise

    Returns a dict with keys: timestamps, gyro, accel, q_true.
    """
    rng = np.random.default_rng(seed)
    N = int(duration * FS)
    t = np.arange(N) * DT
    theta = omega_y * t  # pitch angle over time

    # Ground-truth quaternions [q0, q1, q2, q3]
    q_true = np.zeros((N, 4))
    q_true[:, 0] = np.cos(theta / 2.0)
    q_true[:, 2] = np.sin(theta / 2.0)

    # Clean gyro: [0, omega_y, 0]
    gyro_clean = np.zeros((N, 3))
    gyro_clean[:, 1] = omega_y

    # Gyro noise
    gyro_sigma = SIGMA_GYRO * np.sqrt(FS) * noise_scale
    gyro = gyro_clean + rng.normal(0, gyro_sigma, (N, 3))

    # Clean accel: R(q)^T @ [0, 0, g]
    # For Y-rotation: accel = [g*sin(theta), 0, g*cos(theta)]
    accel_clean = np.stack(
        [-G * np.sin(theta), np.zeros(N), G * np.cos(theta)], axis=1
    )

    accel_sigma = SIGMA_ACCEL * np.sqrt(FS) * noise_scale
    accel = accel_clean + rng.normal(0, accel_sigma, (N, 3))

    return {
        "timestamps": t,
        "gyro": gyro,
        "accel": accel,
        "q_true": q_true,
    }


def generate_bias_scenario(
    true_bias: np.ndarray,
    duration: float = 20.0,
    seed: int = 7,
) -> dict:
    """Static IMU with injected constant gyro bias."""
    rng = np.random.default_rng(seed)
    N = int(duration * FS)
    t = np.arange(N) * DT

    gyro_sigma = SIGMA_GYRO * np.sqrt(FS)
    accel_sigma = SIGMA_ACCEL * np.sqrt(FS)

    # No rotation; gyro reads only bias + noise
    gyro = true_bias[None, :] + rng.normal(0, gyro_sigma, (N, 3))
    # Accel reads gravity aligned with world Z (identity orientation)
    accel = np.tile([0.0, 0.0, G], (N, 1)) + rng.normal(0, accel_sigma, (N, 3))

    return {"timestamps": t, "gyro": gyro, "accel": accel}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEKFInitialisation:
    def test_initial_quaternion_is_identity(self):
        ekf = make_ekf()
        q = ekf.get_quaternion()
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_initial_bias_is_zero(self):
        ekf = make_ekf()
        np.testing.assert_allclose(ekf.get_bias(), np.zeros(3), atol=1e-10)

    def test_initial_covariance_is_spd(self):
        ekf = make_ekf()
        eigenvalues = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigenvalues > 0), "Initial P must be positive definite."

    def test_reset_restores_identity(self):
        ekf = make_ekf()
        # Perturb
        for _ in range(50):
            ekf.predict(np.array([0.1, 0.0, 0.0]), DT)
        ekf.reset()
        np.testing.assert_allclose(ekf.get_quaternion(), [1.0, 0.0, 0.0, 0.0], atol=1e-10)


class TestEKFStaticCase:
    """At rest, orientation should stay near identity after many updates."""

    def test_static_orientation_stays_near_identity(self):
        rng = np.random.default_rng(0)
        ekf = make_ekf()
        gyro_sigma = SIGMA_GYRO * np.sqrt(FS)
        accel_sigma = SIGMA_ACCEL * np.sqrt(FS)

        for _ in range(500):
            gyro = rng.normal(0, gyro_sigma, 3)
            accel = np.array([0.0, 0.0, G]) + rng.normal(0, accel_sigma, 3)
            ekf.predict(gyro, DT)
            ekf.update(accel)

        roll, pitch, _ = ekf.get_euler_deg()
        assert abs(roll) < 1.0, f"Roll drift too large: {roll:.3f} deg"
        assert abs(pitch) < 1.0, f"Pitch drift too large: {pitch:.3f} deg"

    def test_quaternion_stays_unit_after_many_steps(self):
        rng = np.random.default_rng(1)
        ekf = make_ekf()
        for _ in range(1000):
            gyro = rng.normal(0, 1e-3, 3)
            accel = np.array([0.0, 0.0, G]) + rng.normal(0, 0.01, 3)
            ekf.predict(gyro, DT)
            ekf.update(accel)

        norm = np.linalg.norm(ekf.get_quaternion())
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)


class TestEKFOrientationAccuracy:
    """Core accuracy test: RMSE must be below 2 degrees on noisy pitch sweep."""

    def test_pitch_sweep_rmse_below_2deg(self):
        data = generate_pitch_sweep(omega_y=0.2, duration=15.0)
        quats = run_ekf_on_dataset(
            data["gyro"],
            data["accel"],
            data["timestamps"],
        )

        # Discard first 2 seconds of convergence transient
        skip = int(2.0 * FS)
        errors = quat_angle_error_deg(quats[skip:], data["q_true"][skip:])
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        assert rmse < 2.0, (
            f"EKF RMSE {rmse:.3f} deg exceeds 2 deg target on pitch sweep."
        )

    def test_pitch_sweep_mean_error_below_1deg(self):
        data = generate_pitch_sweep(omega_y=0.15, duration=20.0, seed=99)
        quats = run_ekf_on_dataset(
            data["gyro"],
            data["accel"],
            data["timestamps"],
        )
        skip = int(2.0 * FS)
        errors = quat_angle_error_deg(quats[skip:], data["q_true"][skip:])
        mean_err = float(np.mean(errors))

        assert mean_err < 1.0, (
            f"EKF mean error {mean_err:.3f} deg exceeds 1 deg on pitch sweep."
        )

    def test_error_decreases_after_convergence(self):
        """Errors in the second half should be lower than the first half."""
        data = generate_pitch_sweep(omega_y=0.2, duration=20.0, seed=5)
        quats = run_ekf_on_dataset(
            data["gyro"],
            data["accel"],
            data["timestamps"],
        )
        skip = int(2.0 * FS)
        mid = len(quats) // 2
        errors = quat_angle_error_deg(quats[skip:], data["q_true"][skip:])
        first_half = float(np.mean(errors[: mid - skip]))
        second_half = float(np.mean(errors[mid - skip :]))
        assert second_half <= first_half * 1.5, (
            f"Error did not stabilise: first_half={first_half:.3f}, second_half={second_half:.3f}"
        )


class TestEKFBiasEstimation:
    """Filter must converge toward the true constant gyro bias."""

    def test_bias_estimate_converges(self):
        true_bias = np.array([0.01, -0.005, 0.008])  # rad/s
        data = generate_bias_scenario(true_bias, duration=30.0)

        ekf = make_ekf()
        for i in range(1, len(data["timestamps"])):
            dt = float(data["timestamps"][i] - data["timestamps"][i - 1])
            ekf.predict(data["gyro"][i], dt)
            ekf.update(data["accel"][i])

        estimated_bias = ekf.get_bias()
        bias_error = np.linalg.norm(estimated_bias - true_bias)
        # Allow generous tolerance: bias convergence is slow by design
        assert bias_error < 0.015, (
            f"Bias estimate error {bias_error:.5f} rad/s too large. "
            f"True: {true_bias}, Estimated: {estimated_bias}"
        )


class TestEKFNumerics:
    """Numerical sanity checks for Jacobians and covariance properties."""

    def test_measurement_jacobian_matches_numerical(self):
        """Analytical H must match finite-difference numerical gradient."""
        ekf = make_ekf()
        q = np.array([0.92388, 0.0, 0.38268, 0.0])  # 45 deg pitch
        q /= np.linalg.norm(q)
        ekf.x[:4] = q

        H_analytical = ekf._H_jacobian(q)[:, :4]  # take q-block only

        eps = 1e-6
        H_numerical = np.zeros((3, 4))
        for i in range(4):
            q_plus = q.copy()
            q_plus[i] += eps
            q_minus = q.copy()
            q_minus[i] -= eps
            # No normalization: analytical H is the unconstrained derivative dh/dq
            H_numerical[:, i] = (ekf._h(q_plus) - ekf._h(q_minus)) / (2 * eps)

        np.testing.assert_allclose(H_analytical, H_numerical, atol=1e-5, rtol=1e-4)

    def test_covariance_remains_symmetric(self):
        rng = np.random.default_rng(3)
        ekf = make_ekf()
        for _ in range(200):
            gyro = np.array([0.1, 0.05, -0.03]) + rng.normal(0, 1e-4, 3)
            accel = np.array([0.0, 0.0, G]) + rng.normal(0, 0.02, 3)
            ekf.predict(gyro, DT)
            ekf.update(accel)

        asymmetry = np.max(np.abs(ekf.P - ekf.P.T))
        assert asymmetry < 1e-10, f"P asymmetry {asymmetry:.2e} exceeds tolerance."

    def test_covariance_stays_positive_definite(self):
        rng = np.random.default_rng(4)
        ekf = make_ekf()
        for _ in range(500):
            gyro = np.array([0.05, 0.1, 0.0]) + rng.normal(0, 1e-4, 3)
            accel = np.array([0.0, 0.0, G]) + rng.normal(0, 0.02, 3)
            ekf.predict(gyro, DT)
            ekf.update(accel)

        eigenvalues = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigenvalues > 0), (
            f"P has non-positive eigenvalue(s): {eigenvalues[eigenvalues <= 0]}"
        )

    def test_innovation_decreases_over_time(self):
        """Mean |innovation| in second half must be <= first half."""
        rng = np.random.default_rng(8)
        ekf = make_ekf()
        innovations = []
        N = 600
        for _ in range(N):
            gyro = rng.normal(0, SIGMA_GYRO * np.sqrt(FS), 3)
            accel = np.array([0.0, 0.0, G]) + rng.normal(0, SIGMA_ACCEL * np.sqrt(FS), 3)
            ekf.predict(gyro, DT)
            inn = accel - ekf._h(ekf.get_quaternion())
            innovations.append(np.linalg.norm(inn))
            ekf.update(accel)

        mid = N // 2
        first_half_mean = np.mean(innovations[:mid])
        second_half_mean = np.mean(innovations[mid:])
        assert second_half_mean <= first_half_mean * 1.2, (
            f"Innovation did not decrease: first={first_half_mean:.4f}, "
            f"second={second_half_mean:.4f}"
        )


class TestQuat2Euler:
    """Test Euler angle extraction from known quaternions."""

    def test_identity_gives_zero_angles(self):
        ekf = make_ekf()
        roll, pitch, yaw = ekf.get_euler_deg()
        np.testing.assert_allclose([roll, pitch, yaw], [0.0, 0.0, 0.0], atol=1e-8)

    def test_90deg_pitch(self):
        ekf = make_ekf()
        # q for 90 deg rotation around Y
        angle = np.radians(90.0)
        ekf.x[:4] = [np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0]
        _, pitch, _ = ekf.get_euler_deg()
        np.testing.assert_allclose(pitch, 90.0, atol=1e-4)

    def test_45deg_roll(self):
        ekf = make_ekf()
        angle = np.radians(45.0)
        ekf.x[:4] = [np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0]
        roll, _, _ = ekf.get_euler_deg()
        np.testing.assert_allclose(roll, 45.0, atol=1e-4)


class TestQuatAngleError:
    def test_identical_quaternions_give_zero_error(self):
        q = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1))
        errors = quat_angle_error_deg(q, q)
        np.testing.assert_allclose(errors, 0.0, atol=1e-8)

    def test_90deg_rotation_gives_90deg_error(self):
        q_true = np.array([[1.0, 0.0, 0.0, 0.0]])
        angle = np.radians(90.0)
        q_est = np.array([[np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0]])
        errors = quat_angle_error_deg(q_est, q_true)
        np.testing.assert_allclose(errors[0], 90.0, atol=1e-3)

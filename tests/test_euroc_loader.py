"""
test_euroc_loader.py — Smoke tests for the EuRoC data loader.

Uses synthetic CSV files so no real dataset is required.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow importing from src/ without installing as package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from euroc_loader import load_imu, load_ground_truth


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_IMU = 200   # synthetic samples
N_GT = 100


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """Build a minimal EuRoC-shaped directory with synthetic CSV data."""
    imu_dir = tmp_path / "mav0" / "imu0"
    imu_dir.mkdir(parents=True)

    gt_dir = tmp_path / "mav0" / "state_groundtruth_estimate0"
    gt_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)

    # IMU: timestamp_ns, wx, wy, wz, ax, ay, az
    t_imu_ns = np.arange(N_IMU) * 5_000_000  # 200 Hz → 5 ms steps
    gyro = rng.normal(0.0, 0.01, (N_IMU, 3))
    accel = rng.normal(0.0, 0.1, (N_IMU, 3))
    accel[:, 2] += 9.81  # gravity on Z axis

    with open(imu_dir / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#timestamp [ns]", "wx", "wy", "wz", "ax", "ay", "az"])
        for i in range(N_IMU):
            w.writerow(
                [int(t_imu_ns[i])]
                + list(gyro[i])
                + list(accel[i])
            )

    # Ground truth: timestamp_ns, px, py, pz, qw, qx, qy, qz, vx, vy, vz,
    #               bwx, bwy, bwz, bax, bay, baz
    t_gt_ns = np.arange(N_GT) * 10_000_000  # 100 Hz
    gt_data = rng.normal(0.0, 1.0, (N_GT, 16))
    gt_data[:, 3] = 1.0  # qw ≈ 1 (identity quaternion)

    with open(gt_dir / "data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["#timestamp", "px", "py", "pz", "qw", "qx", "qy", "qz",
             "vx", "vy", "vz", "bwx", "bwy", "bwz", "bax", "bay", "baz"]
        )
        for i in range(N_GT):
            w.writerow([int(t_gt_ns[i])] + list(gt_data[i]))

    return tmp_path


# ---------------------------------------------------------------------------
# IMU loader tests
# ---------------------------------------------------------------------------

def test_imu_keys(fake_dataset):
    imu = load_imu(fake_dataset)
    assert set(imu.keys()) == {"t", "gyro", "accel"}


def test_imu_shapes(fake_dataset):
    imu = load_imu(fake_dataset)
    assert imu["t"].shape == (N_IMU,)
    assert imu["gyro"].shape == (N_IMU, 3)
    assert imu["accel"].shape == (N_IMU, 3)


def test_imu_time_starts_at_zero(fake_dataset):
    imu = load_imu(fake_dataset)
    assert imu["t"][0] == pytest.approx(0.0)


def test_imu_sample_rate(fake_dataset):
    """Check that reconstructed sample rate is approximately 200 Hz."""
    imu = load_imu(fake_dataset)
    dt_mean = np.diff(imu["t"]).mean()
    assert dt_mean == pytest.approx(0.005, rel=1e-3)  # 5 ms


def test_imu_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_imu(tmp_path)


# ---------------------------------------------------------------------------
# Ground-truth loader tests
# ---------------------------------------------------------------------------

def test_gt_keys(fake_dataset):
    gt = load_ground_truth(fake_dataset)
    assert set(gt.keys()) == {"t", "pos", "quat", "vel"}


def test_gt_shapes(fake_dataset):
    gt = load_ground_truth(fake_dataset)
    assert gt["t"].shape == (N_GT,)
    assert gt["pos"].shape == (N_GT, 3)
    assert gt["quat"].shape == (N_GT, 4)
    assert gt["vel"].shape == (N_GT, 3)


def test_gt_time_starts_at_zero(fake_dataset):
    gt = load_ground_truth(fake_dataset)
    assert gt["t"][0] == pytest.approx(0.0)


def test_gt_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ground_truth(tmp_path)

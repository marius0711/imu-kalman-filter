# IMU Signal Processing & Extended Kalman Filter

Orientation estimation from 6-axis IMU data using an Extended Kalman Filter.
Built from first principles: noise characterization → baseline filter → optimal state estimator.

Validated on synthetic data generated from real ADIS16448 noise parameters (EuRoC MAV Dataset spec).
Part of a robotics sensing portfolio — see also
[ros2-imu-fusion](https://github.com/marius0711/ros2-imu-fusion) and
[sensor-calibration](https://github.com/marius0711/sensor-calibration).

---

## What this implements

**Week 1 — Noise characterization & baseline**

The filter is only as good as the noise model that feeds it. Before writing a single line of filtering
code, this project characterizes the sensor:

- Raw IMU signal loading and visualization (EuRoC CSV format, 200 Hz)
- Noise characterization: white noise, bias instability, random walk
- Allan deviation analysis — the industry standard for IMU noise quantification, rarely seen in
  portfolio projects. Identifies noise regimes from log-log slope: ARW (−½), bias instability (0),
  rate random walk (+½). The extracted parameters feed directly into the EKF noise matrices Q and R.
- Complementary filter as a calibrated baseline: fuses gyro (high-pass) and accelerometer (low-pass)
  to estimate roll and pitch. Demonstrates why a fixed-gain filter is insufficient — no uncertainty
  quantification, no bias estimation, yaw unobservable.

**Week 2 — Extended Kalman Filter** *(in progress)*

- State vector: quaternion `[q0, q1, q2, q3]` + gyro bias `[bx, by, bz]` — 7 states
- Process model: quaternion kinematics driven by gyroscope (no gimbal lock)
- Measurement model: gravity reference from accelerometer
- Noise matrices Q and R derived from Allan deviation results
- Validation: RMSE < 2° on synthetic circular motion with injected noise
- Visualization with rerun.io + matplotlib

---

## Why EKF, not a neural network?

An EKF makes its assumptions explicit. The process noise matrix Q encodes how much the bias drifts
per second. The measurement noise matrix R encodes how much we trust the accelerometer. The
covariance matrix P tells you at every timestep exactly how uncertain the estimate is. A neural
network gives you a number — the EKF gives you a number and an honest confidence interval. For
safety-critical robotics systems, that distinction matters.

---

## Repository structure

```
imu-kalman-filter/
├── conftest.py                        # pytest path setup
├── data/
│   ├── download_euroc.sh              # fetch MH_01_easy from ETH ASL
│   └── MH_01_easy/                    # synthetic dataset (EuRoC format)
├── src/
│   ├── euroc_loader.py                # IMU + ground-truth CSV reader
│   ├── generate_synthetic_imu.py      # synthetic IMU generator (ADIS16448 noise params)
│   ├── plot_raw_imu.py                # 6-panel raw signal visualization
│   ├── allan_deviation.py             # Allan deviation noise characterization
│   ├── complementary_filter.py        # complementary filter baseline
│   ├── visualize_rerun.py             # rerun.io 3D orientation viewer
│   └── ekf.py                         # Extended Kalman Filter (Week 2)
├── tests/
│   ├── test_euroc_loader.py           # 9 tests
│   ├── test_complementary_filter.py   # 8 tests
│   └── test_ekf.py                    # Week 2
├── results/                           # generated figures (git-ignored)
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/marius0711/imu-kalman-filter
cd imu-kalman-filter

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate synthetic dataset (EuRoC format, 60s, 200 Hz):

```bash
python3 src/generate_synthetic_imu.py --out data/MH_01_easy
```

Or download the real EuRoC MH_01_easy sequence (~1.6 GB):

```bash
bash data/download_euroc.sh
```

---

## Usage

```bash
# Raw IMU signal plot
python3 src/plot_raw_imu.py --dataset data/MH_01_easy --save results/raw_imu.png

# Allan deviation — noise characterization
python3 src/allan_deviation.py --dataset data/MH_01_easy --save results/allan.png

# Complementary filter
python3 src/complementary_filter.py --dataset data/MH_01_easy --save results/complementary.png

# rerun.io 3D visualization (launches viewer)
python3 src/visualize_rerun.py --dataset data/MH_01_easy

# Run all tests
pytest tests/ -v
```

---

## Test results

```
tests/test_complementary_filter.py::test_output_keys              PASSED
tests/test_complementary_filter.py::test_output_shapes            PASSED
tests/test_complementary_filter.py::test_static_roll_zero         PASSED
tests/test_complementary_filter.py::test_static_known_tilt        PASSED
tests/test_complementary_filter.py::test_cf_less_drift_than_gyro  PASSED
tests/test_complementary_filter.py::test_alpha_effect             PASSED
tests/test_complementary_filter.py::test_accel_to_angles_gravity_aligned  PASSED
tests/test_complementary_filter.py::test_accel_to_angles_pure_roll        PASSED
tests/test_euroc_loader.py::test_imu_keys                         PASSED
tests/test_euroc_loader.py::test_imu_shapes                       PASSED
tests/test_euroc_loader.py::test_imu_time_starts_at_zero          PASSED
tests/test_euroc_loader.py::test_imu_sample_rate                  PASSED
tests/test_euroc_loader.py::test_imu_missing_file_raises          PASSED
tests/test_euroc_loader.py::test_gt_keys                          PASSED
tests/test_euroc_loader.py::test_gt_shapes                        PASSED
tests/test_euroc_loader.py::test_gt_time_starts_at_zero           PASSED
tests/test_euroc_loader.py::test_gt_missing_file_raises           PASSED

17 passed in 1.04s
```

---

## Dataset

EuRoC MAV Dataset, MH_01_easy sequence.
Burri et al., *The EuRoC micro aerial vehicle datasets*, IJRR 2016.
IMU: ADIS16448 at 200 Hz. Ground truth: Leica Nova MS50 total station.

Noise parameters used for the synthetic generator:

| Parameter | Gyroscope | Accelerometer |
|---|---|---|
| Noise density | 1.7e-4 rad/s/√Hz | 2.0e-3 m/s²/√Hz |
| Bias random walk | 1.9e-5 rad/s²/√Hz | 3.0e-3 m/s³/√Hz |

---

## Related projects

This project is the first in a three-part robotics sensing portfolio:

1. **imu-kalman-filter** (this repo) — signal processing and state estimation from scratch
2. [ros2-imu-fusion](https://github.com/marius0711/ros2-imu-fusion) — wraps the EKF as a production ROS2 node
3. [sensor-calibration](https://github.com/marius0711/sensor-calibration) — six-position accelerometer calibration pipeline

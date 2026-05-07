# IMU Signal Processing & Extended Kalman Filter

Orientation estimation from 6-axis IMU data using an Extended Kalman Filter.
Built from first principles: noise characterization → baseline filter → optimal state estimator.

Validated on synthetic data generated from real ADIS16448 noise parameters (EuRoC MAV Dataset spec).
Part of a robotics sensing portfolio — see also
[ros2-imu-fusion](https://github.com/marius0711/ros2-imu-fusion) and
[sensor-calibration](https://github.com/marius0711/sensor-calibration).

---

## What this implements

**Noise characterization**

The filter is only as good as the noise model that feeds it. Before any filtering, the sensor noise
is characterized from data:

- Raw IMU signal loading and visualization (EuRoC CSV format, 200 Hz)
- Allan deviation analysis — the industry standard for IMU noise quantification. Identifies noise
  regimes from log-log slope: angle random walk (−½), bias instability (0), rate random walk (+½).
  The extracted parameters feed directly into the EKF noise matrices Q and R — not manual tuning.

**Complementary filter (baseline)**

Fuses gyroscope (high-pass) and accelerometer (low-pass) to estimate roll and pitch. Demonstrates
why a fixed-gain filter is insufficient: no bias estimation, no uncertainty quantification, yaw
unobservable from accelerometer alone.

**Extended Kalman Filter**

7-dimensional state vector: quaternion `[q0, q1, q2, q3]` + gyro bias `[bx, by, bz]`.

The quaternion parametrization avoids gimbal lock and integrates cleanly via the kinematic
equation `dq/dt = 0.5 · Ω(ω) · q`. The bias is estimated online — not pre-calibrated — which is
what separates a 1-minute filter from a 10-minute filter in practice.

Noise matrices Q and R are derived analytically from Allan deviation results. The Jacobians F (7×7)
and H (3×7) are derived by hand and verified numerically against finite differences.

| Filter | RMSE (20 s pitch sweep) |
|---|---|
| Accelerometer only | 175.6° |
| Complementary filter | 193.5° |
| EKF | **0.15°** |

---

## Why EKF, not a neural network?

An EKF makes its assumptions explicit. Q encodes how much the bias drifts per second. R encodes how
much we trust the accelerometer. The covariance matrix P gives an honest uncertainty estimate at
every timestep. A neural network gives you a number — the EKF gives you a number and a confidence
interval. For safety-critical robotics systems, that distinction matters.

---

## Repository structure

```
imu-kalman-filter/
├── conftest.py
├── data/
│   ├── download_euroc.sh              # fetch MH_01_easy from ETH ASL
│   └── MH_01_easy/                    # synthetic dataset (EuRoC format)
├── src/
│   ├── euroc_loader.py                # IMU + ground-truth CSV reader
│   ├── generate_synthetic_imu.py      # synthetic IMU generator (ADIS16448 noise params)
│   ├── plot_raw_imu.py                # 6-panel raw signal visualization
│   ├── allan_deviation.py             # Allan deviation noise characterization
│   ├── complementary_filter.py        # complementary filter baseline
│   ├── ekf.py                         # Extended Kalman Filter
│   ├── compare_filters.py             # accel-only vs. CF vs. EKF comparison plot
│   └── visualize_rerun.py             # rerun.io 3D orientation viewer
├── tests/
│   ├── test_euroc_loader.py           # 9 tests
│   ├── test_complementary_filter.py   # 8 tests
│   └── test_ekf.py                    # 19 tests
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

Generate synthetic dataset (EuRoC format, 60 s, 200 Hz):

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

# EKF — run on dataset, save quaternion output
python3 src/ekf.py --dataset data/MH_01_easy --save results/quats_ekf.npy

# Filter comparison plot
python3 src/compare_filters.py --save results/filter_comparison.png

# rerun.io 3D visualization (launches viewer)
python3 src/visualize_rerun.py --dataset data/MH_01_easy

# Run all tests
pytest tests/ -v
```

---

## Test results

```
36 passed in 1.99s
```

| Module | Tests | Coverage |
|---|---|---|
| euroc_loader | 9 | shapes, dtypes, time zeroing, missing file |
| complementary_filter | 8 | static tilt, drift correction, alpha effect |
| ekf | 19 | init, static case, RMSE < 2°, bias convergence, Jacobian check, covariance SPD |

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

1. **imu-kalman-filter** (this repo) — signal processing and state estimation from scratch
2. [ros2-imu-fusion](https://github.com/marius0711/ros2-imu-fusion) — wraps the EKF as a ROS2 node
3. [sensor-calibration](https://github.com/marius0711/sensor-calibration) — six-position accelerometer calibration pipeline

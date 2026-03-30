from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np


# ======================================================================
# Base sensor
# ======================================================================

class Sensor:
    """Abstract base for a noisy sensor."""

    def __init__(self, noise_std: float = 0.0):
        self.noise_std = noise_std

    def _add_noise(self, value: np.ndarray | float) -> np.ndarray | float:
        if isinstance(value, np.ndarray):
            return value + np.random.normal(0.0, self.noise_std, value.shape)
        return value + np.random.normal(0.0, self.noise_std)


# ======================================================================
# Simple sensors
# ======================================================================

class PositionSensor(Sensor):
    """Returns the agent's world position with additive Gaussian noise."""

    def __init__(self, noise_std: float = 0.51):
        super().__init__(noise_std)

    def read(self, agent) -> np.ndarray:
        true_pos = np.array(agent.position, dtype=np.float64)
        return self._add_noise(true_pos)


class OrientationSensor(Sensor):
    """Returns the agent's heading angle (rad) with additive Gaussian noise."""

    def __init__(self, noise_std: float = 0.2):
        super().__init__(noise_std)

    def read(self, agent) -> float:
        return float(self._add_noise(agent.orientation))


# ======================================================================
# IMU sensors
# ======================================================================

class Accelerometer(Sensor):
    """Body-frame linear acceleration via velocity finite-difference."""

    def __init__(self, noise_std: float = 0.5):
        super().__init__(noise_std)
        self._prev_velocity: np.ndarray | None = None

    def read(self, agent, dt: float) -> np.ndarray:
        vel = np.array(agent.velocity, dtype=np.float64)
        if self._prev_velocity is None:
            self._prev_velocity = vel.copy()

        # World-frame acceleration
        world_accel = (vel - self._prev_velocity) / dt if dt > 0 else np.zeros(2)
        self._prev_velocity = vel.copy()

        # Rotate into body frame (forward, lateral)
        c = np.cos(-agent.orientation)
        s = np.sin(-agent.orientation)
        body_accel = np.array([
            c * world_accel[0] - s * world_accel[1],
            s * world_accel[0] + c * world_accel[1],
        ])
        return self._add_noise(body_accel)


class Gyroscope(Sensor):
    """Returns the agent's angular velocity (rad/s) with additive noise."""

    def __init__(self, noise_std: float = 0.01):
        super().__init__(noise_std)

    def read(self, agent) -> float:
        return float(self._add_noise(agent.angular_velocity))


class Magnetometer(Sensor):
    """Returns the agent's heading from a simulated magnetic field + noise."""

    def __init__(self, noise_std: float = 0.03):
        super().__init__(noise_std)

    def read(self, agent) -> float:
        return float(self._add_noise(agent.orientation))


@dataclass
class IMUReading:
    accel: np.ndarray   # body-frame [forward, lateral] (m/s²)
    gyro: float         # angular velocity (rad/s)
    heading: float      # magnetometer heading (rad)


class IMU:
    """Composite IMU: accelerometer + gyroscope + magnetometer."""

    def __init__(self,
                 accel_noise: float = 0.5,
                 gyro_noise: float = 0.01,
                 mag_noise: float = 0.03):
        self.accelerometer = Accelerometer(accel_noise)
        self.gyroscope = Gyroscope(gyro_noise)
        self.magnetometer = Magnetometer(mag_noise)

    def read(self, agent, dt: float) -> IMUReading:
        return IMUReading(
            accel=self.accelerometer.read(agent, dt),
            gyro=self.gyroscope.read(agent),
            heading=self.magnetometer.read(agent),
        )


# Odometry estimators have been consolidated into MotionModel in model.py.

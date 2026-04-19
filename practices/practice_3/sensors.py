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


# ======================================================================
# Range sensor (ray-casting lidar / depth camera)
# ======================================================================

class RangeSensor(Sensor):
    """First-person radial depth sensor via vectorised ray casting.

    Casts ``camera_w`` rays uniformly spread over
    ``[orientation − fov_half, orientation + fov_half]``.  The centre column
    always points forward.

    Attributes:
        camera_w:   Number of rays (columns).
        obs_radius: Maximum measurable range in world units.
        ray_step:   March step along each ray in world units (keep ≤ 0.7 to
                    avoid skipping 1×1 world cells).
        fov_half:   Half-FOV angle in radians.  Use ``π`` for full 360° lidar.
    """

    def __init__(self,
                 camera_w:   int   = 360,
                 obs_radius: float = 25.0,
                 ray_step:   float = 0.1,
                 fov_half:   float = np.radians(40.0),
                 noise_std:  float = 0.0):
        super().__init__(noise_std)
        self.camera_w   = camera_w
        self.obs_radius = obs_radius
        self.ray_step   = ray_step
        self.fov_half   = fov_half

    # ------------------------------------------------------------------

    def read(self, agent) -> np.ndarray:
        """Return distances to the first obstacle for each ray.

        Args:
            agent: Agent instance (provides ``position``, ``orientation``,
                   and ``world``).

        Returns:
            Float32 array of shape ``(camera_w,)``, values in
            ``(0, obs_radius]``.  Close obstacles → small values.
        """
        r = self._cast_rays(agent)
        if self.noise_std > 0.0:
            r = np.clip(self._add_noise(r), 0.0, self.obs_radius).astype(np.float32)
        return r

    def _cast_rays(self, agent) -> np.ndarray:
        """Vectorised ray march — identical algorithm to the original Agent._cast_rays."""
        cx, cy = agent.position
        n_rays = self.camera_w

        angles = np.linspace(
            agent.orientation - self.fov_half,
            agent.orientation + self.fov_half,
            n_rays,
        )
        dx_arr = np.cos(angles)
        dy_arr = np.sin(angles)

        t_vals = np.arange(self.ray_step,
                           self.obs_radius + self.ray_step,
                           self.ray_step)

        xs = cx + dx_arr[:, np.newaxis] * t_vals[np.newaxis, :]
        ys = cy + dy_arr[:, np.newaxis] * t_vals[np.newaxis, :]

        ix_grid = np.floor(xs).astype(np.int32)
        iy_grid = np.floor(ys).astype(np.int32)

        ix = ix_grid % agent.world.width
        iy = iy_grid % agent.world.height

        hits = agent.world.state[iy, ix]

        # Diagonal corner gap fix
        ix_prev = np.concatenate([ix_grid[:, :1], ix_grid[:, :-1]], axis=1)
        iy_prev = np.concatenate([iy_grid[:, :1], iy_grid[:, :-1]], axis=1)
        diag = (ix_grid != ix_prev) & (iy_grid != iy_prev)
        if diag.any():
            ix_p = ix_prev % agent.world.width
            iy_p = iy_prev % agent.world.height
            gap_hits = agent.world.state[iy, ix_p] | agent.world.state[iy_p, ix]
            hits = hits | (diag & gap_hits)

        first_idx = np.argmax(hits, axis=1)
        hit_found = hits[np.arange(n_rays), first_idx].astype(bool)
        return np.where(hit_found, t_vals[first_idx], self.obs_radius).astype(np.float32)


# Odometry estimators have been consolidated into MotionModel in model.py.

from __future__ import annotations

from collections import deque

import numpy as np

from sensors import PositionSensor, OrientationSensor, IMU, IMUReading


class PerceptionModel:
    """Ego-centric depth perception model.

    Wraps the agent's radial depth profile (produced by ``agent.perceive()``)
    and exposes helpers for policies that reason about the sensed environment.

    The camera is **orientation-relative**: column 0 always corresponds to the
    agent's current heading (``agent.orientation``), column ``camera_w // 2``
    to directly behind.  Back-projection therefore accounts for the heading
    offset when converting a column index to a world angle.
    """

    def __init__(self):
        self.depth_profile: np.ndarray | None = None  # (camera_w,) normalised [0, 1]
        self.camera_image:  np.ndarray | None = None  # (camera_h, camera_w) float32
        self.sensor_data: dict | None = None          # latest sensor readings

    # ------------------------------------------------------------------

    def update(self, camera_float: np.ndarray, agent,
               sensor_data: dict | None = None) -> None:
        """Store the latest depth image produced by ``agent.perceive()``.

        Args:
            camera_float: Float32 array (camera_h, camera_w) in [0, 1].
            agent:        The Agent instance (orientation, position, world …).
            sensor_data:  Optional dict of sensor readings for policy access.
        """
        self.camera_image  = camera_float
        self.depth_profile = camera_float[0]   # top row = 1-D depth signal
        self.sensor_data   = sensor_data


# ======================================================================
# Motion model — switchable odometry
# ======================================================================

class MotionModel:
    """Unified motion estimator with a switchable odometry backend.

    Two modes (cycle with ``toggle_mode(agent)``):

    ``'simple'``
        GPS-like: records noisy ``PositionSensor`` + ``OrientationSensor``
        readings each tick.  Bounded to world coordinates; no drift.

    ``'imu'``
        Dead-reckoning: integrates gyroscope for heading, then rotates
        body-frame accelerometer readings to world frame and double-integrates
        for position.  Drifts over time — illustrates dead-reckoning error.

    The IMU is *always* read every tick so that ``last_imu_reading`` stays
    current for policies regardless of the active mode.

    Attributes:
        mode:              Active estimator name (``'simple'`` or ``'imu'``).
        position:          Latest estimated position as ``np.ndarray[2]``.
        orientation:       Latest estimated heading in radians.
        gt_trajectory:     Rolling deque of ground-truth ``(x, y)`` tuples.
        est_trajectory:    Rolling deque of estimated ``np.ndarray[2]`` positions.
        last_imu_reading:  Most recent ``IMUReading`` (always fresh).
    """

    MODES = ('simple', 'imu')

    def __init__(self,
                 pos_sensor: PositionSensor,
                 ori_sensor: OrientationSensor,
                 imu: IMU,
                 initial_mode: str = 'simple',
                 max_length: int = 2000):
        self.pos_sensor = pos_sensor
        self.ori_sensor = ori_sensor
        self.imu = imu
        self.mode = initial_mode

        self.gt_trajectory:  deque = deque(maxlen=max_length)
        self.est_trajectory: deque = deque(maxlen=max_length)

        # Public estimated state
        self.position:    np.ndarray = np.zeros(2)
        self.orientation: float      = 0.0

        # Last IMU reading — always populated; available to policies
        self.last_imu_reading: IMUReading | None = None

        # IMU dead-reckoning internal state
        self._imu_velocity:    np.ndarray = np.zeros(2)
        self._imu_initialised: bool       = False

    # ------------------------------------------------------------------

    def _seed_imu(self, agent) -> None:
        """Bootstrap IMU dead-reckoning state from ground truth."""
        self.position         = np.array(agent.position, dtype=np.float64)
        self.orientation      = agent.orientation
        self._imu_velocity    = np.array(agent.velocity, dtype=np.float64)
        self._imu_initialised = True

    def toggle_mode(self, agent) -> str:
        """Cycle to the next odometry mode.

        Clears the estimated trajectory and re-seeds IMU state from ground
        truth when switching to ``'imu'`` mode.

        Returns:
            The new mode name.
        """
        idx       = self.MODES.index(self.mode)
        self.mode = self.MODES[(idx + 1) % len(self.MODES)]
        self.est_trajectory.clear()
        if self.mode == 'imu':
            self._seed_imu(agent)
        return self.mode

    def update(self, agent, dt: float) -> None:
        """Record ground truth and advance the active estimator by one tick.

        The IMU is always read so ``last_imu_reading`` is never stale.

        Args:
            agent: Agent instance (position, velocity, orientation, …).
            dt:    Physics time step in seconds.
        """
        # Ground-truth record
        self.gt_trajectory.append((agent.position[0], agent.position[1]))

        # Always read IMU (keeps last_imu_reading fresh for policies)
        reading = self.imu.read(agent, dt)
        self.last_imu_reading = reading

        if self.mode == 'simple':
            pos              = self.pos_sensor.read(agent)
            self.orientation = self.ori_sensor.read(agent)
            self.position    = pos
            self.est_trajectory.append(pos.copy())

        elif self.mode == 'imu':
            if not self._imu_initialised:
                self._seed_imu(agent)

            # Heading from gyroscope integration
            self.orientation += reading.gyro * dt

            # Body-frame accel → world frame
            c = np.cos(self.orientation)
            s = np.sin(self.orientation)
            world_accel = np.array([
                c * reading.accel[0] - s * reading.accel[1],
                s * reading.accel[0] + c * reading.accel[1],
            ])

            # Double integration: accel → velocity → position
            self._imu_velocity += world_accel * dt
            self.position       = self.position + self._imu_velocity * dt
            self.est_trajectory.append(self.position.copy())

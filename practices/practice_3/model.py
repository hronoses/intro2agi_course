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
# Occupancy grid world model
# ======================================================================

class OccupancyGridModel:
    """Bayesian log-odds occupancy grid updated from ``RangeSensor`` sweeps.

    The grid is **independent of the CA world** — it has its own resolution
    and extent, set at construction time.

    Log-odds inverse sensor model (per ray, measured range *r*, max *R*):

    * **Free** — cells at distance *t < r − Δ/2* along the ray.
    * **Occupied** — cell at *t ≈ r* (only when *r < R*; max-range = no hit).

    Three orientation modes expose the full SLAM uncertainty spectrum:

    ``update(ranges, position, orientation, sensor)``
        Precise known orientation.

    ``update_noisy_orientation(ranges, position, orientation, ori_std, sensor)``
        Gaussian-uncertain orientation: averaged over *n_samples* Monte-Carlo
        draws so uncertainty smears the map rather than creating ghost walls.

    ``update_unknown_orientation(ranges, position, sensor)``
        Orientation completely unknown.  Consecutive scans are matched via
        circular cross-correlation (360° sensor) or normalised cross-correlation
        search (partial FOV) to estimate Δθ, which is accumulated into a
        running absolute-orientation estimate.

    Attributes:
        log_odds:  Float32 array ``(map_height, map_width)`` of log-odds values.
        origin_wx: World-X coordinate of the map's (0, 0) corner.
        origin_wy: World-Y coordinate of the map's (0, 0) corner.
        resolution:  World units per map cell.
    """

    LOG_ODDS_OCC  =  0.85   # increment: occupied observation
    LOG_ODDS_FREE = -0.40   # decrement: free (clear-path) observation
    LOG_ODDS_MIN  = -5.0    # lower saturation (p ≈ 0.7 %)
    LOG_ODDS_MAX  =  5.0    # upper saturation (p ≈ 99.3 %)

    def __init__(self,
                 map_width:  int,
                 map_height: int,
                 resolution: float = 1.0,
                 origin:     tuple[float, float] = (0.0, 0.0),
                 p_prior:    float = 0.5):
        """
        Args:
            map_width:   Number of cells along the X axis.
            map_height:  Number of cells along the Y axis.
            resolution:  World units per cell (e.g. 0.5 → 2 cells per world unit).
            origin:      World coordinates of the map's bottom-left corner.
            p_prior:     Prior occupancy probability (0.5 = fully unknown).
        """
        self.map_width  = map_width
        self.map_height = map_height
        self.resolution = float(resolution)
        self.origin_wx  = float(origin[0])
        self.origin_wy  = float(origin[1])

        prior_lo = float(np.log(p_prior / (1.0 - p_prior)))
        self.log_odds: np.ndarray = np.full(
            (map_height, map_width), prior_lo, dtype=np.float32
        )

        # State for scan-matching orientation tracker
        self._prev_scan:        np.ndarray | None = None
        self._scan_orientation: float             = 0.0

    # ---- Backward-compat aliases ---------------------------------------

    @property
    def width(self) -> int:
        return self.map_width

    @property
    def height(self) -> int:
        return self.map_height

    # ---- Public API ---------------------------------------------------

    @property
    def probability(self) -> np.ndarray:
        """Occupancy probability P(occupied) ∈ [0, 1] — shape (H, W)."""
        return (1.0 / (1.0 + np.exp(-self.log_odds))).astype(np.float32)

    def update(self,
               ranges:      np.ndarray,
               position:    tuple | np.ndarray,
               orientation: float,
               sensor) -> None:
        """Precise-orientation Bayesian update.

        Args:
            ranges:      Raw distances from ``RangeSensor.read()``, shape
                         ``(sensor.camera_w,)``.
            position:    Agent world position ``(x, y)``.
            orientation: Agent heading in radians (precise).
            sensor:      ``RangeSensor`` instance (geometry source).
        """
        buf = np.zeros_like(self.log_odds)
        self._ray_update(buf, ranges, position, orientation, sensor, 1.0)
        self.log_odds += buf
        np.clip(self.log_odds, self.LOG_ODDS_MIN, self.LOG_ODDS_MAX,
                out=self.log_odds)

    def update_noisy_orientation(self,
                                  ranges:      np.ndarray,
                                  position:    tuple | np.ndarray,
                                  orientation: float,
                                  ori_std:     float,
                                  sensor,
                                  n_samples:   int = 7) -> None:
        """Noisy-orientation Bayesian update via Monte-Carlo averaging.

        Draws *n_samples* orientations from N(orientation, ori_std²) and
        averages their log-odds contributions.  The result is a properly
        smeared (uncertain) map update rather than a biased one.

        Args:
            ori_std:   Standard deviation of orientation noise in radians.
            n_samples: Number of orientation samples (≥ 1).
        """
        buf = np.zeros_like(self.log_odds)
        w   = 1.0 / n_samples
        for theta in np.random.normal(orientation, ori_std, n_samples):
            self._ray_update(buf, ranges, position, theta, sensor, w)
        self.log_odds += buf
        np.clip(self.log_odds, self.LOG_ODDS_MIN, self.LOG_ODDS_MAX,
                out=self.log_odds)

    def update_unknown_orientation(self,
                                    ranges:   np.ndarray,
                                    position: tuple | np.ndarray,
                                    sensor) -> float:
        """Unknown-orientation update via scan-matching.

        Matches the current scan against the previous one to estimate Δθ,
        accumulates it into a running orientation estimate, then performs a
        standard precise-orientation update with that estimate.

        Matching strategy:

        * **Full 360° sensor** (``fov_half ≈ π``): FFT circular
          cross-correlation — O(N log N), exact for noise-free scans.
        * **Partial-FOV sensor**: Brute-force normalised cross-correlation
          over ±⌊N/3⌋ shift candidates — slower but works with limited
          overlap.

        Args:
            ranges:   Current range scan, shape ``(camera_w,)``.
            position: Agent world position.
            sensor:   ``RangeSensor`` instance.

        Returns:
            Current accumulated orientation estimate in radians.
        """
        if self._prev_scan is not None:
            delta = self._scan_match_rotation(self._prev_scan, ranges, sensor)
            self._scan_orientation += delta
        self._prev_scan = ranges.copy()
        buf = np.zeros_like(self.log_odds)
        self._ray_update(buf, ranges, position, self._scan_orientation, sensor, 1.0)
        self.log_odds += buf
        np.clip(self.log_odds, self.LOG_ODDS_MIN, self.LOG_ODDS_MAX,
                out=self.log_odds)
        return self._scan_orientation

    # ---- Internals -----------------------------------------------------

    def _ray_update(self,
                    target:      np.ndarray,
                    ranges:      np.ndarray,
                    position:    tuple | np.ndarray,
                    orientation: float,
                    sensor,
                    weight:      float) -> None:
        """Accumulate log-odds increments into *target* (no clipping)."""
        n_rays = sensor.camera_w
        obs_r  = float(sensor.obs_radius)
        step   = float(sensor.ray_step)
        half   = step * 0.5
        cx, cy = float(position[0]), float(position[1])

        angles = np.linspace(
            orientation - sensor.fov_half,
            orientation + sensor.fov_half,
            n_rays,
        )
        dx = np.cos(angles)
        dy = np.sin(angles)

        t_vals = np.arange(step, obs_r + step, step)       # (n_steps,)

        # World positions of all (ray × step) samples
        xs = cx + dx[:, np.newaxis] * t_vals[np.newaxis, :]   # (n_rays, n_steps)
        ys = cy + dy[:, np.newaxis] * t_vals[np.newaxis, :]

        # Map-cell indices (floating-point → integer, no toroidal wrap)
        ix = np.floor((xs - self.origin_wx) / self.resolution).astype(np.int32)
        iy = np.floor((ys - self.origin_wy) / self.resolution).astype(np.int32)

        in_bounds = (
            (ix >= 0) & (ix < self.map_width) &
            (iy >= 0) & (iy < self.map_height)
        )

        r_col   = ranges.astype(np.float64)[:, np.newaxis]   # (n_rays, 1)
        is_free = (t_vals[np.newaxis, :] < r_col - half) & in_bounds
        is_hit  = (
            (np.abs(t_vals[np.newaxis, :] - r_col) <= half) &
            (r_col < obs_r) &    # max-range = no obstacle → skip hit
            in_bounds
        )

        np.add.at(target, (iy[is_free], ix[is_free]), self.LOG_ODDS_FREE * weight)
        np.add.at(target, (iy[is_hit],  ix[is_hit]),  self.LOG_ODDS_OCC  * weight)

    @staticmethod
    def _scan_match_rotation(scan_a: np.ndarray,
                              scan_b: np.ndarray,
                              sensor) -> float:
        """Estimate rotation Δθ such that scan_b ≈ scan_a rotated by Δθ.

        **Convention**: Δθ > 0 means agent turned counter-clockwise (world
        features shift right in the sensor frame), matching the standard
        geometry where orientation increases CCW.

        Full-360° derivation (FFT cross-correlation C = ifft(A* ⊙ B)):

            C[k] = Σ_n a[n] · b[n+k]

        If b[n] = a[n + s] (scan_b is scan_a shifted left by *s* rays), the
        peak is at k = −s. After the standard wrap correction (k > N/2 → k−N):
        k_wrapped = −s → Δθ = −k_wrapped · dφ = s · dφ.

        Args:
            scan_a: Reference scan, shape ``(camera_w,)``.
            scan_b: New scan, shape ``(camera_w,)``.
            sensor: ``RangeSensor`` providing ``camera_w`` and ``fov_half``.

        Returns:
            Δθ in radians.
        """
        n    = sensor.camera_w
        fov  = 2.0 * sensor.fov_half
        dPhi = fov / n          # radians per ray

        if abs(fov - 2.0 * np.pi) < 0.05:
            # ---- Full 360°: FFT circular cross-correlation ----
            A    = np.fft.fft(scan_a.astype(np.float64))
            B    = np.fft.fft(scan_b.astype(np.float64))
            corr = np.real(np.fft.ifft(A.conj() * B))
            k    = int(np.argmax(corr))
            if k > n // 2:
                k -= n
            return -k * dPhi        # Δθ = −k · dφ  (see derivation above)

        else:
            # ---- Partial FOV: brute-force NCC search ----
            # scan_b[i] ≈ scan_a[i + s_true]  →  np.roll(scan_b, s_true)[i]
            # = scan_b[i − s_true] ≈ scan_a[i]
            # Peak of NCC(scan_a, roll(scan_b, s)) is at s = s_true.
            max_shift = n // 3
            a_z     = scan_a - scan_a.mean()
            norm_a  = float(np.linalg.norm(a_z)) + 1e-9
            best_s, best_ncc = 0, -np.inf
            for s in range(-max_shift, max_shift + 1):
                b_r   = np.roll(scan_b, s)
                b_z   = b_r - b_r.mean()
                ncc   = float(np.dot(a_z, b_z)) / (norm_a * (float(np.linalg.norm(b_z)) + 1e-9))
                if ncc > best_ncc:
                    best_ncc = ncc
                    best_s   = s
            return best_s * dPhi    # Δθ = s_true · dφ


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

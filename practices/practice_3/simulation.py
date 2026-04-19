import time
import queue
import numpy as np
from pynput import keyboard as kb

from world import World
from agent import Agent
from model import PerceptionModel, MotionModel, OccupancyGridModel
from action import Policy
from viz_rerun import RerunViz
from sensors import PositionSensor, OrientationSensor, IMU


_CONTROLS = """\
Controls (type anywhere — global keyboard capture):
  a / d          - Rotate left / right (hold for continuous torque)  [keyboard mode]
  w / s          - Move forward / backward (hold for continuous force)  [keyboard mode]
  SPACE          - Pause / Resume GoL
  + / -          - GoL speed up / slow down (halve / double world_dt)
  e              - Toggle cell under agent (0 ↔ 1)
  z / x          - Kill / Revive cell under agent
  t              - Toggle keyboard / auto policy
  m              - Cycle odometry estimator (simple ↔ imu)
  o              - Cycle occupancy orientation mode (precise / noisy / unknown)
  q / ESC        - Quit"""

class Simulation:
    def __init__(self,
                 world_size:  tuple[int, int]     = (100, 100),
                 start_pos:   tuple[float, float] | None = None,
                 camera_w:    int   = 360,
                 camera_h:    int   = 64,
                 obs_radius:  float = 25.0,
                 ray_step:    float = 0.1,
                 warmup:      int   = 1100,
                 use_rerun:   bool  = True):
        """Create the world, agent, and feature tracker.

        Args:
            world_size:  (width, height) of the GoL grid.
            start_pos:   Initial agent position in world units (default: centre).
            camera_w:    Number of rays / depth image width.
            camera_h:    Depth image display height.
            obs_radius:  Maximum sensing distance in world units.
            ray_step:    Ray march step size in world units (keep ≤ 0.7).
            warmup:      Number of GoL steps run before the main loop starts.
            use_rerun:   Whether to open the Rerun visualisation window.
        """
        self.world = World(*world_size)
        if start_pos is None:
            start_pos = (world_size[0] / 2.0, world_size[1] / 2.0)

        self.agent   = Agent(self.world, start_pos,
                             camera_w=camera_w, camera_h=camera_h,
                             obs_radius=obs_radius, ray_step=ray_step)
        self.tracker  = PerceptionModel()

        # Occupancy map: 2× map resolution vs. CA grid (not aligned)
        _map_res = 0.5
        self.occupancy = OccupancyGridModel(
            map_width  = int(self.world.width  / _map_res),
            map_height = int(self.world.height / _map_res),
            resolution = _map_res,
            origin     = (0.0, 0.0),
        )
        self.occ_orientation_mode: str = 'precise'   # 'precise' | 'noisy' | 'unknown'

        self._policy:      Policy | None = None
        self.auto_control: bool          = False

        self._quit        = False
        self.paused       = False
        self.world_dt     = 1.0    # seconds per world (GoL) step
        self._world_acc   = 0.0    # accumulated world time
        self.dt           = 0.01   # agent physics time step (seconds)
        self._agent_tick  = 0      # physics-step counter for high-rate metrics
        self._held_keys:  set[str] = set()  # currently depressed movement keys

        self._rr = RerunViz(world_size=(self.world.width, self.world.height)) if use_rerun else None

        # ---- Sensors & motion model ----------------------------------
        pos_sensor = PositionSensor(noise_std=0.05)
        ori_sensor = OrientationSensor(noise_std=0.02)
        imu        = IMU(accel_noise=0.5, gyro_noise=0.21, mag_noise=0.03)
        self.motion_model = MotionModel(pos_sensor, ori_sensor, imu)

        print(f"Running {warmup} warm-up steps...")
        for _ in range(warmup):
            self.world.step()
        print(f"Done. Starting at step {self.world.time_step}.")

    # ------------------------------------------------------------------
    # Policy property
    # ------------------------------------------------------------------

    @property
    def policy(self) -> Policy | None:
        return self._policy

    @policy.setter
    def policy(self, p: Policy):
        self._policy = p
        print(f"Policy set: {p.name}")

    # ------------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------------

    def _handle_key(self, key: str) -> bool:
        """Process one-shot key events.  Returns False to request quit."""
        if key in ('q', '\x1b'):
            return False
        elif key == ' ':
            self.paused = not self.paused
        elif key in ('+', '='):
            self.world_dt = max(0.1, self.world_dt / 2.0)
            print(f"World step period: {self.world_dt:.2f}s")
        elif key == '-':
            self.world_dt = min(16.0, self.world_dt * 2.0)
            print(f"World step period: {self.world_dt:.2f}s")
        elif key == 't':
            self.auto_control = not self.auto_control
            mode_str = (f'AUTO ({self._policy.name})' if (self.auto_control and self._policy)
                        else 'KEYBOARD')
            print(f"Control mode: {mode_str}")
        elif key == 'm':
            new_mode = self.motion_model.toggle_mode(self.agent)
            print(f"Odometry estimator: {new_mode}")
        elif key == 'o':
            modes = ('precise', 'noisy', 'unknown')
            self.occ_orientation_mode = modes[
                (modes.index(self.occ_orientation_mode) + 1) % len(modes)
            ]
            print(f"Occupancy orientation mode: {self.occ_orientation_mode}")
        elif key == 'e':
            self.agent.toggle_cell()
        elif key == 'z':
            self.agent.set_cell(0)
        elif key == 'x':
            self.agent.set_cell(1)
        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        print(_CONTROLS)
        key_q: queue.SimpleQueue = queue.SimpleQueue()

        _MOVE_KEYS = frozenset('wasd')

        def on_press(key):
            try:
                c = key.char
            except AttributeError:
                if key == kb.Key.esc:
                    key_q.put('\x1b')
                elif key == kb.Key.space:
                    key_q.put(' ')
                return
            if c in _MOVE_KEYS:
                self._held_keys.add(c)
            else:
                key_q.put(c)

        def on_release(key):
            try:
                c = key.char
            except AttributeError:
                return
            self._held_keys.discard(c)

        listener = kb.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()

        last_time = time.time()

        try:
            while not self._quit:
                # ---- Drain keyboard queue ------------------------------
                while True:
                    try:
                        k = key_q.get_nowait()
                        if not self._handle_key(k):
                            self._quit = True
                            break
                    except queue.Empty:
                        break

                if self._quit:
                    break

                # ---- Real-time physics ---------------------------------
                now     = time.time()
                elapsed = now - last_time
                last_time = now
                # Cap accumulated time to avoid spiral-of-death on slow hardware
                accumulated = min(elapsed, 0.05)
                while accumulated >= self.dt:
                    # Continuous force from held movement keys
                    if not self.auto_control:
                        heading = self.agent.orientation
                        f = self.agent.kick_force
                        fw = self.agent.kick_torque
                        if 'w' in self._held_keys:
                            self.agent.apply_force(
                                f * np.cos(heading) * self.dt,
                                f * np.sin(heading) * self.dt,
                            )
                        if 's' in self._held_keys:
                            self.agent.apply_force(
                                -f * np.cos(heading) * self.dt,
                                -f * np.sin(heading) * self.dt,
                            )
                        if 'a' in self._held_keys:
                            self.agent.apply_torque(- fw * self.dt)
                        if 'd' in self._held_keys:
                            self.agent.apply_torque(fw * self.dt)
                    self.agent.update(self.dt)
                    self._agent_tick += 1

                    # ---- Motion model update --------------------------
                    self.motion_model.update(self.agent, self.dt)

                    if self._rr is not None:
                        self._rr.log_metrics(self._agent_tick, self.agent)
                        self._rr.log_odometry_metrics(
                            self._agent_tick, self.motion_model, self.agent)
                    accumulated -= self.dt

                # ---- GoL advance (time-based, independent of agent dt) -
                if not self.paused:
                    self._world_acc += elapsed
                    # Cap so we don't burst many steps after a long pause/lag
                    self._world_acc = min(self._world_acc, self.world_dt)
                    while self._world_acc >= self.world_dt:
                        self.world.step()
                        self._world_acc -= self.world_dt

                # ---- Perceive + update perception model ----------------
                camera_float = self.agent.perceive()
                sensor_data = {
                    'estimated_position':    self.motion_model.position.copy(),
                    'estimated_orientation': self.motion_model.orientation,
                    'imu':                   self.motion_model.last_imu_reading,
                    'odom_mode':             self.motion_model.mode,
                }
                self.tracker.update(camera_float, self.agent,
                                    sensor_data=sensor_data)

                # ---- Occupancy-map update (orientation mode) -----------
                ranges = self.agent._depth_profile
                if ranges is not None:
                    pos    = self.agent.position
                    sensor = self.agent.range_sensor
                    mode   = self.occ_orientation_mode
                    if mode == 'precise':
                        self.occupancy.update(
                            ranges, pos, self.agent.orientation, sensor)
                    elif mode == 'noisy':
                        self.occupancy.update_noisy_orientation(
                            ranges, pos,
                            self.motion_model.orientation,
                            ori_std=0.15,
                            sensor=sensor,
                        )
                    else:  # 'unknown'
                        self.occupancy.update_unknown_orientation(
                            ranges, pos, sensor)

                # ---- Auto policy ---------------------------------------
                if self.auto_control and self._policy is not None and not self.paused:
                    result = self._policy(self.agent, self.tracker)
                    if result is not None:
                        fx, fy = result
                        self.agent.apply_force(fx, fy)

                # ---- Rerun visualisation -------------------------------
                if self._rr is not None:
                    self._rr.log_frame(
                        self.world.time_step, self.world, self.agent, self.tracker,
                        camera_float=camera_float,
                        paused=self.paused,
                        odom_mode=self.motion_model.mode,
                        occupancy=self.occupancy,
                    )
                    self._rr.log_trajectories(
                        self.motion_model.gt_trajectory,
                        self.motion_model.est_trajectory,
                        self.motion_model.mode,
                    )

                time.sleep(0.01)

        finally:
            listener.stop()

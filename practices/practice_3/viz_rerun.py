import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from collections import deque


class RerunViz:
    """Rerun visualization for practice_3: continuous-agent and ego-centric
    depth perception.

    Panels:
      world/state           — full world grid + agent marker + heading arrow
      agent/camera          — ego-centric depth panorama (column 0 = forward)
      metrics/pos_x         — agent x(t) time series
      metrics/pos_y         — agent y(t) time series
      metrics/speed         — agent speed |v| time series
      metrics/orientation   — agent heading angle (rad) time series
      debug/info            — Markdown state summary
    """

    def __init__(self, app_id: str = 'practice3_sim',
                 world_size: tuple[int, int] = (100, 100)):
        rr.init(app_id)
        self._world_size = world_size

        blueprint = rrb.Blueprint(
            rrb.Vertical(
                # Top row: world overview + camera view + depth curve
                rrb.Horizontal(
                    rrb.Spatial2DView(name='World State', origin='world/state'),
                    rrb.Vertical(
                        rrb.Spatial2DView(name='Camera + Features', origin='agent/camera'),
                        rrb.Spatial2DView(name='Depth Curve',       origin='agent/depth_curve'),
                        row_shares=[2, 1],
                    ),
                    column_shares=[1, 1],
                ),
                # Middle row: agent metrics
                rrb.Horizontal(
                    rrb.TimeSeriesView(name='X position',   origin='metrics/pos_x'),
                    rrb.TimeSeriesView(name='Y position',   origin='metrics/pos_y'),
                    rrb.TimeSeriesView(name='Speed',        origin='metrics/speed'),
                    rrb.TimeSeriesView(name='Orientation',  origin='metrics/orientation'),
                    column_shares=[1, 1, 1, 1],
                ),
                # Third row: odometry comparison (GT vs estimator)
                rrb.Horizontal(
                    rrb.TimeSeriesView(name='Odom X (GT vs estimated)', origin='metrics/odom_x'),
                    rrb.TimeSeriesView(name='Odom Y (GT vs estimated)', origin='metrics/odom_y'),
                    column_shares=[1, 1],
                ),
                # Bottom row: debug text
                rrb.TextDocumentView(name='Debug', origin='debug/info'),
                row_shares=[3, 2, 1, 1],
            ),
            collapse_panels=True,
        )

        rr.spawn()
        rr.send_blueprint(blueprint, make_active=True, make_default=True)

        # Static series style metadata (logged once)
        rr.log('metrics/pos_x',       rr.SeriesLines(names=['x'],           colors=[(100, 220, 100)]), static=True)
        rr.log('metrics/pos_y',       rr.SeriesLines(names=['y'],           colors=[(100, 180, 255)]), static=True)
        rr.log('metrics/speed',       rr.SeriesLines(names=['speed'],       colors=[(255, 200,  80)]), static=True)
        rr.log('metrics/orientation', rr.SeriesLines(names=['orientation'], colors=[(220, 100, 255)]), static=True)

        # Odometry comparison series (GT + single active estimator)
        rr.log('metrics/odom_x/gt',  rr.SeriesLines(names=['GT x'],  colors=[(50, 220, 50)]),  static=True)
        rr.log('metrics/odom_x/est', rr.SeriesLines(names=['Est x'], colors=[(255, 200, 50)]), static=True)
        rr.log('metrics/odom_y/gt',  rr.SeriesLines(names=['GT y'],  colors=[(50, 220, 50)]),  static=True)
        rr.log('metrics/odom_y/est', rr.SeriesLines(names=['Est y'], colors=[(255, 200, 50)]), static=True)

        # Visual legend overlay on world panel (static — logged once)
        self._log_legend()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def log_frame(self, world_time: int, world, agent, model,
                  camera_float: np.ndarray | None = None,
                  paused: bool = False,
                  odom_mode: str = 'simple'):
        """Log one simulation frame to Rerun.

        Args:
            world_time:   Current GoL step number (used as Rerun timeline).
            world:        World instance.
            agent:        Agent instance.
            model:        PerceptionModel instance.
            camera_float: Precomputed float32 (H, W) depth image, or None to
                          call agent.perceive() internally.
            paused:       Whether the simulation is paused.
        """
        rr.reset_time()
        rr.set_time('step', sequence=world_time)

        self._log_world(world, agent)
        self._log_camera(agent, camera_float)
        self._log_depth_curve(agent)
        self._log_debug(world, agent, paused, odom_mode)

    def log_metrics(self, agent_tick: int, agent):
        """Log agent metrics on a faster physics timeline."""
        rr.reset_time()
        rr.set_time('agent_tick', sequence=agent_tick)
        self._log_metrics(agent)

    def log_odometry_metrics(self, agent_tick: int, motion_model, agent) -> None:
        """Log GT vs active estimator X/Y comparison on the physics timeline."""
        rr.reset_time()
        rr.set_time('agent_tick', sequence=agent_tick)
        rr.log('metrics/odom_x/gt',  rr.Scalars([agent.position[0]]))
        rr.log('metrics/odom_y/gt',  rr.Scalars([agent.position[1]]))
        rr.log('metrics/odom_x/est', rr.Scalars([float(motion_model.position[0])]))
        rr.log('metrics/odom_y/est', rr.Scalars([float(motion_model.position[1])]))

    def log_trajectories(self,
                         gt: deque,
                         estimated: deque,
                         mode: str) -> None:
        """Overlay GT and estimated trajectory lines on the world panel.

        Positions are mod-wrapped to world bounds and split at toroidal
        discontinuities.  The estimated line color reflects the active mode:
        blue for ``'simple'``, red for ``'imu'``.

        Args:
            gt:        Ground-truth positions — deque of (x, y) tuples.
            estimated: Estimated positions — deque of np.ndarray[2].
            mode:      Active estimator name (``'simple'`` or ``'imu'``).
        """
        ww, wh = float(self._world_size[0]), float(self._world_size[1])
        est_color = (80, 160, 255) if mode == 'simple' else (255, 80, 80)

        if len(gt) >= 2:
            segs = self._wrap_segments(np.array(gt, dtype=np.float32), ww, wh)
            if segs:
                rr.log('world/state/trajectory_gt', rr.LineStrips2D(
                    segs, colors=[(50, 220, 50)],
                ))
        if len(estimated) >= 2:
            segs = self._wrap_segments(
                np.array(list(estimated), dtype=np.float32), ww, wh)
            if segs:
                rr.log('world/state/trajectory_est', rr.LineStrips2D(
                    segs, colors=[est_color],
                ))
        else:
            # Clear stale line after mode switch
            rr.log('world/state/trajectory_est', rr.LineStrips2D([]))

    # ------------------------------------------------------------------
    # Per-panel helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_segments(pts: np.ndarray, ww: float, wh: float) -> list[np.ndarray]:
        """Mod-wrap positions to world bounds and split at wrap discontinuities.

        A jump > 40 % of world size on either axis signals a toroidal wrap;
        the strip is cut there so no line crosses the world edge visually.
        """
        wrapped = pts % np.array([[ww, wh]], dtype=np.float32)
        if len(wrapped) < 2:
            return []
        diff = np.abs(np.diff(wrapped, axis=0))
        jump = (diff[:, 0] > ww * 0.4) | (diff[:, 1] > wh * 0.4)
        splits = np.flatnonzero(jump) + 1
        return [seg for seg in np.split(wrapped, splits) if len(seg) >= 2]

    def _log_legend(self) -> None:
        """Log a static colored-line legend in the top-left of the world panel."""
        lx, ly = 1.5, 1.5   # top-left anchor
        ll     = 7.0         # line length (world units)
        gap    = 2.2         # vertical spacing
        r      = 0.35        # line radius (thickness)

        entries = [
            ('world/state/legend/gt',     ly,       (50,  220,  50), 'GT'),
            ('world/state/legend/simple', ly + gap, (80,  160, 255), 'Simple odom (m)'),
            ('world/state/legend/imu',    ly + 2*gap, (255, 80,  80), 'IMU odom (m)'),
        ]
        for path, y, color, _label in entries:
            rr.log(path, rr.LineStrips2D(
                [[[lx, y], [lx + ll, y]]], colors=[color], radii=[r]
            ), static=True)
        rr.log('world/state/legend/labels', rr.Points2D(
            [[lx + ll + 0.5, ly],
             [lx + ll + 0.5, ly + gap],
             [lx + ll + 0.5, ly + 2*gap]],
            labels=['GT', 'Simple odom (m)', 'IMU odom (m)'],
            colors=[(50, 220, 50), (80, 160, 255), (255, 80, 80)],
            radii=[0.01],
        ), static=True)

    def _log_world(self, world, agent):
        """World grid image with agent marker and camera FOV overlay."""
        rgb = np.zeros((world.height, world.width, 3), dtype=np.uint8)
        rgb[world.state == 1] = (255, 255, 255)
        rr.log('world/state', rr.Image(rgb))

        ax = agent.position[0] % world.width
        ay = agent.position[1] % world.height
        rr.log('world/state/agent', rr.Points2D(
            [[ax, ay]], colors=[(255, 50, 50)], radii=[0.7],
        ))

        # Orientation arrow
        arrow_len = 3.0
        dx = arrow_len * np.cos(agent.orientation)
        dy = arrow_len * np.sin(agent.orientation)
        rr.log('world/state/heading', rr.Arrows2D(
            origins=[[ax, ay]],
            vectors=[[dx, dy]],
            colors=[(255, 200, 0)],
        ))

        # FOV cone: apex → left edge arc → right edge arc → back to apex
        r        = agent.obs_radius
        fov_half = agent.fov_half
        n_arc    = 24
        arc_angles = np.linspace(agent.orientation - fov_half,
                                 agent.orientation + fov_half, n_arc)
        arc_pts = np.column_stack([
            ax + r * np.cos(arc_angles),
            ay + r * np.sin(arc_angles),
        ])
        cone_pts = np.vstack([[[ax, ay]], arc_pts, [[ax, ay]]])
        rr.log('world/state/fov', rr.LineStrips2D(
            [cone_pts], colors=[(50, 200, 50)],
        ))

    def _log_camera(self, agent, camera_float):
        """Ego-centric depth panorama (column 0 = forward / heading)."""
        if camera_float is None:
            camera_float = agent.perceive()

        grey = (camera_float * 255).astype(np.uint8)
        rgb  = np.stack([grey, grey, grey], axis=2)
        rr.log('agent/camera', rr.Image(rgb))

    def _log_depth_curve(self, agent):
        """Signed-distance curve: x = angle in local FOV frame (degrees),
        y = depth distance (world units).  Left edge = -fov_half, centre = 0°
        (forward), right edge = +fov_half."""
        if agent._depth_profile is None:
            return
        fov_deg = np.degrees(agent.fov_half)
        angles  = np.linspace(-fov_deg, fov_deg, agent.camera_w, dtype=np.float32)
        depths  = agent._depth_profile                             # (camera_w,) world units
        pts = np.column_stack([angles, -depths])                   # y inverted so far = up
        rr.log('agent/depth_curve', rr.LineStrips2D(
            [pts], colors=[(80, 220, 120)],
        ))
        # Forward marker at angle=0
        max_d = float(agent.obs_radius)
        rr.log('agent/depth_curve/forward', rr.LineStrips2D(
            [[[0.0, 0.0], [0.0, -max_d]]],
            colors=[(255, 200, 50)],
        ))

    def _log_metrics(self, agent):
        """Position, speed, and orientation time series."""
        vx, vy = agent.velocity
        speed  = float(np.hypot(vx, vy))
        rr.log('metrics/pos_x',       rr.Scalars([agent.position[0]]))
        rr.log('metrics/pos_y',       rr.Scalars([agent.position[1]]))
        rr.log('metrics/speed',       rr.Scalars([speed]))
        rr.log('metrics/orientation', rr.Scalars([agent.orientation]))

    def _log_debug(self, world, agent, paused, odom_mode: str = 'simple'):
        """Markdown debug summary panel."""
        px, py = agent.position
        vx, vy = agent.velocity
        status = 'PAUSED' if paused else 'running'
        heading_deg = np.degrees(agent.orientation) % 360.0
        mode_label = {
            'simple': '🔵 **Blue** — simple odometry (noisy position sensor)',
            'imu':    '🔴 **Red**  — IMU dead-reckoning (accel + gyro)',
        }.get(odom_mode, odom_mode)
        lines = [
            '## Simulation State',
            '',
            '| Field | Value |',
            '|-------|-------|',
            f'| **GoL Step**      | {world.time_step} |',
            f'| **Status**        | {status} |',
            f'| **Position**      | ({px:.2f}, {py:.2f}) |',
            f'| **Velocity**      | ({vx:.2f}, {vy:.2f}) |',
            f'| **Heading**       | {heading_deg:.1f}° |',
            f'| **Obs radius**    | {agent.obs_radius:.1f} wu |',
            f'| **Odom mode**     | `{odom_mode}` |',
            '',
            '### Trajectory Legend',
            '🟢 **Green** — ground truth',
            '',
            f'Active estimator: {mode_label}',
            '',
            '_Press **m** to cycle odometry estimator_',
        ]
        rr.log('debug/info', rr.TextDocument('\n'.join(lines), media_type='text/markdown'))

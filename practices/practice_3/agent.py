import numpy as np

from sensors import RangeSensor


class Agent:
    """Agent with continuous 2D position, impulse-based physics, and first-person
    radial depth perception r(φ)."""

    def __init__(self, world, start_pos=(10.0, 10.0),
                 camera_w=360, camera_h=64,
                 obs_radius=30.0, ray_step=0.5):
        """Initialise agent.

        Args:
            world:       The World instance.
            start_pos:   Continuous (x, y) starting position in world units.
            camera_w:    Number of rays / image width.  Each column covers one
                         evenly-spaced angle in [0, 2π).
            camera_h:    Height of the 2-D display image (rows tiled from the
                         1-D depth signal).
            obs_radius:  Maximum sensing distance in world units.  Cells beyond
                         this distance are invisible to the agent.
            ray_step:    Step size along each ray in world units.  Keep ≤ 0.7 to
                         ensure no 1×1 cell is skipped.
        """
        self.world      = world
        self.position   = list(start_pos)   # continuous [x, y]
        self.velocity   = [0.0, 0.0]        # [vx, vy]

        self.camera_w   = camera_w
        self.camera_h   = camera_h
        self.obs_radius = obs_radius
        self.ray_step   = ray_step

        # Cached depth profile from the most recent perceive() call.
        # Shape: (camera_w,), used by PerceptionModel for back-projection.
        self._depth_profile: np.ndarray | None = None

        # Range sensor — owns the ray-casting geometry
        self.range_sensor = RangeSensor(
            camera_w   = camera_w,
            obs_radius = obs_radius,
            ray_step   = ray_step,
            fov_half   = np.radians(40.0),
        )

        # Orientation (radians, 0 = right / +x, increases clockwise with +y downward)
        self.orientation         = 0.0                           # heading angle
        self.angular_velocity    = 0.0                           # rad / s
        self.fov_half            = self.range_sensor.fov_half    # mirror for viz

        # Geometry
        self.radius = 0.5    # world units — agent occupies ~1 cell

        # Linear dynamics
        self.mass           = 1.0
        # self.mass           = 0.1
        self.linear_damping = 2.0   # N·s/m  — viscous drag
        self.kick_force     = 50.0   # N·s    — linear velocity impulse per key press

        # Rotational dynamics  (solid-disc model: I = ½mr²)
        self.moment_of_inertia = 0.5 * self.mass * self.radius ** 2   # kg·m²
        self.angular_damping   = 0.325   # N·m·s/rad — rotational drag  (= linear_damping × I)
        self.kick_torque       = 0.7    # N·m·s     — angular impulse per key press (Δω = 2 rad/s)

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def apply_force(self, fx: float, fy: float):
        """Apply a linear impulse (N·s) — changes velocity by F/m."""
        self.velocity[0] += fx / self.mass
        self.velocity[1] += fy / self.mass

    def apply_torque(self, tau: float):
        """Apply an angular impulse (N·m·s) — changes angular velocity by τ/I."""
        self.angular_velocity += tau / self.moment_of_inertia

    def update(self, dt: float, restitution: float = 0.45):
        """Advance physics by one time step *dt* with live-cell collision.

        Each linear axis is solved independently.  Angular velocity is
        integrated and damped by angular friction each step.
        """
        # ---- Angular physics ------------------------------------------
        # Damping torque τ = −b·ω  →  dω/dt = τ/I = −(b/I)·ω
        self.angular_velocity += -(self.angular_damping / self.moment_of_inertia) * self.angular_velocity * dt
        if abs(self.angular_velocity) < 0.001:
            self.angular_velocity = 0.0
        self.orientation = (self.orientation + self.angular_velocity * dt) % (2.0 * np.pi)

        # ---- Linear physics -------------------------------------------
        for i in range(2):
            friction_force = -self.linear_damping * self.velocity[i]
            self.velocity[i] += (friction_force / self.mass) * dt

            dim       = self.world.width if i == 0 else self.world.height
            new_coord = (self.position[i] + self.velocity[i] * dt) % dim

            cx = int(new_coord)        % self.world.width  if i == 0 else int(self.position[0]) % self.world.width
            cy = int(self.position[1]) % self.world.height if i == 0 else int(new_coord)        % self.world.height

            if self.world.state[cy, cx] == 1:
                self.velocity[i] = -self.velocity[i] * restitution
            else:
                self.position[i] = new_coord

            if abs(self.velocity[i]) < 0.01:
                self.velocity[i] = 0.0

    # ------------------------------------------------------------------
    # First-person radial depth perception
    # ------------------------------------------------------------------

    def perceive(self) -> np.ndarray:
        """Ego-centric radial depth image within the ±fov_half arc.

        Delegates to ``self.range_sensor``.  Column 0 = left edge (−fov_half),
        centre column = forward, last column = right edge (+fov_half).
        Values are normalised to [0, 1]: bright = close obstacle, dark = clear.

        Returns:
            Float32 array of shape (camera_h, camera_w), values in [0, 1].
        """
        r = self.range_sensor.read(self)               # (camera_w,) world-unit distances
        self._depth_profile = r                        # cache raw ranges for models
        row = (1.0 - r / self.obs_radius).astype(np.float32)
        return np.tile(row[np.newaxis, :], (self.camera_h, 1))

    # ------------------------------------------------------------------
    # Cell editing helpers
    # ------------------------------------------------------------------

    def toggle_cell(self):
        """Flip the cell under the agent's current position."""
        x = int(self.position[0]) % self.world.width
        y = int(self.position[1]) % self.world.height
        self.world.set_value(x, y, 1 - self.world.get_value(x, y))

    def set_cell(self, value: int):
        """Set the cell under the agent's current position to *value*."""
        x = int(self.position[0]) % self.world.width
        y = int(self.position[1]) % self.world.height
        self.world.set_value(x, y, value)

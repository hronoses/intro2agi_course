import numpy as np
import cv2
from dataclasses import dataclass

np.random.seed(42)  # For reproducibility

class World:
    """2D binary world represented as a binary matrix."""
    
    def __init__(self, width=20, height=20):
        """Initialize world with random binary values."""
        self.width = width
        self.height = height
        self.state = np.random.randint(0, 2, size=(height, width))
        self.time_step = 0
    
    def get_value(self, x, y):
        """Get the value at a specific position (with wrapping)."""
        return self.state[y % self.height, x % self.width]
    
    def set_value(self, x, y, value):
        """Set the value at a specific position (with wrapping)."""
        self.state[y % self.height, x % self.width] = value

    def step(self):
        """Advance the world by one time step using Conway's Game of Life rules."""
        # Pad once with wrap-around, then sum all 8 neighbours via slicing.
        # This avoids 16 temporary arrays that np.roll would create.
        p = np.pad(self.state, 1, mode='wrap')
        neighbours = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                      p[1:-1, :-2]               + p[1:-1, 2:] +
                      p[2:,  :-2]  + p[2:,  1:-1]  + p[2:,  2:])
        survive = (self.state == 1) & ((neighbours == 2) | (neighbours == 3))
        born    = (self.state == 0) &  (neighbours == 3)
        self.state = (survive | born).view(np.uint8)
        self.time_step += 1

    def visualize_opencv(self, agent_pos=None, cell_size=7, fov=None):
        """Create an OpenCV image of the world state (fully vectorised).
        
        Args:
            agent_pos: Continuous (x, y) float position of the agent.
            cell_size: Pixels per world cell.
            fov: Optional (fov_w, fov_h) in world units to draw the camera FOV rectangle.
        """
        # Build a (height x width x 3) colour map in one shot
        colour_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        colour_map[self.state == 1] = (255, 255, 255)   # live  -> white

        # Scale up to pixel resolution via np.repeat (no Python loops)
        img = np.repeat(np.repeat(colour_map, cell_size, axis=0), cell_size, axis=1)

        # Draw grid lines with single array writes
        for i in range(self.width):
            img[:, i * cell_size] = (50, 50, 50)
        for i in range(self.height):
            img[i * cell_size, :] = (50, 50, 50)

        # Draw agent as a dot at its continuous pixel position
        if agent_pos is not None:
            img_h, img_w = img.shape[:2]
            ax_f = (agent_pos[0] % self.width) * cell_size
            ay_f = (agent_pos[1] % self.height) * cell_size
            cv2.circle(img, (int(ax_f), int(ay_f)), max(3, cell_size // 2), (0, 0, 255), -1)

            # Draw camera FOV rectangle with toroidal wrapping:
            # split into up to 4 visible segments when the FOV crosses a boundary.
            if fov is not None:
                fov_w, fov_h = fov
                half_w_px = fov_w / 2 * cell_size
                half_h_px = fov_h / 2 * cell_size

                def wrap_intervals(lo, hi, size):
                    """Return 1 or 2 pixel intervals mapping [lo,hi] onto [0,size)."""
                    lo_w = lo % size
                    hi_w = hi % size
                    if lo_w <= hi_w:
                        return [(int(lo_w), int(hi_w))]
                    else:  # crosses boundary
                        return [(0, int(hi_w)), (int(lo_w), int(size - 1))]

                x_segs = wrap_intervals(ax_f - half_w_px, ax_f + half_w_px, img_w)
                y_segs = wrap_intervals(ay_f - half_h_px, ay_f + half_h_px, img_h)

                for xs1, xs2 in x_segs:
                    for ys1, ys2 in y_segs:
                        cv2.rectangle(img, (xs1, ys1), (xs2, ys2), (0, 255, 0), 1)

        return img



class Agent:
    """Agent with continuous 2D position, impulse-based physics, and a 2D camera perception."""

    def __init__(self, world, start_pos=(10.0, 10.0),
                 camera_w=64, camera_h=64,
                 fov_w=20.0, fov_h=20.0):
        """Initialize agent.

        Args:
            world: The World instance.
            start_pos: Continuous (x, y) starting position in world units.
            camera_w: Horizontal resolution of the camera image in pixels.
            camera_h: Vertical resolution of the camera image in pixels.
            fov_w: Field of view width in world units.
            fov_h: Field of view height in world units.
        """
        self.world = world
        self.position = list(start_pos)   # Continuous [x, y]
        self.velocity = [0.0, 0.0]        # [vx, vy]

        # Camera parameters
        self.camera_w = camera_w
        self.camera_h = camera_h
        self.fov_w = fov_w
        self.fov_h = fov_h

        # Physics parameters  (same design as 1D continuous case)
        self.mass = 1.0
        self.friction_coeff = 5.0
        self.kick_force = 50.0

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def apply_force(self, fx, fy):
        """Apply an impulse force (instantaneous velocity change)."""
        self.velocity[0] += fx / self.mass
        self.velocity[1] += fy / self.mass

    def update(self, dt, restitution=0.45):
        """Advance physics by one time step dt with live-cell collision.

        Each axis is solved independently.  If the candidate position lands
        on a live (white) cell the velocity on that axis is reflected and
        damped by *restitution* and the agent does not move on that axis.
        """
        for i in range(2):
            friction_force = -self.friction_coeff * self.velocity[i]
            self.velocity[i] += (friction_force / self.mass) * dt

            dim = self.world.width if i == 0 else self.world.height
            new_coord = (self.position[i] + self.velocity[i] * dt) % dim

            # Sample the cell at the candidate position (other axis unchanged)
            cx = int(new_coord)    % self.world.width  if i == 0 else int(self.position[0]) % self.world.width
            cy = int(self.position[1]) % self.world.height if i == 0 else int(new_coord) % self.world.height

            if self.world.state[cy, cx] == 1:
                # Collision: bounce and damp, stay put on this axis
                self.velocity[i] = -self.velocity[i] * restitution
            else:
                self.position[i] = new_coord

            # Kill negligible velocity
            if abs(self.velocity[i]) < 0.01:
                self.velocity[i] = 0.0

    def kick(self, direction):
        """Apply an impulse kick in a cardinal direction."""
        f = self.kick_force
        kicks = {'up': (0, -f), 'down': (0, f), 'left': (-f, 0), 'right': (f, 0)}
        fx, fy = kicks.get(direction, (0, 0))
        self.apply_force(fx, fy)

    # ------------------------------------------------------------------
    # 2D Camera perception
    # ------------------------------------------------------------------

    def perceive(self, mode='bilinear'):
        """Project the world onto the camera image via orthographic sampling.

        Each camera pixel (px, py) maps to world coordinate:
            wx = cx + (px / camera_w - 0.5) * fov_w
            wy = cy + (py / camera_h - 0.5) * fov_h

        Args:
            mode: 'nearest'  - snap to closest cell (hard binary)
                  'bilinear' - blend 4 surrounding cells by fractional distance
                  'area'     - average a 4x4 sub-sample grid inside each pixel footprint

        Returns:
            Float NumPy array of shape (camera_h, camera_w), values in [0, 1].
        """
        cx, cy = self.position

        px = np.arange(self.camera_w)
        wx = cx + (px / self.camera_w - 0.5) * self.fov_w   # (camera_w,)
        py = np.arange(self.camera_h)
        wy = cy + (py / self.camera_h - 0.5) * self.fov_h   # (camera_h,)

        if mode == 'nearest':
            wx_idx = np.floor(wx).astype(int) % self.world.width
            wy_idx = np.floor(wy).astype(int) % self.world.height
            return self.world.state[np.ix_(wy_idx, wx_idx)].astype(np.float32)

        elif mode == 'bilinear':
            s = self.world.state.astype(np.float32)

            x0 = np.floor(wx).astype(int) % self.world.width
            x1 = (x0 + 1)                 % self.world.width
            fx = (wx - np.floor(wx)).astype(np.float32)      # (camera_w,)

            y0 = np.floor(wy).astype(int) % self.world.height
            y1 = (y0 + 1)                 % self.world.height
            fy = (wy - np.floor(wy)).astype(np.float32)      # (camera_h,)

            # Broadcast to (camera_h, camera_w)
            fx = fx[np.newaxis, :]   # (1, W)
            fy = fy[:, np.newaxis]   # (H, 1)

            c00 = s[np.ix_(y0, x0)]
            c10 = s[np.ix_(y0, x1)]
            c01 = s[np.ix_(y1, x0)]
            c11 = s[np.ix_(y1, x1)]

            return (c00 * (1 - fx) * (1 - fy) +
                    c10 *      fx  * (1 - fy) +
                    c01 * (1 - fx) *      fy  +
                    c11 *      fx  *      fy)

        elif mode == 'area':
            # 4x4 stratified sub-samples per pixel (centred offsets in [-0.5, 0.5])
            n = 4
            offsets = (np.arange(n) + 0.5) / n - 0.5
            step_x = self.fov_w / self.camera_w
            step_y = self.fov_h / self.camera_h

            acc = np.zeros((self.camera_h, self.camera_w), dtype=np.float32)
            for dy in offsets:
                for dx in offsets:
                    xi = np.floor(wx + dx * step_x).astype(int) % self.world.width
                    yi = np.floor(wy + dy * step_y).astype(int) % self.world.height
                    acc += self.world.state[np.ix_(yi, xi)]
            return acc / (n * n)

        else:
            raise ValueError(f"Unknown perception mode: {mode!r}")

    def get_sensory_input_opencv(self, display_scale=4, mode='bilinear',
                                   camera_float=None):
        """Return a BGR OpenCV image of the camera output.

        Args:
            display_scale: Integer upscale factor for display.
            mode: Perception mode — 'nearest', 'bilinear', or 'area'.
            camera_float: Optional precomputed float32 (H, W) array from perceive().
                          If None, perceive() is called internally.
        """
        if camera_float is None:
            camera_float = self.perceive(mode=mode)

        # Float [0,1] -> uint8 greyscale BGR
        grey = (camera_float * 255).astype(np.uint8)
        img = np.stack([grey, grey, grey], axis=2)

        # Mark the agent centre in red
        cv2.circle(img, (self.camera_w // 2, self.camera_h // 2), 2, (0, 0, 255), -1)

        # Scale up for display (nearest-neighbour keeps pixel grid crisp)
        img = cv2.resize(img,
                         (self.camera_w * display_scale, self.camera_h * display_scale),
                         interpolation=cv2.INTER_NEAREST)
        return img

    # ------------------------------------------------------------------
    # Cell editing helpers (still useful for seeding patterns)
    # ------------------------------------------------------------------

    def toggle_cell(self):
        """Flip the cell under the agent's position."""
        x = int(self.position[0]) % self.world.width
        y = int(self.position[1]) % self.world.height
        self.world.set_value(x, y, 1 - self.world.get_value(x, y))

    def set_cell(self, value):
        """Set the cell under the agent's position to value."""
        x = int(self.position[0]) % self.world.width
        y = int(self.position[1]) % self.world.height
        self.world.set_value(x, y, value)


# ---------------------------------------------------------------------------
# Computer vision: Harris corner detection + frame-to-frame tracking
# ---------------------------------------------------------------------------

@dataclass
class Feature:
    """A Harris corner feature detected in the camera image."""
    id: int             # Unique persistent ID
    px: float           # Camera pixel column (original camera resolution)
    py: float           # Camera pixel row
    strength: float     # Harris corner response
    world_x: float      # Back-projected world x coordinate (toroidal)
    world_y: float      # Back-projected world y coordinate
    age: int = 1        # Frames this feature has been continuously matched


class FeatureTracker:
    """Detects Harris corners in the camera image and tracks them across frames.

    Because the world is dynamic (Game of Life), features appear and disappear
    frame-to-frame.  Tracking uses nearest-neighbour matching in camera pixel
    space: if a detection falls within *match_radius* pixels of a prior feature
    the old ID and age are inherited; otherwise a new ID is minted.
    """

    def __init__(self,
                 block_size: int    = 3,
                 ksize: int         = 3,
                 harris_k: float    = 0.04,
                 threshold: float   = 0.005,
                 max_features: int  = 200,
                 match_radius: float = 10.0,
                 min_distance: float = 8.0,
                 blur_sigma: float   = 1.5):
        """
        Args:
            block_size:    Neighbourhood size for the Harris matrix (3 best for sharp edges).
            ksize:         Sobel aperture for gradient computation.
            harris_k:      Harris sensitivity parameter (0.04 standard).
            threshold:     Fraction of the max response used as a cutoff.
            max_features:  Keep only the top-N strongest corners per frame.
            match_radius:  Max pixel distance to link a detection to a prior track.
            min_distance:  Minimum pixel distance between any two kept corners (NMS).
            blur_sigma:    Sigma of Gaussian pre-blur applied before Harris.
                           Non-zero smooths near-binary images so Sobel works reliably.
        """
        self.block_size   = block_size
        self.ksize        = ksize
        self.harris_k     = harris_k
        self.threshold    = threshold
        self.max_features = max_features
        self.match_radius = match_radius
        self.min_distance = min_distance
        self.blur_sigma   = blur_sigma

        self.features: list = []   # list[Feature]
        self._next_id: int  = 0

    # ------------------------------------------------------------------

    def _backproject(self, px: float, py: float, agent) -> tuple:
        """Convert a camera-space pixel to a (wrapped) world coordinate."""
        wx = agent.position[0] + (px / agent.camera_w - 0.5) * agent.fov_w
        wy = agent.position[1] + (py / agent.camera_h - 0.5) * agent.fov_h
        return wx % agent.world.width, wy % agent.world.height

    def update(self, camera_float: np.ndarray, agent) -> list:
        """Run Harris detection and match against the previous frame's features.

        Args:
            camera_float: Float32 array (H, W) in [0, 1] from agent.perceive().
            agent:        The Agent instance (used for back-projection).

        Returns:
            Updated list of Feature objects (also stored in self.features).
        """
        img_f = (camera_float * 255).astype(np.float32)

        # Pre-blur: near-binary images have near-impulse gradients;
        # a small Gaussian gives Sobel smooth, differentiable edges.
        # if self.blur_sigma > 0:
        #     ks = int(self.blur_sigma * 4) | 1  # odd kernel size ~4-sigma wide
        #     img_f = cv2.GaussianBlur(img_f, (ks, ks), self.blur_sigma)

        response = cv2.cornerHarris(img_f, self.block_size, self.ksize, self.harris_k)
        response = cv2.dilate(response, None)   # local-max suppression

        max_r = response.max()
        if max_r <= 0:
            self.features = []
            return self.features

        ys, xs = np.where(response > self.threshold * max_r)
        strengths = response[ys, xs]

        # Sort by strength descending, then apply greedy min-distance NMS
        order = np.argsort(strengths)[::-1]
        xs, ys, strengths = xs[order], ys[order], strengths[order]

        kept_xs, kept_ys, kept_s = [], [], []
        for i in range(len(xs)):
            cx_i, cy_i = float(xs[i]), float(ys[i])
            too_close = any(
                (cx_i - kx) ** 2 + (cy_i - ky) ** 2 < self.min_distance ** 2
                for kx, ky in zip(kept_xs, kept_ys)
            )
            if not too_close:
                kept_xs.append(cx_i)
                kept_ys.append(cy_i)
                kept_s.append(strengths[i])
            if len(kept_xs) >= self.max_features:
                break

        xs = np.array(kept_xs)
        ys = np.array(kept_ys)
        strengths = np.array(kept_s)

        # Build prev-feature lookup
        prev_by_id = {f.id: f for f in self.features}
        prev_pts   = (np.array([[f.px, f.py] for f in self.features], dtype=np.float32)
                      if self.features else np.empty((0, 2), dtype=np.float32))

        new_features = []
        used_ids     = set()

        for i in range(len(xs)):
            dpx, dpy = float(xs[i]), float(ys[i])
            s        = float(strengths[i])
            wx, wy   = self._backproject(dpx, dpy, agent)

            # Try to match to a previous feature
            fid = None
            if len(prev_pts) > 0:
                dists = np.hypot(prev_pts[:, 0] - dpx, prev_pts[:, 1] - dpy)
                best  = int(np.argmin(dists))
                if dists[best] <= self.match_radius:
                    cid = self.features[best].id
                    if cid not in used_ids:
                        fid = cid
                        used_ids.add(fid)

            if fid is None:
                fid = self._next_id
                self._next_id += 1
                age = 1
            else:
                age = prev_by_id[fid].age + 1

            new_features.append(Feature(
                id=fid, px=dpx, py=dpy, strength=s,
                world_x=wx, world_y=wy, age=age
            ))

        self.features = new_features
        return self.features

    def draw(self, img: np.ndarray, scale: int = 1) -> np.ndarray:
        """Overlay detected features on a copy of *img*.

        Corners are drawn as small circles whose colour encodes age:
        cyan = just appeared, yellow = 30+ frames old.
        The feature ID and world position are printed next to each corner.

        Args:
            img:   BGR image at *display_scale* resolution.
            scale: The display_scale factor so pixel coords line up.

        Returns:
            Annotated copy of *img*.
        """
        out = img.copy()
        for f in self.features:
            cx = int(f.px * scale)
            cy = int(f.py * scale)
            # Colour: cyan (young) → yellow (old)
            # t   = min(f.age / 30.0, 1.0)
            # col = (int(255 * (1 - t)), int(255 * t), 255)
            col = (0, 0, 255) 
            radius = max(3, scale)
            cv2.circle(out, (cx, cy), radius, col, 1)
            # label = f"{f.id}({f.age})"
            # cv2.putText(out, label, (cx + radius + 1, cy - radius),
            #             cv2.FONT_HERSHEY_PLAIN, 0.65, col, 1)
        return out


# ---------------------------------------------------------------------------
# Motion model: rolling trajectory buffer + live x(t)/y(t) plots
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    """Stores a fixed-length rolling window of agent positions and renders
    them as two stacked time-series plots: x(t) on top, y(t) on bottom."""

    def __init__(self,
                 max_len: int = 600,
                 img_w: int   = 600,
                 plot_h: int  = 180,
                 padding: int = 36):
        """
        Args:
            max_len:  Maximum number of samples in the rolling buffer.
            img_w:    Width of the output image in pixels.
            plot_h:   Height of each individual plot in pixels.
            padding:  Pixel margin around each plot for axes / labels.
        """
        self.max_len = max_len
        self.img_w   = img_w
        self.plot_h  = plot_h
        self.padding = padding

        self.times:  list = []   # float  – wall-clock seconds from start
        self.xs:     list = []   # float  – world x
        self.ys:     list = []   # float  – world y
        self._t0: float  = None

    def record(self, x: float, y: float, t: float) -> None:
        """Append a new (x, y) sample at time t."""
        if self._t0 is None:
            self._t0 = t
        self.times.append(t - self._t0)
        self.xs.append(x)
        self.ys.append(y)
        if len(self.times) > self.max_len:
            self.times.pop(0)
            self.xs.pop(0)
            self.ys.pop(0)

    def _draw_plot(self,
                   canvas: np.ndarray,
                   y_offset: int,
                   values: list,
                   world_max: float,
                   label: str,
                   colour: tuple) -> None:
        """Draw one time-series plot into *canvas* starting at row *y_offset*."""
        p   = self.padding
        w   = self.img_w
        h   = self.plot_h
        n   = len(values)

        # Plot area corners
        x0, y0 = p, y_offset + p
        x1, y1 = w - p, y_offset + h - p
        pw, ph = x1 - x0, y1 - y0   # plot width / height in pixels

        # Background
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (25, 25, 25), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 80, 80),  1)

        # Horizontal grid lines (4 divisions)
        for k in range(5):
            gy = y0 + int(k / 4 * ph)
            cv2.line(canvas, (x0, gy), (x1, gy), (50, 50, 50), 1)
            val = world_max * (1 - k / 4)
            cv2.putText(canvas, f"{val:.0f}",
                        (2, gy + 4), cv2.FONT_HERSHEY_PLAIN, 0.75, (120, 120, 120), 1)

        # Axis label
        cv2.putText(canvas, label,
                    (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        if n < 2:
            return

        # Time axis span
        t_span = max(self.times[-1] - self.times[0], 1e-6)

        # Draw polyline
        pts = []
        for i, v in enumerate(values):
            tx = x0 + int((self.times[i] - self.times[0]) / t_span * pw)
            ty = y1 - int(np.clip(v / world_max, 0, 1) * ph)
            pts.append((tx, ty))

        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], colour, 1)

        # Current-value marker
        cv2.circle(canvas, pts[-1], 3, (255, 255, 255), -1)
        cv2.putText(canvas, f"{values[-1]:.1f}",
                    (pts[-1][0] + 5, pts[-1][1] - 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (255, 255, 255), 1)

    def draw(self, world_w: float, world_h: float) -> np.ndarray:
        """Return a BGR image with two stacked plots: x(t) and y(t)."""
        total_h = self.plot_h * 2 + self.padding
        canvas  = np.zeros((total_h, self.img_w, 3), dtype=np.uint8)

        # Time axis label at bottom
        t_label = f"t = {self.times[-1]:.1f} s" if self.times else "t"
        cv2.putText(canvas, t_label,
                    (self.img_w // 2 - 30, total_h - 6),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (160, 160, 160), 1)

        self._draw_plot(canvas, 0,               self.xs, world_w,
                        "x(t)", (100, 220, 100))
        self._draw_plot(canvas, self.plot_h,     self.ys, world_h,
                        "y(t)", (100, 180, 255))
        return canvas


def main():
    """Main simulation loop."""
    import time

    # Initialize world and agent
    world = World(width=100, height=100)
    agent = Agent(world,
                  start_pos=(50.0, 50.0),
                  camera_w=256, camera_h=256,
                  fov_w=30.0, fov_h=30.0)
    tracker = FeatureTracker(
        block_size=3, ksize=3, harris_k=0.04,
        threshold=0.005, max_features=200,
        match_radius=10.0, min_distance=8.0, blur_sigma=1.5
    )
    trajectory = TrajectoryBuffer(max_len=600, img_w=620, plot_h=180)
    DISPLAY_SCALE = 3

    # Create OpenCV windows
    cv2.namedWindow('2D World State', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Camera View', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Trajectory', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('2D World State', 50, 50)
    cv2.moveWindow('Camera View', 800, 50)
    cv2.moveWindow('Trajectory', 800, 830)

    print("=" * 55)
    print("2D Agent-World Simulation  (continuous + camera + CV)")
    print("=" * 55)
    print("Controls (focus an OpenCV window):")
    print("  'w' / 's' / 'a' / 'd' - Kick Up / Down / Left / Right")
    print("  'q' or ESC             - Quit")
    print("  SPACE                  - Pause / Resume Game of Life")
    print("  '+' / '-'              - Speed up / slow down GoL")
    print("  'e'                    - Toggle cell under agent")
    print("  'z'                    - Kill cell under agent")
    print("  'x'                    - Revive cell under agent")
    print("  'p'                    - Cycle perception mode (nearest/bilinear/area)")
    print("  'f'                    - Toggle Harris corner overlay")
    print("  'c'                    - Clear trajectory buffer")
    print("=" * 55)
    print("Physics: impulse kicks with friction")
    print(f"Camera:  {agent.camera_w}x{agent.camera_h}px  FOV {agent.fov_w}x{agent.fov_h} world units")
    print("CV:      Harris corners, nearest-neighbour tracking")
    print("=" * 55)

    # Warm-up: run GoL steps without rendering
    WARMUP_STEPS = 1100
    print(f"Running {WARMUP_STEPS} warm-up steps...")
    for _ in range(WARMUP_STEPS):
        world.step()
    print(f"Done. Starting visualization at step {world.time_step}.")

    running = True
    paused = False
    frames_per_step = 1
    frame_count = 0
    perception_modes = ['bilinear', 'area', 'nearest']
    perception_mode_idx = 2
    show_features = True

    dt = 0.01           # Physics time step (seconds)
    last_time = time.time()

    while running:
        # ---- Physics update (real-time) --------------------------------
        now = time.time()
        elapsed = now - last_time
        last_time = now
        accumulated = elapsed
        while accumulated >= dt:
            agent.update(dt)
            accumulated -= dt

        # ---- GoL advance -----------------------------------------------
        if not paused:
            frame_count += 1
            if frame_count >= frames_per_step:
                world.step()
                frame_count = 0

        # ---- Record trajectory -----------------------------------------
        trajectory.record(agent.position[0], agent.position[1], now)

        # ---- Visualise -------------------------------------------------
        world_img = world.visualize_opencv(
            agent_pos=agent.position,
            cell_size=7,
            fov=(agent.fov_w, agent.fov_h)
        )

        # Single perceive() call shared by display and tracker
        mode = perception_modes[perception_mode_idx]
        camera_raw = agent.perceive(mode=mode)
        tracker.update(camera_raw, agent)

        sensory_img = agent.get_sensory_input_opencv(
            display_scale=DISPLAY_SCALE, camera_float=camera_raw)

        if show_features:
            sensory_img = tracker.draw(sensory_img, scale=DISPLAY_SCALE)

        # HUD
        status = "PAUSED" if paused else f"step {world.time_step}"
        px, py = agent.position
        vx, vy = agent.velocity
        cv2.putText(world_img,
                    f"Pos: ({px:.1f}, {py:.1f})  Vel: ({vx:.1f}, {vy:.1f})  |  {status}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(world_img,
                    f"GoL speed: 1 step / {frames_per_step} frames",
                    (6, world_img.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 100), 1)

        cv2.putText(sensory_img,
                    f"Camera {agent.camera_w}x{agent.camera_h}  FOV {agent.fov_w}x{agent.fov_h} wu"
                    f"  [{mode}]  corners:{len(tracker.features)}  {'[F]' if show_features else ''}",
                    (4, sensory_img.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 200, 100), 1)

        cv2.imshow('2D World State', world_img)
        cv2.imshow('Camera View', sensory_img)
        cv2.imshow('Trajectory', trajectory.draw(world.width, world.height))

        # ---- Input handling --------------------------------------------
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q') or key == 27:
            running = False
            print("\nExiting...")
        elif key == ord('w'):
            agent.kick('up')
            print(f"↑ Kick UP    | Pos: ({px:.2f}, {py:.2f}) | Vel: ({vx:.2f}, {vy:.2f})")
        elif key == ord('s'):
            agent.kick('down')
            print(f"↓ Kick DOWN  | Pos: ({px:.2f}, {py:.2f}) | Vel: ({vx:.2f}, {vy:.2f})")
        elif key == ord('a'):
            agent.kick('left')
            print(f"← Kick LEFT  | Pos: ({px:.2f}, {py:.2f}) | Vel: ({vx:.2f}, {vy:.2f})")
        elif key == ord('d'):
            agent.kick('right')
            print(f"→ Kick RIGHT | Pos: ({px:.2f}, {py:.2f}) | Vel: ({vx:.2f}, {vy:.2f})")
        elif key == ord(' '):
            paused = not paused
        elif key == ord('+') or key == ord('='):
            frames_per_step = max(1, frames_per_step - 1)
        elif key == ord('-'):
            frames_per_step = min(30, frames_per_step + 1)
        elif key == ord('e'):
            agent.toggle_cell()
        elif key == ord('z'):
            agent.set_cell(0)
        elif key == ord('x'):
            agent.set_cell(1)
        elif key == ord('p'):
            perception_mode_idx = (perception_mode_idx + 1) % len(perception_modes)
            print(f"Perception mode: {perception_modes[perception_mode_idx]}")
        elif key == ord('f'):
            show_features = not show_features
            print(f"Feature overlay: {'ON' if show_features else 'OFF'}  ({len(tracker.features)} corners)")
        elif key == ord('c'):
            trajectory.times.clear()
            trajectory.xs.clear()
            trajectory.ys.clear()
            trajectory._t0 = None
            print("Trajectory cleared.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

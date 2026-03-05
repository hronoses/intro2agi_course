import numpy as np
import cv2

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

    def visualize_opencv(self, agent_pos=None, cell_size=30):
        """Create an OpenCV image of the world state (fully vectorised)."""
        # Build a (height x width x 3) colour map in one shot
        colour_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        colour_map[self.state == 1] = (255, 255, 255)   # live  -> white
        if agent_pos is not None:
            colour_map[agent_pos[1], agent_pos[0]] = (0, 0, 255)  # agent -> red

        # Scale up to pixel resolution via np.repeat (no Python loops)
        img = np.repeat(np.repeat(colour_map, cell_size, axis=0), cell_size, axis=1)

        # Draw grid lines with single array writes
        for i in range(self.width):
            img[:, i * cell_size] = (50, 50, 50)
        for i in range(self.height):
            img[i * cell_size, :] = (50, 50, 50)

        return img



class Agent:
    """Agent that can perceive and move in the 2D world."""
    
    def __init__(self, world, start_pos=[0, 0], perception_radius=1):
        """Initialize agent with world, starting position [x, y], and perception radius."""
        self.world = world
        self.position = list(start_pos) # Mutable list [x, y]
        self.perception_radius = perception_radius
    
    def perceive(self):
        """Get local perception of the world around agent's position."""
        r = self.perception_radius
        x, y = self.position
        # Build wrapped index arrays and extract the patch in one numpy call
        ys = np.arange(y - r, y + r + 1) % self.world.height
        xs = np.arange(x - r, x + r + 1) % self.world.width
        return self.world.state[np.ix_(ys, xs)]

    def get_sensory_input_opencv(self, cell_size=30):
        """Get OpenCV visualization of sensory input (fully vectorised)."""
        perception = self.perceive()
        size = perception.shape[0]
        r = self.perception_radius

        colour_map = np.full((size, size, 3), (50, 50, 50), dtype=np.uint8)  # default dark
        colour_map[perception == 1] = (255, 255, 255)  # live  -> white
        colour_map[r, r] = (0, 0, 255)                 # agent -> red

        img = np.repeat(np.repeat(colour_map, cell_size, axis=0), cell_size, axis=1)

        for i in range(size):
            img[:, i * cell_size] = (100, 100, 100)
            img[i * cell_size, :] = (100, 100, 100)

        return img

    def move(self, direction):
        """Move agent in a given direction."""
        dx, dy = 0, 0
        if direction == 'up':
            dy = -1
        elif direction == 'down':
            dy = 1
        elif direction == 'left':
            dx = -1
        elif direction == 'right':
            dx = 1
            
        # Update position with wrapping
        new_x = (self.position[0] + dx) % self.world.width
        new_y = (self.position[1] + dy) % self.world.height
        self.position = [new_x, new_y]

    def toggle_cell(self):
        """Flip the cell value at the agent's current position (0->1 or 1->0)."""
        x, y = self.position
        current = self.world.get_value(x, y)
        self.world.set_value(x, y, 1 - current)

    def set_cell(self, value):
        """Set the cell at the agent's current position to a specific value (0 or 1)."""
        self.world.set_value(self.position[0], self.position[1], value)


class SaliencyMap:
    """
    Builds two saliency signals from the agent's perception patch each frame.

    Motion saliency  – highlights cells that *changed* (births / deaths).
                       Uses an exponential moving average (EMA) of the
                       absolute frame-to-frame difference so recent change
                       decays smoothly rather than vanishing in one step.

    Static saliency  – highlights *where* live cells are, irrespective of
                       motion.  A Gaussian density blur turns isolated cells
                       into soft blobs and clusters into bright peaks, making
                       structure visible at a glance.
    """

    def __init__(self, perception_size, motion_decay=0.6, static_sigma=1.2):
        """
        Parameters
        ----------
        perception_size : int   side length of the square perception patch (2r+1)
        motion_decay    : float EMA weight kept from the previous motion map (0-1).
                                Higher = longer memory, slower fade.
        static_sigma    : float Gaussian sigma for the static density blob.
        """
        self.size = perception_size
        self.motion_decay = motion_decay
        self.static_sigma = static_sigma

        self._prev_frame: np.ndarray | None = None
        self._motion_ema = np.zeros((perception_size, perception_size), dtype=np.float32)
        self._static_map = np.zeros((perception_size, perception_size), dtype=np.float32)

    def update(self, perception: np.ndarray):
        """Ingest the latest perception patch and recompute both maps."""
        frame = perception.astype(np.float32)

        # --- motion saliency ---
        if self._prev_frame is not None:
            diff = np.abs(frame - self._prev_frame)          # 1 where cell flipped
            self._motion_ema = (self.motion_decay * self._motion_ema
                                + (1.0 - self.motion_decay) * diff)
        self._prev_frame = frame.copy()

        # --- static saliency ---
        # Gaussian blur turns each live cell into a soft blob; dense clusters
        # accumulate higher values.
        k = max(3, int(self.static_sigma * 6) | 1)          # odd kernel size
        self._static_map = cv2.GaussianBlur(frame, (k, k), self.static_sigma)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_motion_saliency(self) -> np.ndarray:
        """Return the current motion saliency map (float32, 0-1 range approx)."""
        return self._motion_ema.copy()

    def get_static_saliency(self) -> np.ndarray:
        """Return the current static (density) saliency map (float32)."""
        return self._static_map.copy()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def _to_heatmap(arr: np.ndarray, colormap: int) -> np.ndarray:
        """Normalise a float array to [0,255] and apply an OpenCV colormap."""
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            normalised = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            normalised = np.zeros_like(arr, dtype=np.uint8)
        return cv2.applyColorMap(normalised, colormap)

    def visualize(self, cell_size: int = 20) -> np.ndarray:
        """
        Render both saliency maps as a side-by-side BGR image.

        Left  panel : Motion saliency  – HOT colormap (black → red → white).
        Right panel : Static saliency  – WINTER colormap (blue → green).

        Returns the combined (H x 2W x 3) uint8 image.
        """
        motion_col = self._to_heatmap(self._motion_ema, cv2.COLORMAP_HOT)
        static_col = self._to_heatmap(self._static_map, cv2.COLORMAP_WINTER)

        def scale_up(img: np.ndarray) -> np.ndarray:
            return np.repeat(np.repeat(img, cell_size, axis=0), cell_size, axis=1)

        motion_big = scale_up(motion_col)
        static_big = scale_up(static_col)

        # Label bars above each panel
        bar_h = 22
        w = motion_big.shape[1]

        def label_bar(text: str, text_color) -> np.ndarray:
            bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
            cv2.putText(bar, text, (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            return bar

        motion_panel = np.vstack([label_bar("Motion Saliency", (160, 160, 255)), motion_big])
        static_panel = np.vstack([label_bar("Static Saliency", (160, 255, 160)), static_big])

        return np.hstack([motion_panel, static_panel])


def main():
    """Main simulation loop."""
    # Initialize world and agent
    world = World(width=50, height=50)
    agent = Agent(world, start_pos=[5, 5], perception_radius=9)
    
    # Create OpenCV windows
    cv2.namedWindow('2D World State', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Sensory Input', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Saliency Map', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('Sensory Input', 660, 50)
    cv2.moveWindow('Saliency Map', 660, 450)
    
    print("=" * 50)
    print("2D Agent-World Simulation")
    print("=" * 50)
    print("Controls:")
    print("  'w' - Move Up")
    print("  's' - Move Down")
    print("  'a' - Move Left")
    print("  'd' - Move Right")
    print("  'q' or ESC - Quit")
    print("  SPACE     - Pause / Resume Game of Life")
    print("  '+' / '-' - Speed up / slow down simulation")
    print("  'e'       - Toggle cell at agent position (0 <-> 1)")
    print("  'z'       - Kill cell at agent position (set to 0)")
    print("  'x'       - Revive cell at agent position (set to 1)")
    print("=" * 50)

    # Saliency processor
    perception_size = 2 * agent.perception_radius + 1
    saliency = SaliencyMap(perception_size, motion_decay=0.6, static_sigma=1.2)

    # Warm-up: run GoL steps without rendering
    WARMUP_STEPS = 400
    print(f"Running {WARMUP_STEPS} warm-up steps...")
    for _ in range(WARMUP_STEPS):
        world.step()
    print(f"Done. Starting visualization at step {world.time_step}.")

    running = True
    paused = False
    # How many display frames between world steps (lower = faster)
    frames_per_step = 1
    frame_count = 0

    while running:
        # Advance Game of Life
        if not paused:
            frame_count += 1
            if frame_count >= frames_per_step:
                world.step()
                frame_count = 0

        # Visualize World
        world_img = world.visualize_opencv(agent.position, cell_size=30)

        # Visualize Sensory Input
        sensory_img = agent.get_sensory_input_opencv(cell_size=30)

        # Update and visualize saliency
        saliency.update(agent.perceive())
        saliency_img = saliency.visualize(cell_size=20)

        # HUD overlay on World View
        status = "PAUSED" if paused else f"step {world.time_step}"
        cv2.putText(world_img, f"Pos: {agent.position}  |  {status}",
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(world_img, f"Speed: 1 step / {frames_per_step} frames",
                   (10, world_img.shape[0] - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 100), 1)

        cv2.imshow('2D World State', world_img)
        cv2.imshow('Sensory Input', sensory_img)
        cv2.imshow('Saliency Map', saliency_img)

        # Handle input
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q') or key == 27:
            running = False
        elif key == ord('w'):
            agent.move('up')
        elif key == ord('s'):
            agent.move('down')
        elif key == ord('a'):
            agent.move('left')
        elif key == ord('d'):
            agent.move('right')
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

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

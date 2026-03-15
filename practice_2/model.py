import numpy as np
import cv2

from objects import ObjectTracker


_UNCERTAIN = (128, 128, 128)  # grey – never observed
_DEAD      = (20,  20,  20)   # dark – known dead
_ALIVE     = (255, 255, 255)  # white – known alive
_AGENT     = (0,   0,   255)  # red – agent position
_BBOX_CLR  = (0,   200, 80)   # green – object bounding box


class WorldModel:
    """
    Agent's internal map of the world built incrementally from observations.

    Cells that have never been inside the agent's perception patch are
    'uncertain' and rendered grey.  Cells that have been observed carry
    their last known value (alive = white, dead = dark).

    When a SaliencyMap is attached, objects (connected clusters of live cells)
    are detected each frame and stored with world-coordinate bounding boxes.
    """

    def __init__(self, width: int, height: int, saliency=None):
        self.width   = width
        self.height  = height
        self.saliency = saliency
        self._values  = np.zeros((height, width), dtype=np.uint8)
        self._known   = np.zeros((height, width), dtype=bool)
        self._tracker = ObjectTracker(width, height)

    @property
    def objects(self) -> dict:
        return self._tracker.objects

    def update(self, perception: np.ndarray, agent_pos: list,
               perception_radius: int, world_time: int = 0):
        """Stamp the latest perception patch and, if saliency is set, detect objects."""
        r = perception_radius
        x, y = agent_pos
        ys = np.arange(y - r, y + r + 1) % self.height
        xs = np.arange(x - r, x + r + 1) % self.width
        idx = np.ix_(ys, xs)
        self._values[idx] = perception
        self._known[idx]  = True

        if self.saliency is not None:
            self.saliency.update(perception)
            blobs = self.saliency.segment()
            self._tracker.update(blobs, agent_pos, r, world_time)

    @property
    def coverage(self) -> float:
        return self._known.mean()

    def visualize_opencv(self, agent_pos=None, cell_size=10) -> np.ndarray:
        colour_map = np.full((self.height, self.width, 3), _UNCERTAIN, dtype=np.uint8)
        colour_map[self._known & (self._values == 0)] = _DEAD
        colour_map[self._known & (self._values == 1)] = _ALIVE
        if agent_pos is not None:
            colour_map[agent_pos[1] % self.height, agent_pos[0] % self.width] = _AGENT

        img = np.repeat(np.repeat(colour_map, cell_size, axis=0), cell_size, axis=1)
        img[:, ::cell_size] = (50, 50, 50)
        img[::cell_size, :] = (50, 50, 50)

        # Draw object bounding boxes and IDs
        for obj in self._tracker.objects.values():
            bx, by, bw, bh = obj.bbox
            x1 = bx * cell_size
            y1 = by * cell_size
            x2 = ((bx + bw) % self.width)  * cell_size
            y2 = ((by + bh) % self.height) * cell_size
            cv2.rectangle(img, (x1, y1), (x2, y2), _BBOX_CLR, 1)
            cv2.putText(img, f'{obj.id} a={obj.area}',
                        (x1 + 2, y1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, _BBOX_CLR, 1, cv2.LINE_AA)

        return img

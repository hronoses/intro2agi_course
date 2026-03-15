from dataclasses import dataclass, field
import numpy as np


@dataclass
class WorldObject:
    id:            int
    centroid:      np.ndarray   # (x, y) float, world coordinates
    area:          int
    bbox:          tuple        # (x, y, w, h) world coordinates
    motion_score:  float
    first_seen:    int
    last_seen:     int


class ObjectTracker:
    """
    Maintains a set of WorldObjects across frames.

    Each call to update():
      1. Converts blob centroids from local patch coords → world coords.
      2. Matches blobs to existing objects by nearest-centroid (toroidal).
      3. Spawns new objects for unmatched blobs.
      4. Evicts objects not seen for more than max_age steps.
    """

    def __init__(self, world_width: int, world_height: int,
                 match_radius: float = 5.0, max_age: int = 5):
        self.world_width   = world_width
        self.world_height  = world_height
        self.match_radius  = match_radius
        self.max_age       = max_age
        self._objects: dict[int, WorldObject] = {}
        self._next_id = 0

    @property
    def objects(self) -> dict[int, WorldObject]:
        return self._objects

    def _to_world(self, cx_local: float, cy_local: float,
                  agent_pos: list, radius: int) -> np.ndarray:
        """Convert a patch-local (x, y) to world coordinates (with wrapping)."""
        wx = (agent_pos[0] - radius + cx_local) % self.world_width
        wy = (agent_pos[1] - radius + cy_local) % self.world_height
        return np.array([wx, wy], dtype=np.float32)

    def _toroidal_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        dx = min(dx, self.world_width  - dx)
        dy = min(dy, self.world_height - dy)
        return float(np.sqrt(dx * dx + dy * dy))

    def _blob_to_world_bbox(self, bbox_local: tuple, agent_pos: list,
                            radius: int) -> tuple:
        bx, by, bw, bh = bbox_local
        wx = int((agent_pos[0] - radius + bx) % self.world_width)
        wy = int((agent_pos[1] - radius + by) % self.world_height)
        return (wx, wy, bw, bh)

    def update(self, blobs: list[dict], agent_pos: list,
               radius: int, world_time: int) -> dict[int, WorldObject]:

        # Convert blobs to world centroids
        new_centroids = [
            self._to_world(*b['centroid_local'], agent_pos, radius)
            for b in blobs
        ]

        matched_ids   = set()
        matched_blobs = set()

        # Greedy nearest-centroid matching
        for obj_id, obj in self._objects.items():
            best_blob, best_dist = None, self.match_radius
            for i, centroid in enumerate(new_centroids):
                if i in matched_blobs:
                    continue
                d = self._toroidal_dist(obj.centroid, centroid)
                if d < best_dist:
                    best_dist, best_blob = d, i
            if best_blob is not None:
                b = blobs[best_blob]
                obj.centroid     = new_centroids[best_blob]
                obj.area         = b['area']
                obj.bbox         = self._blob_to_world_bbox(b['bbox_local'], agent_pos, radius)
                obj.motion_score = b['motion_score']
                obj.last_seen    = world_time
                matched_ids.add(obj_id)
                matched_blobs.add(best_blob)

        # Spawn new objects for unmatched blobs
        for i, b in enumerate(blobs):
            if i not in matched_blobs:
                obj = WorldObject(
                    id           = self._next_id,
                    centroid     = new_centroids[i],
                    area         = b['area'],
                    bbox         = self._blob_to_world_bbox(b['bbox_local'], agent_pos, radius),
                    motion_score = b['motion_score'],
                    first_seen   = world_time,
                    last_seen    = world_time,
                )
                self._objects[self._next_id] = obj
                self._next_id += 1

        # Evict stale objects
        self._objects = {
            oid: obj for oid, obj in self._objects.items()
            if world_time - obj.last_seen <= self.max_age
        }

        return self._objects

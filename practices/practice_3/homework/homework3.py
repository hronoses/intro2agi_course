"""
Homework 3 — Sample Code
========================
1. Run the 2-D Game-of-Life cellular automaton for 1 500 steps and save
   the resulting world state (ground truth for later reconstruction).
2. Pick N random live-cell locations and record a full-360° range-sensor
   sweep at each one.
"""

import sys
import numpy as np
from dataclasses import dataclass

# Make sure the practice_3 package root is on the path when running from
# the homework/ sub-directory.
sys.path.insert(0, "..")

from world import World
from sensors import RangeSensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORLD_W    = 50        # grid columns
WORLD_H    = 50        # grid rows
N_STEPS    = 1500       # GoL steps before sampling
N_SAMPLES  = 30         # number of random live-cell locations to probe
OBS_RADIUS = 10.0       # maximum ray distance (world units)
N_RAYS     = 360        # one ray per degree → full rotation
RAY_STEP   = 0.01        # march step (keep ≤ 0.7)
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Measurement data structure
# ---------------------------------------------------------------------------

@dataclass
class RangeMeasurement:
    """A single full-rotation range sweep taken at one location."""
    x:         float          # sensor x position (world units)
    y:         float          # sensor y position (world units)
    angles_deg: np.ndarray    # shape (N_RAYS,) — ray angles in degrees [0, 360)
    ranges:    np.ndarray     # shape (N_RAYS,) — hit distances in world units


# ---------------------------------------------------------------------------
# Minimal stand-in so RangeSensor.read() has all the fields it needs.
# ---------------------------------------------------------------------------

class _SensorPose:
    """Lightweight agent-like object that holds pose + world reference."""

    def __init__(self, world: World, x: float, y: float, orientation: float = 0.0):
        self.world       = world
        self.position    = [x, y]
        self.orientation = orientation          # heading in radians


# ---------------------------------------------------------------------------
# Step 1 — run the automaton
# ---------------------------------------------------------------------------

np.random.seed(RANDOM_SEED)

world = World(WORLD_W, WORLD_H)

print(f"Running GoL for {N_STEPS} steps …")
for _ in range(N_STEPS):
    world.step()

# Ground-truth snapshot — copy so later edits to `world` don't corrupt it.
ground_truth: np.ndarray = world.state.copy()   # shape (H, W), dtype uint8

print(f"Done. Live cells after {N_STEPS} steps: {ground_truth.sum()}")


# ---------------------------------------------------------------------------
# Step 2 — sample N live-cell locations and record 360° range sweeps
# ---------------------------------------------------------------------------

# Indices of all empty (dead) cells (value == 0) — sensor placed in free space
# so rays travel outward and measure distances to live-cell obstacles.
empty_ys, empty_xs = np.where(ground_truth == 0)

if len(empty_xs) < N_SAMPLES:
    raise RuntimeError(
        f"Only {len(empty_xs)} empty cells available; "
        f"cannot sample {N_SAMPLES}. Reduce N_SAMPLES."
    )

chosen = np.random.choice(len(empty_xs), size=N_SAMPLES, replace=False)

# Full-rotation sensor: fov_half = π covers [heading−π, heading+π] = 360°.
sensor = RangeSensor(
    camera_w   = N_RAYS,
    obs_radius = OBS_RADIUS,
    ray_step   = RAY_STEP,
    fov_half   = np.pi,          # ← full 360°
    noise_std  = 0.0,
)

# True ray bearings: CW degrees from East in the y-down world frame.
# RangeSensor casts rays at linspace(orientation - π, orientation + π, N_RAYS).
# With orientation=0 this maps directly to a CW bearing (East=0°, South=90°, …).
_world_angles_rad = np.linspace(-np.pi, np.pi, N_RAYS)
angles_deg = (_world_angles_rad * 180.0 / np.pi) % 360.0

measurements: list[RangeMeasurement] = []

for idx in chosen:
    gx = float(empty_xs[idx])   # grid column  →  world x  (cell centre + 0.5)
    gy = float(empty_ys[idx])   # grid row     →  world y

    pose   = _SensorPose(world, x=gx + 0.5, y=gy + 0.5)
    ranges = sensor.read(pose)          # shape (N_RAYS,), float32

    measurements.append(RangeMeasurement(
        x          = pose.position[0],
        y          = pose.position[1],
        angles_deg = angles_deg.copy(),
        ranges     = ranges.copy(),
    ))

print(f"Collected {len(measurements)} range sweeps.")


# ---------------------------------------------------------------------------
# Quick sanity check — print a summary of the first measurement
# ---------------------------------------------------------------------------

m0 = measurements[0]
print(f"\nSample measurement #0:")
print(f"  location : ({m0.x:.1f}, {m0.y:.1f})")
print(f"  rays     : {len(m0.ranges)}")
print(f"  min range: {m0.ranges.min():.2f}")
print(f"  max range: {m0.ranges.max():.2f}")


# ---------------------------------------------------------------------------
# Visualisation — range vs angle for a few selected sweeps
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt

N_SHOW   = 4    # number of sweeps to plot
WIN_HALF = 20   # half-size of the world-window in grid cells
indices  = np.linspace(0, len(measurements) - 1, N_SHOW, dtype=int)

fig = plt.figure(figsize=(4 * N_SHOW, 8))

for col, i in enumerate(indices):
    m = measurements[i]

    # ---- row 0: polar range plot ----------------------------------------
    ax_polar = fig.add_subplot(2, N_SHOW, col + 1, projection="polar")
    theta = np.deg2rad(m.angles_deg)
    ax_polar.plot(theta, m.ranges, linewidth=0.8, color="steelblue")
    ax_polar.fill(theta, m.ranges, alpha=0.15, color="steelblue")
    ax_polar.set_title(f"#{i}  ({m.x:.0f}, {m.y:.0f})", fontsize=9, pad=10)
    ax_polar.set_theta_direction(-1)   # CW — matches y-down world frame
    ax_polar.set_rmax(OBS_RADIUS)
    ax_polar.set_rticks([OBS_RADIUS / 2, OBS_RADIUS])
    ax_polar.set_rlabel_position(45)
    ax_polar.grid(True, linewidth=0.4)

    # ---- row 1: world window around the agent ---------------------------
    ax_world = fig.add_subplot(2, N_SHOW, N_SHOW + col + 1)

    cx, cy = int(m.x), int(m.y)   # centre cell
    x0, x1 = cx - WIN_HALF, cx + WIN_HALF
    y0, y1 = cy - WIN_HALF, cy + WIN_HALF

    # Crop with wrapping (np.take handles out-of-bounds indices via modulo)
    rows = np.arange(y0, y1) % WORLD_H
    cols = np.arange(x0, x1) % WORLD_W
    window = ground_truth[np.ix_(rows, cols)]

    ax_world.imshow(window, cmap="binary", origin="upper",
                    extent=[x0, x1, y1, y0])
    ax_world.plot(m.x, m.y, "r+", markersize=12, markeredgewidth=1.5,
                  label="agent")
    # draw a circle showing the sensor's max range
    circle = plt.Circle((m.x, m.y), OBS_RADIUS, color="steelblue",
                         fill=False, linewidth=0.8, linestyle="--")
    ax_world.add_patch(circle)

    # Draw cardinal arrows so the polar plot can be matched to world directions.
    # Ray index k uses actual angle: orientation - π + k * (2π / (N_RAYS-1))
    # (orientation = 0 for all samples in this script)
    arrow_len = OBS_RADIUS * 0.38
    # k indices for cardinal directions (0°=East, 90°=South, 180°=West, 270°=North)
    for label_deg, k in [("0°", N_RAYS // 2), ("90°", 3 * N_RAYS // 4),
                          ("180°", 0), ("270°", N_RAYS // 4)]:
        angle = np.deg2rad(m.angles_deg[k])
        dx = np.cos(angle) * arrow_len
        dy = np.sin(angle) * arrow_len
        ax_world.annotate(
            "", xy=(m.x + dx, m.y + dy), xytext=(m.x, m.y),
            arrowprops=dict(arrowstyle="-|>", color="crimson",
                            lw=1.2, mutation_scale=8),
        )
        ax_world.text(m.x + dx * 1.28, m.y + dy * 1.28, label_deg,
                      fontsize=7, ha="center", va="center", color="crimson")

    ax_world.set_xlim(x0, x1)
    ax_world.set_ylim(y1, y0)   # y-axis: top=y0, bottom=y1 (image convention)
    ax_world.set_aspect("equal")
    ax_world.set_xlabel("x (cells)", fontsize=8)
    ax_world.set_ylabel("y (cells)", fontsize=8)
    ax_world.tick_params(labelsize=7)

fig.suptitle("Range sensor — full-rotation sweeps", fontsize=12)
plt.tight_layout()
plt.show()

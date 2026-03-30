import numpy as np


class World:
    """2D binary world represented as a binary matrix (Conway's Game of Life)."""

    def __init__(self, width=20, height=20):
        """Initialize world with random binary values."""
        self.width = width
        self.height = height
        self.state = np.random.randint(0, 2, size=(height, width)).astype(np.uint8)
        self.time_step = 0

    def get_value(self, x, y):
        """Get the value at a specific position (with wrapping)."""
        return self.state[y % self.height, x % self.width]

    def set_value(self, x, y, value):
        """Set the value at a specific position (with wrapping)."""
        self.state[y % self.height, x % self.width] = value

    def step(self):
        """Advance the world by one time step using Conway's Game of Life rules."""
        p = np.pad(self.state, 1, mode='wrap')
        neighbours = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                      p[1:-1, :-2]               + p[1:-1, 2:] +
                      p[2:,   :-2] + p[2:,  1:-1] + p[2:,  2:])
        survive = (self.state == 1) & ((neighbours == 2) | (neighbours == 3))
        born    = (self.state == 0) &  (neighbours == 3)
        self.state = (survive | born).view(np.uint8)
        self.time_step += 1

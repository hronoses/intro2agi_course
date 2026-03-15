import numpy as np
import cv2


class World:
    """2D binary world (Conway's Game of Life)."""

    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.state = np.random.randint(0, 2, size=(height, width)).astype(np.uint8)
        self.time_step = 0

    def get_value(self, x, y):
        return self.state[y % self.height, x % self.width]

    def set_value(self, x, y, value):
        self.state[y % self.height, x % self.width] = value

    def step(self):
        p = np.pad(self.state, 1, mode='wrap')
        neighbours = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                      p[1:-1, :-2]                + p[1:-1, 2:] +
                      p[2:,   :-2] + p[2:,  1:-1] + p[2:,  2:])
        survive = (self.state == 1) & ((neighbours == 2) | (neighbours == 3))
        born    = (self.state == 0) &  (neighbours == 3)
        self.state = (survive | born).view(np.uint8)
        self.time_step += 1

    def visualize_opencv(self, agent_pos=None, cell_size=30):
        colour_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        colour_map[self.state == 1] = (255, 255, 255)
        if agent_pos is not None:
            colour_map[agent_pos[1], agent_pos[0]] = (0, 0, 255)

        img = np.repeat(np.repeat(colour_map, cell_size, axis=0), cell_size, axis=1)
        img[:, ::cell_size] = (50, 50, 50)
        img[::cell_size, :] = (50, 50, 50)
        return img

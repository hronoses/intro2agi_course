import numpy as np
import cv2


_DIRECTIONS = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}


class Agent:
    """Agent that can perceive and move in the 2D world."""

    def __init__(self, world, start_pos=(0, 0), perception_radius=1):
        self.world = world
        self.position = list(start_pos)
        self.perception_radius = perception_radius

    def perceive(self):
        r = self.perception_radius
        x, y = self.position
        ys = np.arange(y - r, y + r + 1) % self.world.height
        xs = np.arange(x - r, x + r + 1) % self.world.width
        return self.world.state[np.ix_(ys, xs)]

    def get_sensory_input_opencv(self, cell_size=30):
        perception = self.perceive()
        size = perception.shape[0]

        colour_map = np.full((size, size, 3), (50, 50, 50), dtype=np.uint8)
        colour_map[perception == 1] = (255, 255, 255)
        colour_map[self.perception_radius, self.perception_radius] = (0, 0, 255)

        img = np.repeat(np.repeat(colour_map, cell_size, axis=0), cell_size, axis=1)
        img[:, ::cell_size] = (100, 100, 100)
        img[::cell_size, :] = (100, 100, 100)
        return img

    def move(self, direction):
        dx, dy = _DIRECTIONS.get(direction, (0, 0))
        self.position[0] = (self.position[0] + dx) % self.world.width
        self.position[1] = (self.position[1] + dy) % self.world.height

    def toggle_cell(self):
        x, y = self.position
        self.world.set_value(x, y, 1 - self.world.get_value(x, y))

    def set_cell(self, value):
        self.world.set_value(self.position[0], self.position[1], value)

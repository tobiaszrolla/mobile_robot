from .World import World
from .obstacles import CircularObstacle, RectangularObstacle
import numpy as np

class WorldFactory:
    @staticmethod
    def create_empty_world(width=500, length=1000):
        return World(width=width, length=length)
    
    def create_random_world(width=500, length=1000, num_circles=5, num_rects=3, seed=None):
        if seed is not None:
            np.random.seed(seed)

        world = World(width=width, length=length)

        for _ in range(num_circles):
            pos = np.random.uniform([0,0], [width,length])
            radius = np.random.uniform(5, 20)
            world.add_obstacle(CircularObstacle(pos, radius))

        for _ in range(num_rects):
            bottom_left = np.random.uniform([0,0], [width-20,length-20])
            w, h = np.random.uniform(5, 30, size=2)
            world.add_obstacle(RectangularObstacle(bottom_left, w, h))

        return world
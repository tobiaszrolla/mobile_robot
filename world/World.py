import numpy as np

class World:
    def __init__(self, width = 500, length = 1000):
        self.width = width
        self.lenght = length
        self.obstacles = []
        
    def add_obstacle(self, obs):
        self.obstacles.append(obs)
        
    def check_collision(self, point):
        return any(obs.is_colliding(point) for obs in self.obstacles)
    
    def is_inside_bounds(self, point):
        x, y = point
        return 0 <= x <= self.width and 0 <= y <= self.lenght
        
import numpy as np


class Obstacle:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
    
    def is_colliding(self, point):
        """Sprawdza kolizję z punktem (np. robotem)"""
        raise NotImplementedError("Implement in subclass")


class CircularObstacle(Obstacle):
    def __init__(self, position, radius):
        super().__init__(position)
        self.radius = radius
    
    def is_colliding(self, point):
        return np.linalg.norm(point - self.position) <= self.radius


class RectangularObstacle(Obstacle):
    def __init__(self, bottom_left, width, height):
        super().__init__(bottom_left)
        self.width = width
        self.height = height
    
    def is_colliding(self, point):
        x, y = point
        bx, by = self.position
        return bx <= x <= bx + self.width and by <= y <= by + self.height
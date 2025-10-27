from world.WorldFactory import WorldFactory
import matplotlib.pyplot as plt
from world.obstacles import CircularObstacle, RectangularObstacle

# Tworzymy losowy świat
world = WorldFactory.create_random_world(width=100, length=200, num_circles=3, num_rects=2, seed=42)

# Rysujemy przeszkody
plt.figure(figsize=(8,6))
for obs in world.obstacles:
    if isinstance(obs, CircularObstacle):
        circle = plt.Circle(obs.position, obs.radius, color='r', alpha=0.5)
        plt.gca().add_patch(circle)
    elif isinstance(obs, RectangularObstacle):
        rect = plt.Rectangle(obs.position, obs.width, obs.height, color='b', alpha=0.5)
        plt.gca().add_patch(rect)

plt.xlim(0, world.width)
plt.ylim(0, world.lenght)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Random World with obstacles")
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("world.png")
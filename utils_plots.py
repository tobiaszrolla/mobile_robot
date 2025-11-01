import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def _draw_obstacles(ax, world):
    """Pomocnicza funkcja do rysowania przeszkód świata."""
    if world is None or not hasattr(world, "obstacles"):
        return

    for obs in world.obstacles:
        if hasattr(obs, "radius"):
            circle = plt.Circle(obs.position, obs.radius, color='gray', alpha=0.4)
            ax.add_patch(circle)
        elif hasattr(obs, "width") and hasattr(obs, "height"):
            rect = plt.Rectangle(
                obs.position,
                obs.width,
                obs.height,
                color='gray',
                alpha=0.4
            )
            ax.add_patch(rect)


def plot_trajectory(trajectory, speeds, goal, world=None):
    """Rysuje trajektorię robota wraz z przeszkodami (jeśli podano world)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')

    # rysowanie przeszkód
    _draw_obstacles(ax, world)

    # Trajektoria robota
    ax.scatter(trajectory[:, 0], trajectory[:, 1], c='b', s=1, label="Ścieżka robota")

    # Strzałki (orientacja)
    for i in range(0, len(trajectory), 10):
        arrow_length = np.clip(1 * speeds[i - 1], 0.01, 1)
        x, y, heading = trajectory[i]
        dx = np.cos(heading) * arrow_length
        dy = np.sin(heading) * arrow_length
        ax.arrow(x, y, dx, dy, head_width=0.25, color='r', length_includes_head=True)

    # Punkty charakterystyczne
    ax.scatter(goal[0], goal[1], c='orange', s=80, marker='X', label="Cel")
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='g', s=60, label="Start")
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='r', s=60, label="Koniec")

    ax.set_title("Trajektoria ruchu robota z przeszkodami")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_speed(time, speed_values):
    """Rysuje prędkość robota w czasie."""
    plt.figure()
    plt.plot(time, speed_values)
    plt.title("Prędkość robota w czasie")
    plt.xlabel("Czas [s]")
    plt.ylabel("Prędkość [m/s]")
    plt.grid(True)
    plt.show()


def plot_animation(trajectory, goal, world=None):
    """Animacja ruchu robota z przeszkodami."""
    fig, ax = plt.subplots()
    ax.set_xlim(np.min(trajectory[:, 0]) - 5, np.max(trajectory[:, 0]) + 5)
    ax.set_ylim(np.min(trajectory[:, 1]) - 5, np.max(trajectory[:, 1]) + 5)
    ax.set_aspect('equal')

    # --- przeszkody ---
    _draw_obstacles(ax, world)

    line, = ax.plot([], [], 'b-')
    ax.scatter(goal[0], goal[1], c='orange', s=80, marker='X', label="Cel")

    def update(frame):
        # usuwanie poprzednich strzałek
        for artist in list(ax.patches):
            if isinstance(artist, plt.Arrow):
                artist.remove()

        line.set_data(trajectory[:frame, 0], trajectory[:frame, 1])
        x, y, heading = trajectory[frame]
        dx = np.cos(heading)
        dy = np.sin(heading)
        ax.arrow(x, y, dx, dy, head_width=0.25, color='r')
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=50, blit=False)
    plt.title("Animacja ruchu robota z przeszkodami")
    plt.show()


def plot_cost(time, cost_values):
    """Rysuje koszt w kolejnych krokach symulacji."""
    plt.figure()
    plt.plot(time, cost_values, 'r')
    plt.title("Koszt w kolejnych krokach symulacji")
    plt.xlabel("Czas [s]")
    plt.ylabel("Koszt")
    plt.grid(True)
    plt.show()

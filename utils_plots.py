import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_trajectory(trajectory, speeds, goal):
    # Trajektoria wykres
    plt.figure(figsize=(8, 6))
    # plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label="Ścieżka robota")  # Tor ruchu jako linia
    plt.scatter(trajectory[:, 0], trajectory[:, 1], c='b', s=1,
                label="Ścieżka robota")  # Położenie robota w kolejnych krokach (lepiej widać że przyspiesza)

    # strzałki do rysowania kątu robota (przydatne przy jakichś zabawach z dziwnymi ścieżkami)
    for i in range(0, len(trajectory), 10):
        arrow_length = np.clip(1 * speeds[i - 1], 0.01, 1)
        x, y, heading = trajectory[i]
        dx = np.cos(heading) * arrow_length
        dy = np.sin(heading) * arrow_length
        plt.arrow(x, y, dx, dy, head_width=0.25, color='r', length_includes_head=True)

    plt.scatter(goal[0], goal[1], c='b', s=80, marker='X', label="Cel")
    plt.scatter(trajectory[0, 0], trajectory[0, 1], c='g', s=60, label="Start")
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='r', s=60, label="Koniec")

    plt.title("Trajektoria ruchu robota")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_speed(time, speed_values):
    # Wykres predkości w czasie
    plt.figure()
    plt.plot(time, speed_values)
    plt.title("Prędkość robota w czasie")
    plt.xlabel("Czas [s]")
    plt.ylabel("Prędkość [m/s]")
    plt.grid(True)
    plt.show()


def plot_animation(trajectory, goal):
    # Animacja
    fig, ax = plt.subplots()
    ax.set_xlim(-5, 30)
    ax.set_ylim(-5, 30)
    ax.set_aspect('equal')
    line, = ax.plot([], [], 'b-')
    ax.scatter(goal[0], goal[1], c='b', s=80, marker='X', label="Cel")

    def update(frame):
        for p in reversed(ax.patches):
            p.remove()
        line.set_data(trajectory[:frame, 0], trajectory[:frame, 1])
        x, y, heading = trajectory[frame]
        dx = np.cos(heading)
        dy = np.sin(heading)
        ax.arrow(x, y, dx, dy, head_width=0.25, color='r')
        if x == trajectory[-1, 0] and y == trajectory[-1, 1]:
            ani.event_source.stop()
            return line,
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=50, blit=False)
    plt.show()


def plot_cost(time, cost_values):
    # Wykres kosztu w czasie
    plt.figure()
    plt.plot(time, cost_values, 'r')
    plt.title("Koszt w kolejnych krokach symulacji")
    plt.xlabel("Czas [s]")
    plt.ylabel("Koszt")
    plt.grid(True)
    plt.show()
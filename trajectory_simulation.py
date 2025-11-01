import numpy as np

from MobileRobot import MobileRobot
from cost_function import compute_cost
from utils_plots import plot_speed, plot_trajectory, plot_animation, plot_cost


def run_simulation(goal=(25.0, 25.0), starting_state=None, world=None, visualize=True):
    robot = MobileRobot()
    state = np.array(starting_state if starting_state is not None else [0.0, 0.0, np.deg2rad(0.0)])
    goal = np.array(goal)
    dt = 0.1
    steps = 2000

    trajectory = [state.copy()]
    speed_values = []
    cost_values = []
    controls = []
    total_cost = 0.0

    k_steer = 1.5
    k_speed = 0.5
    dist_stop = 0.5

    for t in range(steps):
        x, y, heading = state
        dx = goal[0] - x
        dy = goal[1] - y

        distance = np.hypot(dx, dy)
        angle_to_goal = np.arctan2(dy, dx)

        steering = k_steer * (angle_to_goal - heading)
        accel = k_speed * distance if distance > dist_stop else 0.0

        control = np.array([accel, steering])
        deriv = robot.deriv(state, control)
        state = state + deriv * dt

        trajectory.append(state.copy())
        speed_values.append(robot.speed)
        controls.append(control)

        cost = compute_cost(state, control, goal, world)
        cost_values.append(cost)
        total_cost += cost

        if distance < dist_stop:
            print(f"Robot dotarł do celu po {t * dt:.1f} s")
            break

    trajectory = np.array(trajectory)
    time = np.arange(len(cost_values)) * dt

    result = {
        "trajectory": trajectory,
        "controls": np.array(controls),
        "total_cost": total_cost,
        "steps": t,
    }

    if visualize:
        print(f"\nŁączny koszt trajektorii: {total_cost:.2f}")
        print(f"Średni koszt na krok: {np.mean(cost_values):.3f}")
        plot_trajectory(trajectory, speed_values, goal, world=world)
        plot_speed(time, speed_values)
        plot_animation(trajectory, goal, world=world)
        plot_cost(time, cost_values)

    return result



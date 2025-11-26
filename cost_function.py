import numpy as np


def compute_cost(state, control, goal, world=None, weights=None):
    if weights is None:
        weights = {
            'w_goal': 5.0,
            'w_energy': 3.0,
        }

    x, y, heading = state
    accel, steering = control

    goal_dist = np.linalg.norm([goal[0] - x, goal[1] - y])

    energy = accel**2 + 0.1 * steering**2

    cost = (
        weights['w_goal'] * goal_dist +
        weights['w_energy'] * energy
    )



    if world and world.check_collision(state[:2]):
        print("----------------------------------------------------------detected")
        cost += 100000

    return cost

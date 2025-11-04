import numpy as np
from MobileRobot import MobileRobot
from cost_function import compute_cost


class QLearningRobot:
    def __init__(self, state_discretization=20, learning_rate=0.1, discount_factor=0.95, epsilon=0.3):
        """Initialize the Q-learning robot."""
        self.robot = MobileRobot()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.state_discretization = state_discretization
        
        # Action space: (acceleration, steering)
        # Discretize actions to match the original control strategy better
        self.accelerations = np.array([-0.5, 0.0, 0.5])  # Smaller acceleration steps
        num_steering_angles = 7  # More steering angles for finer control
        self.steering_angles = np.linspace(-np.deg2rad(30), np.deg2rad(30), num_steering_angles)
        
        # Initialize Q-table
        self.q_table = {}
        
        # Add collision penalty
        self.collision_penalty = 1000
        
        # Goal reward
        self.goal_reward = 1000
        
        # Distance threshold for goal
        self.goal_threshold = 0.5

    def discretize_state(self, state, goal):
        """Convert continuous state to discrete state representation."""
        x, y, heading = state
        dx = goal[0] - x
        dy = goal[1] - y
        
        distance = np.hypot(dx, dy)
        angle_to_goal = np.arctan2(dy, dx)
        angle_diff = (angle_to_goal - heading + np.pi) % (2 * np.pi) - np.pi
        
        # Discretize values with better resolution for important state variables
        distance_disc = int(np.clip(distance / 5, 0, self.state_discretization - 1))
        angle_diff_disc = int(np.clip(
            (angle_diff + np.pi) / (2 * np.pi) * self.state_discretization,
            0, self.state_discretization - 1
        ))
        
        return (distance_disc, angle_diff_disc)
    
    def get_action(self, state_disc):
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            # Exploration: random action
            accel_idx = np.random.randint(len(self.accelerations))
            steer_idx = np.random.randint(len(self.steering_angles))
        else:
            # Exploitation: best known action
            if state_disc not in self.q_table:
                self.q_table[state_disc] = np.zeros((len(self.accelerations), len(self.steering_angles)))
            
            q_values = self.q_table[state_disc]
            action_indices = np.unravel_index(np.argmax(q_values), q_values.shape)
            accel_idx, steer_idx = action_indices
        
        return np.array([self.accelerations[accel_idx], self.steering_angles[steer_idx]])
    
    def update_q_value(self, state_disc, action, next_state, next_state_disc, goal, world):
        """Update Q-value for state-action pair."""
        if state_disc not in self.q_table:
            self.q_table[state_disc] = np.zeros((len(self.accelerations), len(self.steering_angles)))
        if next_state_disc not in self.q_table:
            self.q_table[next_state_disc] = np.zeros((len(self.accelerations), len(self.steering_angles)))
            
        # Find indices of the action in our discrete action space
        accel_idx = np.argmin(np.abs(self.accelerations - action[0]))
        steer_idx = np.argmin(np.abs(self.steering_angles - action[1]))
        
        # Check for collision and goal reached
        collision = world.check_collision(next_state[:2])
        distance_to_goal = np.hypot(goal[0] - next_state[0], goal[1] - next_state[1])
        goal_reached = distance_to_goal < self.goal_threshold
        
        # Calculate reward
        if collision:
            reward = -self.collision_penalty
        elif goal_reached:
            reward = self.goal_reward
        else:
            # Use negative cost as reward, but scale it to make it more meaningful
            reward = -compute_cost(next_state, action, goal, world) / 10
            # Add distance-based component to encourage moving towards goal
            reward += (1.0 / (distance_to_goal + 1.0)) * 10
        
        # Q-learning update
        current_q = self.q_table[state_disc][accel_idx, steer_idx]
        if collision:
            next_max_q = -self.collision_penalty
        elif goal_reached:
            next_max_q = self.goal_reward
        else:
            next_max_q = np.max(self.q_table[next_state_disc])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state_disc][accel_idx, steer_idx] = new_q

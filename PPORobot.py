import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
from MobileRobot import MobileRobot
from cost_function import compute_cost


class ActorCritic(nn.Module):
    """Actor-Critic network."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.ones(action_dim) * -0.5)  # Lower std
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
    
    def forward(self, state):
        features = self.shared(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean).clamp(min=0.05, max=0.5)
        value = self.critic(features)
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            return action_mean, None, value
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, state, action):
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


class PPOBuffer:
    """Rollout buffer."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self):
        data = {
            'states': np.array(self.states, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.float32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'values': np.array(self.values, dtype=np.float32),
            'log_probs': np.array(self.log_probs, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.float32)
        }
        self.clear()
        return data
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def __len__(self):
        return len(self.states)


class PPORobot:
    """PPO - BALANCED for good movement and collision avoidance."""
    
    def __init__(self, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, epochs=10, batch_size=64):
        """Initialize PPO robot."""
        self.robot_template = MobileRobot()
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.state_dim = 22  # 6 (goal/speed) + 16 (lidar)
        self.action_dim = 2
        
        self.accel_max = self.robot_template.accel_max
        self.steer_max = np.deg2rad(30)
        
        self.device = torch.device("cpu")
        
        self.policy = ActorCritic(self.state_dim, self.action_dim, hidden_dim=256).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.buffer = PPOBuffer()
        
        self.goal_reward = 2000.0
        self.collision_penalty = 1000.0
        self.goal_threshold = 5.0
        
        # BALANCED: 5 repeats = 0.5 second per action
        self.action_repeat = 5
        
        print(f"PPO Robot initialized (BALANCED)")
        print(f"  Accel max: {self.accel_max:.2f}")
        print(f"  Action repeat: {self.action_repeat}x (0.5 second per action)")
        print(f"  Expected movement: ~2-5m per action")
        print(f"  Goal threshold: {self.goal_threshold}m")
    
    def get_state(self, robot_state, speed, goal, world):
        """Get state representation with LIDAR."""
        x, y, heading = robot_state
        
        # 1. Goal info
        dx = goal[0] - x
        dy = goal[1] - y
        distance = np.hypot(dx, dy)
        
        dx_norm = np.clip(dx / 100.0, -2, 2)
        dy_norm = np.clip(dy / 100.0, -2, 2)
        distance_norm = np.clip(distance / 100.0, 0, 2)
        
        angle_to_goal = np.arctan2(dy, dx)
        heading_diff = (angle_to_goal - heading + np.pi) % (2 * np.pi) - np.pi
        cos_heading = np.cos(heading_diff)
        sin_heading = np.sin(heading_diff)
        
        speed_norm = speed / self.robot_template.speed_max
        
        # 2. LIDAR (16 rays)
        lidar_readings = []
        num_rays = 16
        max_range = 30.0
        angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
        
        for angle in angles:
            ray_angle = heading + angle
            dist = max_range
            
            # Raycast (finer steps for accuracy)
            # Check every 1.0m to save perf, but 0.5 near robot
            for r in np.linspace(0.5, max_range, 30):
                px = x + r * np.cos(ray_angle)
                py = y + r * np.sin(ray_angle)
                
                if not world.is_inside_bounds((px, py)) or world.check_collision((px, py)):
                    dist = r
                    break
            
            lidar_readings.append(dist / max_range)
            
        return np.concatenate([
            [dx_norm, dy_norm, distance_norm, cos_heading, sin_heading, speed_norm],
            lidar_readings
        ], dtype=np.float32)
    
    def select_action(self, state, deterministic=False):
        """Select action - LEARNS both accel AND steering."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean, action_std, value = self.policy.forward(state_tensor)
        
        if deterministic:
            action = action_mean.cpu().numpy()[0]
            log_prob = None
            value = value.cpu().item() if value is not None else None
        else:
            dist = Normal(action_mean, action_std)
            action_sample = dist.sample()
            log_prob = dist.log_prob(action_sample).sum(dim=-1).cpu().item()
            action = action_sample.cpu().numpy()[0]
            value = value.cpu().item() if value is not None else None
        
        # Map network output to valid ranges
        # Network outputs roughly [-2, 2], map to action ranges
        # Allow braking/reversing: map action[0] directly to accel
        accel = np.clip(action[0] * self.accel_max, -self.accel_max, self.accel_max)
        steer = np.clip(action[1] * self.steer_max, -self.steer_max, self.steer_max)
        
        return np.array([accel, steer]), log_prob, value
    
    def execute_action_repeated(self, robot, state, action, goal, world):
        """
        Execute action 30 times - balanced movement and control.
        """
        total_reward = 0
        trajectory = []
        
        for repeat_idx in range(self.action_repeat):
            deriv = robot.deriv(state, action)
            next_state = state + deriv * 0.1
            
            # Check collision using cost function
            step_cost = compute_cost(next_state, action, goal, world)
            
            if step_cost > 10000:
                total_reward -= self.collision_penalty
                # End episode immediately on collision
                return next_state, total_reward, True, trajectory
            
            if not world.is_inside_bounds(next_state[:2]):
                total_reward -= self.collision_penalty
                # End episode immediately on out of bounds
                return next_state, total_reward, True, trajectory
            
            # Check goal
            distance = np.hypot(goal[0] - next_state[0], goal[1] - next_state[1])
            if distance < self.goal_threshold:
                total_reward += self.goal_reward
                return next_state, total_reward, True, trajectory
            
            # Compute reward
            curr_dist = np.hypot(goal[0] - state[0], goal[1] - state[1])
            progress = curr_dist - distance
            
            # Reward components
            progress_reward = progress * 100.0  # Stronger progress signal
            
            # Heading reward (guidance)
            dx = goal[0] - next_state[0]
            dy = goal[1] - next_state[1]
            angle_to_goal = np.arctan2(dy, dx)
            heading_diff = (angle_to_goal - next_state[2] + np.pi) % (2 * np.pi) - np.pi
            heading_reward = np.cos(heading_diff) * 0.5

            # Small penalty for being far from goal
            distance_penalty = -distance * 0.01
            
            # Penalty for obstacles nearby (from cost function)
            # Increase obstacle avoidance weight
            obstacle_penalty = -step_cost * 0.1 if step_cost < 10000 else 0
            
            # Time penalty
            time_penalty = -0.1
            
            step_reward = progress_reward + heading_reward + distance_penalty + obstacle_penalty + time_penalty
            
            total_reward += step_reward
            trajectory.append(next_state.copy())
            
            state = next_state
        
        return state, total_reward, False, trajectory
    
    def compute_step_cost(self, state, action, goal, world):
        """Compute cost - for comparison only."""
        return compute_cost(state, action, goal, world)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute GAE."""
        advantages = []
        gae = 0
        values = np.append(values, next_value)
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                gae = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def update(self, next_state_value):
        """Update policy."""
        data = self.buffer.get()
        
        if len(data['states']) < self.batch_size:
            return None, None, None
        
        advantages, returns = self.compute_gae(
            data['rewards'], data['values'], data['dones'], next_state_value
        )
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states = torch.FloatTensor(data['states']).to(self.device)
        actions = torch.FloatTensor(data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for _ in range(self.epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                log_probs, entropy, values = self.policy.evaluate_actions(batch_states, batch_actions)
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * (batch_returns - values.squeeze()).pow(2).mean()
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + 0.5 * value_loss + 0.02 * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        if num_updates == 0:
            return None, None, None
        
        return total_policy_loss / num_updates, total_value_loss / num_updates, total_entropy / num_updates
    
    def save_model(self, filepath):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
        print(f"Model saved: {filepath}")
    
    def load_model(self, filepath):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded: {filepath}")

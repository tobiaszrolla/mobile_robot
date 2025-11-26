import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from MobileRobot import MobileRobot
from cost_function import compute_cost

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared features
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor (Policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim) - 0.5) # Init std around 0.6
        
        # Critic (Value)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Orthogonal initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Actor output layer scaling
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, state):
        x = self.net(state)
        return x

    def get_action(self, state, deterministic=False):
        x = self.forward(state)
        mean = self.actor_mean(x)
        std = self.actor_logstd.exp().expand_as(mean)
        
        if deterministic:
            return mean, None, self.critic(x)
        
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(x)
        
        return action, log_prob, value

    def evaluate(self, state, action):
        x = self.forward(state)
        mean = self.actor_mean(x)
        std = self.actor_logstd.exp().expand_as(mean)
        value = self.critic(x)
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value

class PPOBuffer:
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
        return {
            'states': np.array(self.states, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.float32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'values': np.array(self.values, dtype=np.float32),
            'log_probs': np.array(self.log_probs, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.float32)
        }
    
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
    def __init__(self, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2):
        self.robot_template = MobileRobot()
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = 10
        self.batch_size = 64
        
        # State: [dx_norm, dy_norm, dist_norm, cos_h, sin_h, speed_norm, 16_lidar]
        self.state_dim = 6 + 16
        self.action_dim = 2
        
        self.device = torch.device("cpu")
        self.policy = ActorCritic(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        
        self.buffer = PPOBuffer()
        
        self.action_repeat = 5
        self.goal_threshold = 0.5
        
        # Limits
        self.accel_max = self.robot_template.accel_max
        self.steer_max = np.deg2rad(30)

    def get_state(self, robot_state, speed, goal, world):
        x, y, heading = robot_state
        
        # Goal relative
        dx = goal[0] - x
        dy = goal[1] - y
        dist = np.hypot(dx, dy)
        
        # Angle to goal
        angle_to_goal = np.arctan2(dy, dx)
        heading_err = (angle_to_goal - heading + np.pi) % (2*np.pi) - np.pi
        
        # Normalize
        # Assume max dist approx 200m
        state = [
            np.clip(dx / 100.0, -2, 2),
            np.clip(dy / 100.0, -2, 2),
            np.clip(dist / 100.0, 0, 2),
            np.cos(heading_err),
            np.sin(heading_err),
            speed / self.robot_template.speed_max
        ]
        
        # Lidar
        lidar = self._raycast(x, y, heading, world)
        state.extend(lidar)
        
        return np.array(state, dtype=np.float32)

    def _raycast(self, x, y, heading, world):
        num_rays = 16
        max_range = 30.0
        angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
        readings = []
        
        for angle in angles:
            ray_angle = heading + angle
            dist = max_range
            
            # Coarse check then fine check could be better, but linear is robust
            # Step size 1.0m is safe for 5m+ obstacles
            for r in np.arange(1.0, max_range + 1.0, 1.0):
                px = x + r * np.cos(ray_angle)
                py = y + r * np.sin(ray_angle)
                
                if not world.is_inside_bounds((px, py)) or world.check_collision((px, py)):
                    dist = r
                    break
            
            readings.append(dist / max_range)
            
        return readings

    def select_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_t, deterministic)
            
        action_np = action.cpu().numpy()[0]
        val_np = value.cpu().item() if value is not None else 0.0
        lp_np = log_prob.cpu().item() if log_prob is not None else 0.0
        
        return action_np, lp_np, val_np

    def execute_action_repeated(self, robot, state, action_raw, goal, world):
        # Map raw action [-inf, inf] (from Normal) to physical limits
        # Tanh in network would bound it to [-1, 1], but we used Linear output
        # So we clip here.
        
        # Action 0: Accel [-1, 1] -> [-accel_max, accel_max]
        # Action 1: Steer [-1, 1] -> [-steer_max, steer_max]
        
        # Clip raw action to [-1, 1] for stability
        action_clipped = np.clip(action_raw, -1.0, 1.0)
        
        accel = action_clipped[0] * self.accel_max
        steer = action_clipped[1] * self.steer_max
        
        phys_action = np.array([accel, steer])
        
        total_reward = 0
        trajectory = []
        done = False
        
        for _ in range(self.action_repeat):
            # Physics step
            deriv = robot.deriv(state, phys_action)
            next_state = state + deriv * 0.1
            
            # Check bounds/collision
            if not world.is_inside_bounds(next_state[:2]) or world.check_collision(next_state[:2]):
                total_reward = -100.0 # Crash penalty
                done = True
                state = next_state
                break
                
            # Check goal
            dist = np.hypot(goal[0] - next_state[0], goal[1] - next_state[1])
            if dist < self.goal_threshold:
                total_reward = 100.0 # Goal reward
                done = True
                state = next_state
                break
            
            # Step Reward
            # 1. Progress
            prev_dist = np.hypot(goal[0] - state[0], goal[1] - state[1])
            progress = prev_dist - dist
            total_reward += progress * 2.0
            
            # 2. Speed bonus (small)
            total_reward += 0.01 * (robot.speed / self.robot_template.speed_max)
            
            # 3. Time penalty
            total_reward -= 0.01
            
            state = next_state
            trajectory.append(state.copy())
            
        return state, total_reward, done, trajectory

    def update(self, next_value):
        data = self.buffer.get()
        if len(data['states']) == 0: return None, None, None
        
        # Compute GAE
        rewards = data['rewards']
        values = data['values']
        dones = data['dones']
        
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        
        # Append next_value for the last step
        values_next = np.append(values, next_value)
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                lastgaelam = delta
            else:
                delta = rewards[t] + self.gamma * values_next[t+1] - values[t]
                lastgaelam = delta + self.gamma * self.gae_lambda * lastgaelam
            advantages[t] = lastgaelam
            
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to torch
        states = torch.FloatTensor(data['states']).to(self.device)
        actions = torch.FloatTensor(data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # PPO Epochs
        total_p_loss = 0
        total_v_loss = 0
        
        for _ in range(self.epochs):
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                idx = indices[start:start+self.batch_size]
                
                b_states = states[idx]
                b_actions = actions[idx]
                b_old_lp = old_log_probs[idx]
                b_ret = returns[idx]
                b_adv = advantages[idx]
                
                log_probs, entropy, values = self.policy.evaluate(b_states, b_actions)
                
                # Ratio
                ratio = torch.exp(log_probs - b_old_lp)
                
                # Clip
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_adv
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (b_ret - values.squeeze()).pow(2).mean()
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_p_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                
        return total_p_loss, total_v_loss, 0.0

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)
        
    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))


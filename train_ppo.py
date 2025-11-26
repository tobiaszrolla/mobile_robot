from world.WorldFactory import WorldFactory
from PPORobot import PPORobot
from MobileRobot import MobileRobot
import numpy as np
from utils_plots import plot_trajectory
import matplotlib.pyplot as plt
import time
from collections import deque


def train_ppo(episodes=2000, max_steps=60, update_every=1024, goal=(50.0, 75.0), save_path='ppo_model.pth'):
    """
    Train PPO agent - COMPATIBLE with MobileRobot class.
    Robot instance manages its own speed internally.
    """
    # Environment
    # Try seeds until we find a safe world
    seed = 42
    while True:
        world = WorldFactory.create_random_world(width=100, length=200, num_circles=3, num_rects=2, seed=seed)
        # Start in the middle of the width, slightly up from bottom
        start_state = np.array([5.0, 10.0, np.deg2rad(90.0)])
        
        # Check if start is safe (obstacles AND bounds)
        if not world.check_collision(start_state[:2]) and world.is_inside_bounds(start_state[:2]):
            print(f"Found safe world with seed {seed}")
            break
        
        print(f"Seed {seed} has collision at start, trying next...")
        seed += 1

    ppo_robot = PPORobot()
    
    print("="*70)
    print("PPO TRAINING - MobileRobot Compatible")
    print("="*70)
    print(f"Goal: ({goal[0]:.1f}, {goal[1]:.1f})")
    print(f"Distance: {np.hypot(goal[0], goal[1]):.1f}m")
    print(f"Episodes: {episodes}")
    print(f"Max steps: {max_steps}")
    print(f"Action repeat: {ppo_robot.action_repeat}x")
    print(f"Sim time per action: {ppo_robot.action_repeat * 0.1:.1f}s")
    print(f"Max episode duration: {max_steps * ppo_robot.action_repeat * 0.1:.1f}s")
    print(f"Update every: {update_every} steps")
    print("="*70 + "\n")
    
    # History
    episode_rewards = []
    episode_lengths = []
    success_history = []
    policy_losses = []
    value_losses = []
    
    best_reward = -float('inf')
    best_trajectory = None
    best_steps = float('inf')
    
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    reward_window = deque(maxlen=100)
    success_window = deque(maxlen=100)
    
    total_steps = 0
    start_time = time.time()
    
    for episode in range(episodes):
        # Reset - FRESH MobileRobot instance per episode
        # Robot starts with speed=0, will accelerate naturally
        robot = MobileRobot()
        state = start_state.copy()
        
        episode_reward = 0
        trajectory = [state.copy()]
        done = False
        
        for step in range(max_steps):
            # Get state representation (includes current robot speed)
            state_repr = ppo_robot.get_state(state, robot.speed, goal, world)
            
            # Select action from policy
            action, log_prob, value = ppo_robot.select_action(state_repr, deterministic=False)
            
            # Execute action (robot.deriv updates robot.speed internally)
            # Returns: next_state, reward, done, sub_trajectory
            next_state, reward, done, sub_trajectory = ppo_robot.execute_action_repeated(
                robot, state, action, goal, world
            )
            
            if (episode+1) % 100 == 0 and step % 10 == 0:
                 print(f"    Step {step}: Action={action}, Speed={robot.speed:.2f}, Reward={reward:.2f}")

            episode_reward += reward
            
            # Store transition in buffer
            ppo_robot.buffer.store(state_repr, action, reward, value, log_prob, done)
            
            # Update state
            state = next_state
            trajectory.append(state.copy())
            trajectory.extend(sub_trajectory)  # Add intermediate states
            total_steps += 1
            
            # Update policy periodically
            if total_steps % update_every == 0 and len(ppo_robot.buffer) > 0:
                # Get value of final state for GAE
                next_state_repr = ppo_robot.get_state(state, robot.speed, goal, world)
                _, _, next_value = ppo_robot.select_action(next_state_repr, deterministic=True)
                
                if done:
                    next_value = 0.0
                
                # Perform PPO update
                policy_loss, value_loss, entropy = ppo_robot.update(next_value)
                
                if policy_loss is not None:
                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
                    print(f"  [Update {len(policy_losses)}] Policy: {policy_loss:.4f}, "
                          f"Value: {value_loss:.4f}, Entropy: {entropy:.4f}")
            
            if done:
                break
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(len(trajectory))
        reward_window.append(episode_reward)
        
        # Determine outcome
        final_dist = np.hypot(goal[0] - state[0], goal[1] - state[1])
        distance_traveled = np.hypot(state[0] - start_state[0], state[1] - start_state[1])
        
        if final_dist < ppo_robot.goal_threshold:
            success_count += 1
            success_window.append(1)
            outcome = "SUCCESS"
            
            if len(trajectory) < best_steps:
                best_steps = len(trajectory)
                best_reward = episode_reward
                best_trajectory = np.array(trajectory)
        else:
            success_window.append(0)
            if world.check_collision(state[:2]) or not world.is_inside_bounds(state[:2]):
                collision_count += 1
                outcome = "COLLISION"
                # Penalize collision heavily in the reward window to track stability
                episode_reward -= 500
            else:
                timeout_count += 1
                outcome = "TIMEOUT"
        
        success_history.append(1 if outcome == "SUCCESS" else 0)
        
        # Print progress with distance traveled
        if (episode + 1) % 10 == 0:
            print(f"Ep {episode+1:4d} | R: {episode_reward:7.1f} | Steps: {step+1:3d} | "
                  f"{outcome:9s} | Dist: {final_dist:5.1f}m | Traveled: {distance_traveled:5.1f}m | "
                  f"Speed: {robot.speed:4.2f}")
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_window)
            success_rate = np.mean(success_window) * 100
            elapsed = time.time() - start_time
            
            print(f"\n{'='*70}")
            print(f"Episode {episode+1}/{episodes}")
            print(f"{'='*70}")
            print(f"Avg Reward (100):     {avg_reward:10.2f}")
            print(f"Success Rate (100):   {success_rate:10.1f}%")
            print(f"Best Steps:           {best_steps if best_steps != float('inf') else 'N/A':>10}")
            print(f"Success/Coll/Time:    {success_count}/{collision_count}/{timeout_count}")
            print(f"Buffer Size:          {len(ppo_robot.buffer):10d}")
            print(f"Elapsed:              {elapsed/60:10.1f} min")
            print(f"{'='*70}\n")
        
        # Save checkpoints
        if (episode + 1) % 500 == 0:
            checkpoint_path = f"{save_path}.ep{episode+1}"
            ppo_robot.save_model(checkpoint_path)
            print(f"✓ Checkpoint: {checkpoint_path}")
            
            # Show progress with best trajectory
            if best_trajectory is not None:
                print(f"  Best so far: {best_steps} steps, reward: {best_reward:.2f}")
    
    # Save final model
    ppo_robot.save_model(save_path)
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("PPO TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time:           {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"Final success rate:   {np.mean(success_window)*100:.1f}%")
    print(f"Total successes:      {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
    print(f"Total collisions:     {collision_count}")
    print(f"Total timeouts:       {timeout_count}")
    print(f"Best trajectory:      {best_steps} steps")
    print(f"Best reward:          {best_reward:.2f}")
    print(f"Model saved:          {save_path}")
    print(f"{'='*70}")
    
    # Plot training curves
    print("\nGenerating training plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, linewidth=0.5, color='blue')
    if len(episode_rewards) > 50:
        smoothed = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
        axes[0, 0].plot(smoothed, linewidth=2, color='blue', label='Smoothed (50)')
    axes[0, 0].set_title('Episode Rewards', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Losses
    if policy_losses:
        axes[0, 1].plot(policy_losses, alpha=0.6, label='Policy Loss', color='red')
        axes[0, 1].plot(value_losses, alpha=0.6, label='Value Loss', color='orange')
        axes[0, 1].set_title('Training Losses', fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No updates yet', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Training Losses', fontweight='bold', fontsize=12)
    
    # Episode lengths
    axes[1, 0].plot(episode_lengths, alpha=0.3, linewidth=0.5, color='green')
    if len(episode_lengths) > 50:
        smoothed = np.convolve(episode_lengths, np.ones(50)/50, mode='valid')
        axes[1, 0].plot(smoothed, linewidth=2, color='green', label='Smoothed (50)')
    axes[1, 0].set_title('Episode Lengths (Trajectory Waypoints)', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Waypoints')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Success rate
    success_rates = []
    for i in range(len(success_history)):
        start = max(0, i - 99)
        success_rates.append(np.mean(success_history[start:i+1]) * 100)
    axes[1, 1].plot(success_rates, linewidth=2, color='purple')
    axes[1, 1].axhline(50, linestyle='--', alpha=0.7, color='orange', label='50% target')
    axes[1, 1].axhline(80, linestyle='--', alpha=0.7, color='red', label='80% target')
    axes[1, 1].set_title('Success Rate (Rolling 100)', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Success Rate (%)')
    axes[1, 1].set_ylim([0, 105])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ppo_training_curves.png', dpi=150)
    print("✓ Saved: ppo_training_curves.png")
    plt.close()
    
    # Best trajectory visualization
    if best_trajectory is not None:
        print(f"\nVisualizing best trajectory ({best_steps} steps)...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Trajectory plot
        ax = axes[0]
        ax.set_aspect('equal')
        
        # Draw obstacles
        from matplotlib.patches import Circle, Rectangle
        from world.obstacles import CircularObstacle, RectangularObstacle
        
        for obs in world.obstacles:
            if isinstance(obs, CircularObstacle):
                circle = Circle(obs.position, obs.radius, color='red', alpha=0.5)
                ax.add_patch(circle)
            elif isinstance(obs, RectangularObstacle):
                rect = Rectangle(obs.position, obs.width, obs.height, color='red', alpha=0.5)
                ax.add_patch(rect)
        
        ax.plot(best_trajectory[:, 0], best_trajectory[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax.scatter(start_state[0], start_state[1], c='green', s=200, marker='o', label='Start', zorder=10)
        ax.scatter(goal[0], goal[1], c='gold', s=300, marker='*', label='Goal', zorder=10)
        
        # Draw heading arrows every N points
        arrow_freq = max(1, len(best_trajectory) // 20)
        for i in range(0, len(best_trajectory), arrow_freq):
            x, y, heading = best_trajectory[i]
            dx = np.cos(heading) * 2
            dy = np.sin(heading) * 2
            ax.arrow(x, y, dx, dy, head_width=1.5, color='darkblue', alpha=0.6, zorder=5)
        
        ax.set_xlim(-5, world.width + 5)
        ax.set_ylim(-5, world.lenght + 5)
        ax.set_xlabel('X [m]', fontsize=11)
        ax.set_ylabel('Y [m]', fontsize=11)
        ax.set_title(f'Best PPO Trajectory\n{best_steps} waypoints, Reward: {best_reward:.2f}', 
                    fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Position over time
        ax = axes[1]
        times = np.arange(len(best_trajectory)) * 0.1
        distances = [np.hypot(goal[0] - pt[0], goal[1] - pt[1]) for pt in best_trajectory]
        
        ax.plot(times, distances, 'r-', linewidth=2, label='Distance to goal')
        ax.axhline(ppo_robot.goal_threshold, color='g', linestyle='--', alpha=0.7, label='Goal threshold')
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel('Distance to Goal [m]', fontsize=11)
        ax.set_title('Progress Over Time', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('best_ppo_trajectory.png', dpi=150)
        print("✓ Saved: best_ppo_trajectory.png")
        plt.show()
    else:
        print("\n⚠ No successful trajectory to visualize")
    
    return ppo_robot, world


if __name__ == "__main__":
    trained_robot, world = train_ppo(
        episodes=3000,
        max_steps=300,      # Increased for complex maneuvers
        update_every=2048,   # Update less frequently (more stable)
        goal=(50.0, 75.0),
        save_path='ppo_model.pth'
    )

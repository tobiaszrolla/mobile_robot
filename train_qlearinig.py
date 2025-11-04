from world.WorldFactory import WorldFactory
from QLearningRobot import QLearningRobot
import numpy as np
from utils_plots import plot_trajectory, plot_animation
import matplotlib.pyplot as plt


def train_robot(episodes=15000, max_steps=2000, goal=(0.0, 215.0)):
    # Create environment
    world = WorldFactory.create_random_world(width=100, length=200, num_circles=3, num_rects=2, seed=42)
    q_robot = QLearningRobot()
    
    # Training history
    episode_rewards = []
    collision_count = 0
    success_count = 0
    best_trajectory = None
    best_steps = max_steps
    
    for episode in range(episodes):
        # Reset state
        state = np.array([0.0, 0.0, np.deg2rad(0.0)])
        total_reward = 0
        trajectory = [state.copy()]
        episode_collision = False
        
        for step in range(max_steps):
            # Check for collision at current state
            if world.check_collision(state[:2]):
                collision_count += 1
                episode_collision = True
                if episode % 50 == 0:
                    print(f"Episode {episode + 1}: Collision detected!")
                break
            
            # Get current state representation
            state_disc = q_robot.discretize_state(state, goal)
            
            # Choose action
            action = q_robot.get_action(state_disc)
            
            # Execute action
            deriv = q_robot.robot.deriv(state, action)
            next_state = state + deriv * 0.1  # dt = 0.1
            
            # Check if next state would result in collision
            if world.check_collision(next_state[:2]):
                collision_count += 1
                episode_collision = True
                # Add collision point to trajectory for visualization
                trajectory.append(next_state.copy())
                if episode % 50 == 0:
                    print(f"Episode {episode + 1}: Collision detected!")
                break
            
            # Get next state representation
            next_state_disc = q_robot.discretize_state(next_state, goal)
            
            # Update Q-table
            q_robot.update_q_value(state_disc, action, next_state, next_state_disc, goal, world)
            
            # Update state
            state = next_state
            trajectory.append(state.copy())
            
            # Check if goal reached
            distance_to_goal = np.hypot(goal[0] - state[0], goal[1] - state[1])
            if distance_to_goal < q_robot.goal_threshold:
                success_count += 1
                if episode % 10 == 0:
                    print(f"Episode {episode + 1}: Goal reached in {step + 1} steps!")
                if step + 1 < best_steps:
                    best_steps = step + 1
                    best_trajectory = np.array(trajectory)
                break
                
        trajectory = np.array(trajectory)
        
        # Decay exploration rate
        if episode % 100 == 0 and q_robot.epsilon > 0.01:
            q_robot.epsilon *= 0.95
        
        # Print progress every 10 episodes
        if (episode + 1) % 50 == 0:
            print(f"\nEpisode {episode + 1} Statistics:")
            print(f"Collisions so far: {collision_count}")
            print(f"Successful goals reached: {success_count}")
            print(f"Success rate: {success_count/(episode+1)*100:.2f}%")
            print(f"Current exploration rate (epsilon): {q_robot.epsilon:.3f}")
            
            # Visualize trajectory every 100 episodes
            # if (episode + 1) % 5000 == 0:
                # plt.figure(figsize=(10, 8))
                # if episode_collision:
                #     plt.title(f"Episode {episode + 1}: Trajectory (Ended in Collision)")
                # else:
                #     plt.title(f"Episode {episode + 1}: Trajectory (Completed)")
                # plot_trajectory(trajectory, [q_robot.robot.speed] * len(trajectory), goal, world)
                # plot_animation(trajectory, goal, world)
    
    # Final statistics
    print("\nTraining Complete!")
    print(f"Total episodes: {episodes}")
    print(f"Total collisions: {collision_count}")
    print(f"Total successful goals reached: {success_count}")
    print(f"Final success rate: {success_count/episodes*100:.2f}%")
    print(f"Best path length: {best_steps} steps")
    
    # Show best trajectory
    if best_trajectory is not None:
        plt.figure(figsize=(10, 8))
        plt.title("Best Trajectory Found")
        plot_trajectory(best_trajectory, [q_robot.robot.speed] * len(best_trajectory), goal, world)
        plot_animation(best_trajectory, goal, world)
    
    return q_robot


if __name__ == "__main__":
    trained_robot = train_robot()

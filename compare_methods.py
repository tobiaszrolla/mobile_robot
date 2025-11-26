from world.WorldFactory import WorldFactory
from PPORobot import PPORobot
from MobileRobot import MobileRobot
from DijkstraPlanner import DijkstraPlanner
import numpy as np
import matplotlib.pyplot as plt
from utils_plots import plot_trajectory, _draw_obstacles
import time
from collections import deque


def compute_path_length(trajectory):
    """Compute total path length."""
    if trajectory is None or len(trajectory) < 2:
        return float('inf')
    
    return sum(np.hypot(trajectory[i+1][0] - trajectory[i][0],
                        trajectory[i+1][1] - trajectory[i][1])
               for i in range(len(trajectory) - 1))


def evaluate_ppo(ppo_robot, world, start_state, goal, max_steps=500):
    """Evaluate trained PPO agent."""
    robot = MobileRobot()
    state = np.array(start_state)
    goal = np.array(goal)
    
    trajectory = [state.copy()]
    
    for step in range(max_steps):
        state_repr = ppo_robot.get_state(state, robot.speed, goal, world)
        action, _, _ = ppo_robot.select_action(state_repr, deterministic=True)
        
        next_state, _, done, sub_traj = ppo_robot.execute_action_repeated(
            robot, state, action, goal, world
        )
        
        state = next_state
        trajectory.extend(sub_traj)
        
        distance = np.hypot(goal[0] - state[0], goal[1] - state[1])
        
        if (step+1) % 10 == 0:
            print(f"  Step {step+1}: Dist to goal = {distance:.1f}m, Speed = {robot.speed:.1f}")

        if distance < ppo_robot.goal_threshold:
            print(f"PPO: Goal reached in {step+1} actions!")
            path_length = compute_path_length(trajectory)
            return np.array(trajectory), path_length, True
        
        if done:
            if world.check_collision(state[:2]) or not world.is_inside_bounds(state[:2]):
                print("PPO: Collision")
                return np.array(trajectory), float('inf'), False
    
    print(f"PPO: Timeout (Final dist: {distance:.1f}m)")
    return np.array(trajectory), float('inf'), False


def follow_path(robot, path, world, start_state, goal_threshold=2.0):
    """
    Simulate robot following a geometric path.
    Simple Pure Pursuit-like controller.
    """
    # Use actual start state (x, y, heading)
    state = np.array(start_state)
    trajectory = [state.copy()]
    
    current_idx = 0
    lookahead_dist = 1.5  # Reduced from 3.0 to avoid cutting corners
    
    max_steps = 1000
    dt = 0.1
    
    for step in range(max_steps):
        # Find target point on path
        # Simple logic: find first point further than lookahead distance
        target_point = path[-1] # Default to end
        
        for i in range(current_idx, len(path)):
            dist = np.hypot(path[i][0] - state[0], path[i][1] - state[1])
            if dist > lookahead_dist:
                target_point = path[i]
                current_idx = i
                break
        
        # Control logic
        dx = target_point[0] - state[0]
        dy = target_point[1] - state[1]
        target_heading = np.arctan2(dy, dx)
        
        heading_error = (target_heading - state[2] + np.pi) % (2 * np.pi) - np.pi
        
        # P-controller for steering
        steer = np.clip(heading_error * 2.0, -robot.max_steering, robot.max_steering)
        
        # Simple speed control: slow down if turning
        target_speed = robot.speed_max
        if abs(heading_error) > np.deg2rad(20):
            target_speed *= 0.5
            
        accel = np.clip((target_speed - robot.speed), -robot.accel_max, robot.accel_max)
        
        # Execute
        action = np.array([accel, steer])
        deriv = robot.deriv(state, action)
        state = state + deriv * dt
        
        trajectory.append(state.copy())
        
        # Check goal
        dist_to_goal = np.hypot(path[-1][0] - state[0], path[-1][1] - state[1])
        if dist_to_goal < goal_threshold:
            # Add final point to trajectory to close the gap for metric calculation
            trajectory.append(np.array([path[-1][0], path[-1][1], state[2]]))
            return np.array(trajectory), True
            
        if not world.is_inside_bounds(state[:2]) or world.check_collision(state[:2]):
            print(f"  Collision at {state[:2]}")
            return np.array(trajectory), False
            
    return np.array(trajectory), False

def compare_dijkstra_vs_ppo(goal=(50.0, 75.0), start_state=[50.0, 10.0, np.deg2rad(90)],
                             ppo_model_path='ppo_model.pth', train_ppo_if_missing=False,
                             train_episodes=1500, dijkstra_debug=True):
    """
    Compare PURE SHORTEST PATHS: Dijkstra (optimal) vs PPO (learned).
    """
    # Create world
    world = WorldFactory.create_random_world(width=100, length=200, num_circles=3, num_rects=2, seed=42)
    start_state = np.array(start_state)
    goal = np.array(goal)
    
    straight_dist = np.hypot(goal[0] - start_state[0], goal[1] - start_state[1])
    
    print("\n" + "="*70)
    print("SHORTEST PATH COMPARISON: Dijkstra vs PPO")
    print("="*70)
    print(f"Start: ({start_state[0]:.1f}, {start_state[1]:.1f})")
    print(f"Goal: ({goal[0]:.1f}, {goal[1]:.1f})")
    print(f"Straight-line distance: {straight_dist:.1f}m")
    print("="*70)
    
    # Check validity
    if world.check_collision(start_state[:2]):
        print("⚠ WARNING: Start position is inside an obstacle!")
    if world.check_collision(goal):
        print("⚠ WARNING: Goal position is inside an obstacle!")
    
    results = {}
    
    # 1. Dijkstra (Optimal Geometric Path)
    print("\n" + "="*60)
    print("Running Dijkstra - Optimal Geometric Path")
    print("="*60)
    # Use robot_radius=2.0 to account for robot size (wheel_base=3)
    dijkstra = DijkstraPlanner(world, grid_size=0.5, robot_radius=2.0, debug=dijkstra_debug)
    dijkstra_result = dijkstra.plan(start_state[:2], goal)
    
    if dijkstra_result and dijkstra_result['success']:
        geometric_path = dijkstra_result['trajectory']
        print(f"✓ Geometric path found: {dijkstra_result['path_length']:.1f}m")
        
        # Simulate robot following this path
        print("  Simulating robot following Dijkstra path...")
        robot = MobileRobot()
        # Pass start_state to ensure correct initial heading
        dijkstra_traj, success = follow_path(robot, geometric_path, world, start_state)
        
        dijkstra_length = compute_path_length(dijkstra_traj)
        
        results['Dijkstra'] = {
            'trajectory': dijkstra_traj,
            'geometric_path': geometric_path,
            'path_length': dijkstra_length,
            'ratio': dijkstra_length / straight_dist,
            'success': success
        }
        print(f"✓ Robot Trajectory: {dijkstra_length:.1f}m (ratio: {dijkstra_length/straight_dist:.2f})")
    else:
        print("✗ Dijkstra failed (No path found)!")
        results['Dijkstra'] = {
            'trajectory': None,
            'geometric_path': None,
            'path_length': float('inf'),
            'ratio': float('inf'),
            'success': False
        }
    
    # 2. PPO (Learned Path)
    print("\n" + "="*60)
    print("Evaluating PPO - Learned Path")
    print("="*60)
    
    ppo_robot = PPORobot()
    try:
        ppo_robot.load_model(ppo_model_path)
        print(f"✓ Loaded model: {ppo_model_path}")
    except:
        print(f"✗ Model not found: {ppo_model_path}")
        results['PPO'] = {
            'trajectory': None,
            'path_length': 0,
            'ratio': 0,
            'success': False
        }
    else:
        ppo_traj, ppo_length, ppo_success = evaluate_ppo(ppo_robot, world, start_state, goal)
        
        results['PPO'] = {
            'trajectory': ppo_traj,
            'path_length': ppo_length,
            'ratio': ppo_length / straight_dist if ppo_length != float('inf') else float('inf'),
            'success': ppo_success
        }
        
        if ppo_success:
            print(f"✓ PPO path: {ppo_length:.1f}m (ratio: {ppo_length/straight_dist:.2f})")
        else:
            print("✗ PPO failed to reach goal")
    
    # Visualization
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Both paths overlaid
    ax = axes[0]
    ax.set_aspect('equal')
    _draw_obstacles(ax, world)
    
    # Straight line
    ax.plot([start_state[0], goal[0]], [start_state[1], goal[1]], 
           'gray', linestyle=':', linewidth=2, alpha=0.5, label=f'Straight ({straight_dist:.1f}m)')
    
    # Dijkstra Geometric (Reference)
    if results['Dijkstra']['geometric_path'] is not None:
        gp = results['Dijkstra']['geometric_path']
        ax.plot(gp[:, 0], gp[:, 1], 'c--', linewidth=2, alpha=0.8, label='Dijkstra (Geometric)')

    # Dijkstra Robot
    if results['Dijkstra']['trajectory'] is not None:
        dt = results['Dijkstra']['trajectory']
        ax.plot(dt[:, 0], dt[:, 1], 'b-', linewidth=2.5, alpha=0.7, 
               label=f"Dijkstra Robot ({results['Dijkstra']['path_length']:.1f}m)")
    
    # PPO
    if results['PPO']['trajectory'] is not None:
        pt = results['PPO']['trajectory']
        label = f"PPO ({results['PPO']['path_length']:.1f}m)" if results['PPO']['success'] else "PPO (Failed)"
        ax.plot(pt[:, 0], pt[:, 1], 'g-', linewidth=2.5, alpha=0.7, label=label)
    
    ax.scatter(start_state[0], start_state[1], c='green', s=200, marker='o', 
              label='Start', zorder=10, edgecolors='black', linewidths=2)
    ax.scatter(goal[0], goal[1], c='gold', s=300, marker='*', 
              label='Goal', zorder=10, edgecolors='black', linewidths=2)
    
    ax.set_xlim(-5, world.width + 5)
    ax.set_ylim(-5, 150)
    ax.set_xlabel('X [m]', fontsize=11)
    ax.set_ylabel('Y [m]', fontsize=11)
    ax.set_title('Path Comparison', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bar chart
    ax = axes[1]
    methods = ['Straight\nLine', 'Dijkstra\n(Robot)', 'PPO\n(Learned)']
    lengths = [straight_dist, results['Dijkstra']['path_length'], results['PPO']['path_length']]
    
    # Handle infinite lengths for plotting
    plot_lengths = [l if l != float('inf') else 0 for l in lengths]
    
    colors = ['gray', 'blue', 'green']
    
    bars = ax.bar(methods, plot_lengths, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, length in zip(bars, lengths):
        height = bar.get_height()
        text = f'{length:.1f}m' if length != float('inf') else 'Failed'
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                text, ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Path Length [m]', fontsize=11)
    ax.set_title('Path Length Comparison', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    if max(plot_lengths) > 0:
        ax.set_ylim(0, max(plot_lengths) * 1.15)
    
    plt.tight_layout()
    plt.savefig('shortest_path_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: shortest_path_comparison.png")
    plt.show()
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS - SHORTEST PATH COMPARISON")
    print("="*70)
    print(f"\n{'Method':<20} {'Path Length':<15} {'vs Straight':<15} {'vs Optimal':<15}")
    print("-" * 70)
    print(f"{'Straight Line':<20} {straight_dist:<15.1f} {'1.00x':<15} {'-':<15}")
    
    d_len = results['Dijkstra']['path_length']
    d_ratio = results['Dijkstra']['ratio']
    d_str = f"{d_len:<15.1f}" if d_len != float('inf') else "Failed         "
    d_rat_str = f"{d_ratio:<15.2f}" if d_len != float('inf') else "-              "
    print(f"{'Dijkstra (Robot)':<20} {d_str} {d_rat_str} {'1.00x':<15}")
    
    p_len = results['PPO']['path_length']
    p_ratio = results['PPO']['ratio']
    p_str = f"{p_len:<15.1f}" if p_len != float('inf') else "Failed         "
    p_rat_str = f"{p_ratio:<15.2f}" if p_len != float('inf') else "-              "
    
    vs_opt = "-"
    if p_len != float('inf') and d_len != float('inf') and d_len > 0:
        vs_opt = f"{p_len / d_len:<15.2f}"
        
    print(f"{'PPO (Learned)':<20} {p_str} {p_rat_str} {vs_opt}")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = compare_dijkstra_vs_ppo(
        goal=(50.0, 75.0),
        start_state=[2.0, 2.0, np.deg2rad(90)],
        ppo_model_path='ppo_model.pth',
        dijkstra_debug=True
    )

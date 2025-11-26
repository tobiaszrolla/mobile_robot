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


def evaluate_ppo(ppo_robot, world, start_state, goal, max_steps=300):
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


def compare_dijkstra_vs_ppo(goal=(50.0, 75.0), start_state=[50.0, 10.0, np.deg2rad(90)],
                             ppo_model_path='ppo_model.pth', train_ppo_if_missing=False,
                             train_episodes=1500, dijkstra_debug=True):
    """
    Compare PURE SHORTEST PATHS: Dijkstra (optimal) vs PPO (learned).
    """
    # Create world
    # Use the same seed logic as training to ensure we get the same "safe" world if possible
    # But for comparison, we'll just use the fixed seed 42 and hope it's safe with the new start
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
    
    results = {}
    
    # 1. Dijkstra (Optimal Shortest Path)
    print("\n" + "="*60)
    print("Running Dijkstra - Optimal Shortest Path")
    print("="*60)
    dijkstra = DijkstraPlanner(world, goal, debug=dijkstra_debug)
    dijkstra_result = dijkstra.plan(start_state, max_iterations=20000)
    
    if dijkstra_result['success']:
        dijkstra_traj = dijkstra_result['trajectory']
        # Re-compute length from actual trajectory points for fair comparison
        dijkstra_length = compute_path_length(dijkstra_traj)
        
        results['Dijkstra'] = {
            'trajectory': dijkstra_traj,
            'path_length': dijkstra_length,
            'ratio': dijkstra_length / straight_dist,
            'success': True
        }
        print(f"✓ Dijkstra path: {dijkstra_length:.1f}m (ratio: {dijkstra_length/straight_dist:.2f})")
    else:
        print("✗ Dijkstra failed!")
        return None
    
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
        if train_ppo_if_missing:
            print("  Training new model...")
            # Import and train here if needed
            print("  ERROR: Training not implemented in this script")
            print("  Run: python train_ppo.py first")
        return None
    
    ppo_traj, ppo_length, ppo_success = evaluate_ppo(ppo_robot, world, start_state, goal)
    
    if ppo_success:
        results['PPO'] = {
            'trajectory': ppo_traj,
            'path_length': ppo_length,
            'ratio': ppo_length / straight_dist,
            'success': True
        }
        print(f"✓ PPO path: {ppo_length:.1f}m (ratio: {ppo_length/straight_dist:.2f})")
    else:
        print("✗ PPO failed to reach goal")
        results['PPO'] = {'success': False}
    
    # Comparison
    if not all(results[k]['success'] for k in results):
        print("\n✗ Cannot compare - one method failed")
        return results
    
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
    
    # Dijkstra
    ax.plot(results['Dijkstra']['trajectory'][:, 0], 
           results['Dijkstra']['trajectory'][:, 1],
           'b-', linewidth=2.5, alpha=0.7, 
           label=f"Dijkstra ({results['Dijkstra']['path_length']:.1f}m)")
    
    # PPO
    ax.plot(results['PPO']['trajectory'][:, 0], 
           results['PPO']['trajectory'][:, 1],
           'g-', linewidth=2.5, alpha=0.7, 
           label=f"PPO ({results['PPO']['path_length']:.1f}m)")
    
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
    methods = ['Straight\nLine', 'Dijkstra\n(Optimal)', 'PPO\n(Learned)']
    lengths = [straight_dist, results['Dijkstra']['path_length'], results['PPO']['path_length']]
    colors = ['gray', 'blue', 'green']
    
    bars = ax.bar(methods, lengths, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, length in zip(bars, lengths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{length:.1f}m', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Path Length [m]', fontsize=11)
    ax.set_title('Path Length Comparison', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(lengths) * 1.15)
    
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
    print(f"{'Dijkstra (Optimal)':<20} {results['Dijkstra']['path_length']:<15.1f} "
          f"{results['Dijkstra']['ratio']:<15.2f} {'1.00x':<15}")
    print(f"{'PPO (Learned)':<20} {results['PPO']['path_length']:<15.1f} "
          f"{results['PPO']['ratio']:<15.2f} "
          f"{results['PPO']['path_length'] / results['Dijkstra']['path_length']:<15.2f}")
    
    ppo_vs_optimal = results['PPO']['path_length'] / results['Dijkstra']['path_length']
    
    print("\n" + "="*70)
    print(f"PPO vs Dijkstra: {ppo_vs_optimal:.2f}x")
    
    if ppo_vs_optimal <= 1.10:
        print("✓ EXCELLENT: PPO matches optimal (≤10% longer)")
    elif ppo_vs_optimal <= 1.25:
        print("✓ GOOD: PPO is near-optimal (10-25% longer)")
    elif ppo_vs_optimal <= 1.50:
        print("⚠ FAIR: PPO is reasonable (25-50% longer)")
    else:
        print("✗ POOR: PPO path is much longer (>50% vs optimal)")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = compare_dijkstra_vs_ppo(
        goal=(50.0, 75.0),
        start_state=[50.0, 10.0, np.deg2rad(90)],
        ppo_model_path='ppo_model.pth',
        dijkstra_debug=True
    )

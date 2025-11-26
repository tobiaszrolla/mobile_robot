import numpy as np
import heapq
from MobileRobot import MobileRobot
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from world.obstacles import CircularObstacle, RectangularObstacle
import itertools
from collections import defaultdict
import time


class DijkstraPlanner:
    """
    PROPER Dijkstra implementation:
    1. Build complete reachable graph
    2. Run Dijkstra to find shortest path
    """
    
    def __init__(self, world, goal, debug=False):
        """
        Initialize Dijkstra planner.
        
        Args:
            world: World object with obstacles
            goal: Goal position [x, y]
            debug: Enable debug visualization
        """
        self.world = world
        self.goal = np.array(goal)
        self.debug = debug
        
        # Robot for dynamics
        self.robot = MobileRobot()
        
        # Action space
        self.accelerations = np.array([self.robot.accel_max])
        self.steering_angles = np.array([0, -np.deg2rad(5), np.deg2rad(5), 
                                         -np.deg2rad(15), np.deg2rad(15),
                                         -np.deg2rad(25), np.deg2rad(25)])
        
        # Time step
        self.dt = 0.1
        self.action_duration = 1.0
        self.steps_per_action = int(self.action_duration / self.dt)
        
        # State discretization (creates the graph nodes)
        self.grid_size = 2.0  # 2 meter grid
        self.heading_bins = 32  # 11.25° bins
        self.speed_bins = 6    # Speed discretization
        
        print(f"Dijkstra Planner - PROPER Implementation")
        print(f"  Phase 1: Build complete reachable graph")
        print(f"  Phase 2: Run Dijkstra on graph")
        print(f"  Grid: {self.grid_size}m, Heading: {self.heading_bins} bins, Speed: {self.speed_bins} bins")
        print(f"  Actions per state: {len(self.accelerations) * len(self.steering_angles)}")
        
        # Graph structure: adjacency list
        # graph[state] = [(next_state, edge_distance, trajectory)]
        self.graph = defaultdict(list)
        self.state_to_continuous = {}  # Maps discrete -> continuous state
        
        # Debug
        if self.debug:
            self.explored_states = []
    
    def discretize(self, state, speed):
        """
        Discretize continuous state to graph node.
        Returns a hashable tuple representing the discrete state.
        """
        x, y, heading = state
        
        x_bin = int(round(x / self.grid_size))
        y_bin = int(round(y / self.grid_size))
        heading_bin = int((heading % (2 * np.pi)) / (2 * np.pi) * self.heading_bins) % self.heading_bins
        speed_bin = min(int(speed / self.robot.speed_max * self.speed_bins), self.speed_bins - 1)
        
        return (x_bin, y_bin, heading_bin, speed_bin)
    
    def is_goal_state(self, discrete_state, threshold=2.0):
        """Check if discrete state is near goal."""
        if discrete_state not in self.state_to_continuous:
            return False
        
        cont_state, _ = self.state_to_continuous[discrete_state]
        distance = np.hypot(self.goal[0] - cont_state[0], self.goal[1] - cont_state[1])
        return distance < threshold
    
    def get_successors(self, state, current_speed):
        """
        Get all valid successor states from current state.
        Returns list of (next_state, next_speed, edge_distance, trajectory).
        """
        successors = []
        
        for accel in self.accelerations:
            for steer in self.steering_angles:
                action = np.array([accel, steer])
                
                # Simulate action
                temp_robot = MobileRobot()
                temp_robot.speed = current_speed
                temp_state = state.copy()
                
                collision = False
                trajectory = [temp_state.copy()]
                
                for step in range(self.steps_per_action):
                    deriv = temp_robot.deriv(temp_state, action)
                    temp_state = temp_state + deriv * self.dt
                    
                    # Check collision
                    if self.world.check_collision(temp_state[:2]):
                        collision = True
                        break
                    
                    # Check bounds
                    if not self.world.is_inside_bounds(temp_state[:2]):
                        collision = True
                        break
                    
                    trajectory.append(temp_state.copy())
                
                if collision:
                    continue
                
                # Compute edge distance (Euclidean)
                edge_distance = np.hypot(temp_state[0] - state[0], temp_state[1] - state[1])
                
                # Cost is purely distance traveled
                edge_weight = edge_distance
                
                temp_state[2] = temp_state[2] % (2 * np.pi)
                final_speed = temp_robot.speed
                
                successors.append((temp_state, final_speed, edge_weight, trajectory))
        
        return successors
    
    def build_graph(self, start_state, start_speed):
        """
        Phase 1: Build the complete reachable state graph using BFS.
        NO iteration limit - explores until no more states to explore!
        """
        print("\n" + "="*70)
        print("PHASE 1: BUILDING REACHABLE STATE GRAPH")
        print("="*70)
        
        start_time = time.time()
        
        start_discrete = self.discretize(start_state, start_speed)
        self.state_to_continuous[start_discrete] = (start_state.copy(), start_speed)
        
        # BFS queue: (discrete_state, continuous_state, speed)
        queue = [(start_discrete, start_state.copy(), start_speed)]
        visited = {start_discrete}
        
        iteration = 0
        
        while queue:
            current_discrete, current_cont, current_speed = queue.pop(0)
            
            if self.debug and iteration % 100 == 0:
                self.explored_states.append(current_cont[:2])
            
            # Get all successors
            successors = self.get_successors(current_cont, current_speed)
            
            for next_cont, next_speed, edge_distance, trajectory in successors:
                next_discrete = self.discretize(next_cont, next_speed)
                
                # Add edge to graph
                self.graph[current_discrete].append((next_discrete, edge_distance, trajectory))
                
                # If not visited, add to queue
                if next_discrete not in visited:
                    visited.add(next_discrete)
                    self.state_to_continuous[next_discrete] = (next_cont.copy(), next_speed)
                    queue.append((next_discrete, next_cont.copy(), next_speed))
            
            iteration += 1
            
            if iteration % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  Iteration {iteration:6d} | States: {len(visited):6d} | "
                      f"Edges: {sum(len(v) for v in self.graph.values()):8d} | "
                      f"Queue: {len(queue):6d} | Time: {elapsed:6.1f}s")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("GRAPH BUILD COMPLETE!")
        print(f"{'='*70}")
        print(f"Total states (nodes):     {len(self.state_to_continuous):8d}")
        print(f"Total edges:              {sum(len(v) for v in self.graph.values()):8d}")
        print(f"Iterations:               {iteration:8d}")
        print(f"Time:                     {elapsed:8.1f}s")
        print(f"{'='*70}\n")
        
        return start_discrete
    
    def dijkstra_search(self, start_discrete):
        """
        Phase 2: Run Dijkstra's algorithm on the built graph.
        Finds shortest path from start to any goal state.
        """
        print("\n" + "="*70)
        print("PHASE 2: DIJKSTRA SHORTEST PATH SEARCH")
        print("="*70)
        
        start_time = time.time()
        
        # Find all goal states in the graph
        goal_states = [state for state in self.state_to_continuous.keys() 
                      if self.is_goal_state(state)]
        
        if not goal_states:
            print("✗ No goal states found in reachable graph!")
            return None
        
        print(f"Found {len(goal_states)} goal states in graph")
        
        # Dijkstra's algorithm
        # Priority queue: (distance, counter, discrete_state, parent, trajectory_from_parent)
        counter = itertools.count()
        pq = [(0.0, next(counter), start_discrete, None, [])]
        
        # Track shortest distances and paths
        distances = {start_discrete: 0.0}
        parents = {start_discrete: (None, [])}  # (parent_state, trajectory)
        visited = set()
        
        iteration = 0
        
        while pq:
            current_dist, _, current_state, parent, traj_from_parent = heapq.heappop(pq)
            
            if current_state in visited:
                continue
            
            visited.add(current_state)
            
            # Update parent
            if parent is not None:
                parents[current_state] = (parent, traj_from_parent)
            
            # Check if goal
            if current_state in goal_states:
                elapsed = time.time() - start_time
                print(f"\n✓ GOAL REACHED!")
                print(f"  Shortest distance: {current_dist:.2f}m")
                print(f"  Dijkstra iterations: {iteration}")
                print(f"  Time: {elapsed:.1f}s")
                
                # Reconstruct path
                path = self.reconstruct_path(start_discrete, current_state, parents)
                
                return {
                    'trajectory': np.array(path),
                    'path_length': current_dist,
                    'success': True,
                    'dijkstra_iterations': iteration,
                    'states_visited': len(visited)
                }
            
            # Expand neighbors
            for next_state, edge_dist, trajectory in self.graph[current_state]:
                new_dist = current_dist + edge_dist
                
                if next_state not in distances or new_dist < distances[next_state]:
                    distances[next_state] = new_dist
                    heapq.heappush(pq, (new_dist, next(counter), next_state, current_state, trajectory))
            
            iteration += 1
            
            if iteration % 10000 == 0:
                elapsed = time.time() - start_time
                best_goal_dist = min([distances.get(g, float('inf')) for g in goal_states])
                print(f"  Iteration {iteration:8d} | Visited: {len(visited):6d} | "
                      f"Queue: {len(pq):6d} | Best goal: {best_goal_dist:7.1f}m | Time: {elapsed:6.1f}s")
        
        # No path found
        elapsed = time.time() - start_time
        print(f"\n✗ No path to goal found!")
        print(f"  States visited: {len(visited)}")
        print(f"  Time: {elapsed:.1f}s")
        
        return {
            'trajectory': None,
            'path_length': float('inf'),
            'success': False,
            'dijkstra_iterations': iteration,
            'states_visited': len(visited)
        }
    
    def reconstruct_path(self, start_state, goal_state, parents):
        """Reconstruct continuous path from discrete states and trajectories."""
        # Backtrack through parents
        discrete_path = []
        current = goal_state
        
        while current is not None:
            discrete_path.append(current)
            parent, _ = parents[current]
            current = parent
        
        discrete_path.reverse()
        
        # Build continuous path with trajectories
        continuous_path = []
        
        for i in range(len(discrete_path)):
            state_discrete = discrete_path[i]
            
            if i == 0:
                # Start state
                cont_state, _ = self.state_to_continuous[state_discrete]
                continuous_path.append(cont_state)
            else:
                # Get trajectory from parent
                _, trajectory = parents[state_discrete]
                continuous_path.extend(trajectory[1:])  # Skip first point (already added)
        
        return continuous_path
    
    def plan(self, start_state, max_iterations=None):
        """
        Complete Dijkstra planning:
        1. Build reachable graph (exhaustive)
        2. Run Dijkstra on graph
        
        max_iterations is IGNORED - explores everything!
        """
        if max_iterations is not None:
            print(f"Note: max_iterations={max_iterations} is IGNORED")
            print("      This implementation explores ALL reachable states!\n")
        
        start_state = np.array(start_state)
        start_state[2] = start_state[2] % (2 * np.pi)
        start_speed = 0.0
        
        straight_dist = np.hypot(self.goal[0] - start_state[0], self.goal[1] - start_state[1])
        print(f"\nGoal: ({self.goal[0]:.1f}, {self.goal[1]:.1f})")
        print(f"Straight-line distance: {straight_dist:.1f}m")
        
        total_start = time.time()
        
        # Phase 1: Build graph
        start_discrete = self.build_graph(start_state, start_speed)
        
        # Phase 2: Dijkstra search
        result = self.dijkstra_search(start_discrete)
        
        total_elapsed = time.time() - total_start
        
        if result['success']:
            path = result['trajectory']
            path_length = result['path_length']
            
            print(f"\n{'='*70}")
            print("FINAL RESULTS")
            print(f"{'='*70}")
            print(f"Status:               SUCCESS")
            print(f"Path length:          {path_length:.2f}m")
            print(f"Straight line:        {straight_dist:.2f}m")
            print(f"Ratio:                {path_length/straight_dist:.2f}x")
            print(f"Waypoints:            {len(path)}")
            print(f"Total time:           {total_elapsed:.1f}s")
            print(f"{'='*70}")
            
            if self.debug:
                self.visualize_result(path, path_length, straight_dist)
            
            return result
        else:
            print(f"\n{'='*70}")
            print("FINAL RESULTS")
            print(f"{'='*70}")
            print(f"Status:               FAILED")
            print(f"Reason:               Goal unreachable from start")
            print(f"States explored:      {result['states_visited']}")
            print(f"Total time:           {total_elapsed:.1f}s")
            print(f"{'='*70}")
            
            return result
    
    def visualize_result(self, path, path_length, straight_dist):
        """Visualize the result."""
        if not self.debug:
            return
        
        fig, ax = plt.subplots(figsize=(10, 14))
        ax.set_aspect('equal')
        
        # Draw obstacles
        for obs in self.world.obstacles:
            if isinstance(obs, CircularObstacle):
                circle = Circle(obs.position, obs.radius, color='red', alpha=0.5)
                ax.add_patch(circle)
            elif isinstance(obs, RectangularObstacle):
                rect = Rectangle(obs.position, obs.width, obs.height, color='red', alpha=0.5)
                ax.add_patch(rect)
        
        # Draw explored states
        if self.explored_states:
            explored_array = np.array(self.explored_states)
            ax.scatter(explored_array[:, 0], explored_array[:, 1], 
                      c='lightblue', s=3, alpha=0.3, label='Explored states')
        
        # Draw path
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=3, 
               label=f'Dijkstra Path ({path_length:.1f}m)', zorder=5)
        
        # Straight line
        ax.plot([path[0][0], self.goal[0]], [path[0][1], self.goal[1]], 
               'g--', linewidth=2, alpha=0.5, label=f'Straight ({straight_dist:.1f}m)')
        
        # Orientation arrows
        for i in range(0, len(path), max(1, len(path)//15)):
            x, y, heading = path[i]
            dx = np.cos(heading) * 2
            dy = np.sin(heading) * 2
            ax.arrow(x, y, dx, dy, head_width=1.5, color='darkblue', alpha=0.7)
        
        # Start and goal
        ax.scatter(path[0][0], path[0][1], c='green', s=200, marker='o', 
                  label='Start', zorder=10, edgecolors='black', linewidths=2)
        ax.scatter(self.goal[0], self.goal[1], c='gold', s=300, marker='*', 
                  label='Goal', zorder=10, edgecolors='black', linewidths=2)
        
        ax.set_xlim(-5, self.world.width + 5)
        ax.set_ylim(-5, self.world.lenght + 5)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'Dijkstra Shortest Path\nRatio: {path_length/straight_dist:.2f}x, Waypoints: {len(path)}',
                    fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dijkstra_shortest_path.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved: dijkstra_shortest_path.png")
        plt.show()

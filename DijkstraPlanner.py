import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from world.obstacles import CircularObstacle, RectangularObstacle
import time


class DijkstraPlanner:
    """
    Standard 2D Grid-based Dijkstra Planner.
    Ignores robot dynamics, finds shortest geometric path.
    """
    
    def __init__(self, world, grid_size=0.5, robot_radius=1.0, debug=False):
        """
        Initialize Planner.
        
        Args:
            world: World object with obstacles
            grid_size: Size of grid cells in meters (smaller = more optimal path)
            robot_radius: Safety margin for collision checking
            debug: Enable debug visualization
        """
        self.world = world
        self.grid_size = grid_size
        self.robot_radius = robot_radius
        self.debug = debug
        
        # 8-connected grid (x, y, cost)
        # Cost is Euclidean distance
        self.motions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),  # Cardinal
            (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)  # Diagonal
        ]
        
    def _to_grid(self, pos):
        return (int(round(pos[0] / self.grid_size)), 
                int(round(pos[1] / self.grid_size)))
    
    def _to_world(self, grid_pos):
        return np.array([grid_pos[0] * self.grid_size, 
                         grid_pos[1] * self.grid_size])

    def _check_collision(self, point):
        """Check collision with robot radius margin."""
        # Check bounds
        if not self.world.is_inside_bounds(point):
            return True
            
        # Check obstacles with margin
        for obs in self.world.obstacles:
            if isinstance(obs, CircularObstacle):
                if np.linalg.norm(point - obs.position) <= (obs.radius + self.robot_radius):
                    return True
            elif isinstance(obs, RectangularObstacle):
                # Expand rectangle by robot_radius
                x, y = point
                bx, by = obs.position
                if (bx - self.robot_radius <= x <= bx + obs.width + self.robot_radius and 
                    by - self.robot_radius <= y <= by + obs.height + self.robot_radius):
                    return True
        return False

    def plan(self, start_pos, goal_pos):
        """
        Plan shortest path using Theta* (Any-Angle Path Planning).
        Finds true Euclidean shortest path on the grid.
        """
        start_node = self._to_grid(start_pos)
        goal_node = self._to_grid(goal_pos)
        
        print(f"Planning path (Theta*): {start_pos} -> {goal_pos}")
        
        # Priority queue: (cost, current_node)
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        
        came_from = {start_node: start_node} # Parent of start is start
        cost_so_far = {start_node: 0}
        
        visited = set()
        nodes_expanded = 0
        
        while open_set:
            current_cost, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            nodes_expanded += 1
            
            # Check if reached goal
            if current == goal_node:
                print(f"Goal reached! Path found. Nodes expanded: {nodes_expanded}")
                return self._reconstruct_path(came_from, current, start_node)
            
            # Expand neighbors
            for dx, dy, _ in self.motions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds and collision for the node itself
                world_pos = self._to_world(neighbor)
                
                if self._check_collision(world_pos):
                    continue
                
                # Theta* Logic: Check Line-of-Sight from Parent(current) to Neighbor
                parent = came_from[current]
                
                # Optimization: If parent is current (start node), treat normally
                if parent == current:
                    # Standard update
                    dist = np.hypot(neighbor[0]-current[0], neighbor[1]-current[1]) * self.grid_size
                    new_cost = cost_so_far[current] + dist
                    potential_parent = current
                else:
                    # Check LOS from parent to neighbor
                    parent_pos = self._to_world(parent)
                    neighbor_pos = self._to_world(neighbor)
                    
                    if self._is_line_safe(parent_pos, neighbor_pos):
                        # Path 2: Parent -> Neighbor (Skip current)
                        dist = np.hypot(neighbor_pos[0]-parent_pos[0], neighbor_pos[1]-parent_pos[1])
                        new_cost = cost_so_far[parent] + dist
                        potential_parent = parent
                    else:
                        # Path 1: Parent -> Current -> Neighbor (Standard)
                        dist = np.hypot(neighbor[0]-current[0], neighbor[1]-current[1]) * self.grid_size
                        new_cost = cost_so_far[current] + dist
                        potential_parent = current
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = potential_parent
                    heapq.heappush(open_set, (new_cost, neighbor))
        
        print("No path found!")
        return None

    def _reconstruct_path(self, came_from, current, start):
        path = []
        # Reconstruct path by following parents
        # In Theta*, parents can be far away!
        while current != start:
            path.append(self._to_world(current))
            current = came_from[current]
        path.append(self._to_world(start))
        path.reverse()
        
        return {
            'trajectory': np.array(path),
            'path_length': sum(np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) for i in range(len(path)-1)),
            'success': True
        }

    def _is_line_safe(self, p1, p2):
        """Check if straight line between p1 and p2 is collision-free."""
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = int(dist / (self.grid_size / 2)) + 1
        
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            
            if self._check_collision(np.array([x, y])):
                return False
        return True


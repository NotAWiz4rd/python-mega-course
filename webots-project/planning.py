"""
Planning module implementing RRT (Rapidly-exploring Random Tree) path planning.

This module provides the Planning behaviour class that uses RRT to compute
collision-free paths through the environment's configuration space.
"""

import numpy as np
import py_trees


def world2map(x_world, y_world):
    """
    Convert world coordinates to map/pixel coordinates.

    The cspace is stored in [y, x] (row, col) convention with shape (300, 200).
    Returns [y_map, x_map] to match this indexing.

    Args:
        x_world: X coordinate in world frame
        y_world: Y coordinate in world frame

    Returns:
        List [y_map, x_map] with clamped pixel coordinates for [row, col] indexing
    """
    x_map = int((x_world + 2.15) / 4.3 * 199)
    y_map = int(-(y_world - 1.66) / 5.58 * 299)

    # Clamp values to valid map range
    x_map = max(0, min(199, x_map))
    y_map = max(0, min(299, y_map))

    # Return [y, x] for standard row, col indexing
    return [y_map, x_map]


def map2world(y_map, x_map):
    """
    Convert map/pixel coordinates back to world coordinates.

    Inverse of world2map function. Takes [y_map, x_map] to match cspace indexing.

    Args:
        y_map: Row coordinate in map frame (0-299)
        x_map: Column coordinate in map frame (0-199)

    Returns:
        Tuple (x_world, y_world) in world coordinates
    """
    x_world = (x_map / 199) * 4.3 - 2.15
    y_world = -(y_map / 299) * 5.58 + 1.66
    return (x_world, y_world)


class RRT:
    """
    Rapidly-exploring Random Tree implementation for path planning.

    RRT builds a tree by randomly sampling points in the configuration space
    and extending the tree towards those samples. This allows efficient
    exploration of high-dimensional spaces while avoiding obstacles.

    Attributes:
        start: Starting position in map coordinates [x, y]
        goal: Goal position in map coordinates [x, y]
        cspace: Configuration space array where values > threshold are obstacles
        obstacle_threshold: Value above which a cell is considered an obstacle
        step_size: Maximum distance to extend tree in one iteration
        max_iterations: Maximum number of RRT iterations
        goal_sample_rate: Probability of sampling the goal directly (bias)
    """

    def __init__(self, start, goal, cspace, obstacle_threshold=0.9,
                 step_size=10, max_iterations=5000, goal_sample_rate=0.1):
        """
        Initialize the RRT planner.

        Args:
            start: Start position [x, y] in map coordinates
            goal: Goal position [x, y] in map coordinates
            cspace: 2D numpy array representing configuration space
            obstacle_threshold: Cells with value > this are obstacles
            step_size: Max extension distance per iteration
            max_iterations: Maximum iterations before giving up
            goal_sample_rate: Probability of sampling goal (0-1)
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.cspace = cspace
        self.obstacle_threshold = obstacle_threshold
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate

        # Tree structure: list of nodes where each node is [x, y]
        # Parent indices stored separately for path reconstruction
        self.nodes = [self.start]
        self.parent = [0]  # Root node is its own parent

    def get_random_sample(self):
        """
        Generate a random sample point in the configuration space.

        With probability goal_sample_rate, returns the goal position
        to bias the tree growth towards the goal.

        Returns:
            numpy array [x, y] of sampled position
        """
        if np.random.random() < self.goal_sample_rate:
            return self.goal

        # Sample uniformly within map bounds
        x = np.random.randint(0, self.cspace.shape[0])
        y = np.random.randint(0, self.cspace.shape[1])
        return np.array([x, y])

    def find_nearest_node(self, sample):
        """
        Find the node in the tree closest to the sample point.

        Args:
            sample: Position [x, y] to find nearest node to

        Returns:
            Index of the nearest node in self.nodes
        """
        distances = [np.linalg.norm(node - sample) for node in self.nodes]
        return np.argmin(distances)

    def steer(self, from_node, to_point):
        """
        Compute a new point by moving from from_node towards to_point.

        If the distance is less than step_size, returns to_point directly.
        Otherwise returns a point step_size distance away in that direction.

        Args:
            from_node: Starting position [x, y]
            to_point: Target position [x, y]

        Returns:
            New position [x, y] after steering
        """
        direction = to_point - from_node
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            return to_point

        # Normalize and scale by step size
        direction = direction / distance * self.step_size
        return from_node + direction

    def is_collision_free(self, from_point, to_point):
        """
        Check if the path between two points is collision-free.

        Uses line interpolation to check intermediate points along
        the path against the configuration space obstacles.

        Args:
            from_point: Starting position [x, y]
            to_point: Ending position [x, y]

        Returns:
            True if path is collision-free, False otherwise
        """
        # Check if target point is within bounds
        if (to_point[0] < 0 or to_point[0] >= self.cspace.shape[0] or
            to_point[1] < 0 or to_point[1] >= self.cspace.shape[1]):
            return False

        # Interpolate points along the line
        distance = np.linalg.norm(to_point - from_point)
        num_checks = max(int(distance), 2)

        for i in range(num_checks + 1):
            t = i / num_checks
            point = from_point + t * (to_point - from_point)
            x, y = int(point[0]), int(point[1])

            # Bounds check
            if x < 0 or x >= self.cspace.shape[0] or y < 0 or y >= self.cspace.shape[1]:
                return False

            # Obstacle check
            if self.cspace[x, y] > self.obstacle_threshold:
                return False

        return True

    def is_goal_reached(self, node):
        """
        Check if a node is close enough to the goal.

        Args:
            node: Position [x, y] to check

        Returns:
            True if within step_size of goal, False otherwise
        """
        return np.linalg.norm(node - self.goal) <= self.step_size

    def extract_path(self, goal_index):
        """
        Extract the path from start to goal by tracing parent pointers.

        Args:
            goal_index: Index of the goal node in self.nodes

        Returns:
            List of [x, y] positions from start to goal
        """
        path = []
        current = goal_index

        while current != 0:
            path.append(self.nodes[current])
            current = self.parent[current]

        path.append(self.nodes[0])  # Add start
        path.reverse()
        return path

    def plan(self):
        """
        Execute the RRT algorithm to find a path from start to goal.

        Returns:
            List of [x, y] positions representing the path, or None if no path found
        """
        for iteration in range(self.max_iterations):
            # Sample a random point
            sample = self.get_random_sample()

            # Find nearest node in tree
            nearest_idx = self.find_nearest_node(sample)
            nearest_node = self.nodes[nearest_idx]

            # Steer towards sample
            new_node = self.steer(nearest_node, sample)

            # Check for collisions
            if self.is_collision_free(nearest_node, new_node):
                # Add node to tree
                self.nodes.append(new_node)
                self.parent.append(nearest_idx)

                # Check if we can reach the goal from this new node
                if self.is_goal_reached(new_node):
                    if self.is_collision_free(new_node, self.goal):
                        # Add goal to tree
                        self.nodes.append(self.goal)
                        self.parent.append(len(self.nodes) - 2)
                        print(f"RRT: Path found after {iteration + 1} iterations")
                        return self.extract_path(len(self.nodes) - 1)

        print(f"RRT: No path found after {self.max_iterations} iterations")
        return None


class Planning(py_trees.behaviour.Behaviour):
    """
    Behaviour tree node for path planning using RRT.

    This behaviour computes a collision-free path from the robot's current
    position to a specified goal using the RRT algorithm. The computed
    waypoints are stored in the blackboard for the Navigation behaviour to use.

    Attributes:
        blackboard: Shared data storage for communication between behaviours
        goal_world: Goal position in world coordinates (x, y)
        robot: Reference to the Webots robot/supervisor instance
    """

    def __init__(self, name: str, blackboard: py_trees.blackboard.Blackboard, goal_world: tuple):
        """
        Initialize the Planning behaviour.

        Args:
            name: Name of this behaviour node
            blackboard: Shared blackboard for inter-behaviour communication
            goal_world: Target position as (x, y) tuple in world coordinates
        """
        super(Planning, self).__init__(name)
        self.blackboard = blackboard
        self.goal_world = goal_world
        self.robot = blackboard.get('robot')

    def setup(self):
        """
        Set up sensors needed for planning (GPS for current position).
        """
        self.timestep = int(self.robot.getBasicTimeStep())

        # GPS for getting current robot position
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

        self.logger.debug(f"Planning setup complete for goal {self.goal_world}")

    def initialise(self):
        """
        Initialize the planning process.

        Loads the configuration space from file created by the Mapping behaviour.
        """
        self.logger.debug("Planning initialise")
        self.path_computed = False

    def update(self):
        """
        Execute the RRT planning algorithm.

        Computes a path from current position to goal, simplifies it,
        converts waypoints to world coordinates, and stores them in blackboard.

        Returns:
            SUCCESS if path found and stored, FAILURE if no path exists
        """
        if self.path_computed:
            return py_trees.common.Status.SUCCESS

        # Load configuration space (created by Mapping behaviour)
        try:
            cspace = np.load('cspace.npy')
        except FileNotFoundError:
            self.logger.error("Configuration space file 'cspace.npy' not found!")
            return py_trees.common.Status.FAILURE

        # Get current robot position
        x_world = self.gps.getValues()[0]
        y_world = self.gps.getValues()[1]

        # Convert positions to map coordinates
        start_map = world2map(x_world, y_world)
        goal_map = world2map(self.goal_world[0], self.goal_world[1])

        self.logger.info(f"Planning path from {start_map} to {goal_map}")
        print(f"Planning: Start (world): ({x_world:.2f}, {y_world:.2f}) -> (map): {start_map}")
        print(f"Planning: Goal (world): {self.goal_world} -> (map): {goal_map}")

        # Run RRT planning
        rrt = RRT(
            start=start_map,
            goal=goal_map,
            cspace=cspace,
            obstacle_threshold=0.9,
            step_size=10,
            max_iterations=5000,
            goal_sample_rate=0.15
        )

        path = rrt.plan()

        if path is None:
            self.logger.error("RRT failed to find a path!")
            return py_trees.common.Status.FAILURE

        # Convert path from map coordinates back to world coordinates
        waypoints_world = [map2world(p[0], p[1]) for p in path]

        # Store waypoints in blackboard for Navigation to use
        self.blackboard.set('waypoints', waypoints_world)

        print(f"Planning: Computed {len(waypoints_world)} waypoints:")
        for i, wp in enumerate(waypoints_world):
            print(f"  {i}: ({wp[0]:.2f}, {wp[1]:.2f})")

        self.path_computed = True
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        """
        Clean up when the behaviour terminates.

        Args:
            new_status: The status the behaviour is transitioning to
        """
        self.logger.debug(f"Planning terminated with status {new_status}")

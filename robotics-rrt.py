import matplotlib.pyplot as plt
import numpy as np


def sample_random_point(map_shape):
    """Sample a random point in the configuration space."""
    random_i = np.random.randint(0, map_shape[0])
    random_j = np.random.randint(0, map_shape[1])
    return (random_i, random_j)


def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def find_nearest_node(tree, sampled_point):
    """Find the nearest node in the tree to the sampled point."""
    nearest_node = None
    min_distance = float('inf')

    for node in tree.keys():
        distance = euclidean_distance(node, sampled_point)
        if distance < min_distance:
            min_distance = distance
            nearest_node = node

    return nearest_node


def steer(from_node, to_point, step_size):
    """
    Create a new node by extending from from_node toward to_point.
    The extension is limited by step_size.
    """
    distance = euclidean_distance(from_node, to_point)

    if distance <= step_size:
        return to_point

    # Extend in the direction of to_point, but only by step_size
    ratio = step_size / distance
    new_i = int(from_node[0] + ratio * (to_point[0] - from_node[0]))
    new_j = int(from_node[1] + ratio * (to_point[1] - from_node[1]))

    return (new_i, new_j)


def is_goal_reached(node, goal, goal_tolerance):
    """Check if the node is close enough to the goal."""
    return euclidean_distance(node, goal) < goal_tolerance


def reconstruct_path(tree, start, goal_node):
    """Reconstruct the path from start to goal using the parent relationships."""
    path = [goal_node]
    current_node = goal_node

    # Trace back through parents until we reach the start
    while current_node != start:
        parent = tree[current_node]
        if parent is None:
            break
        path.append(parent)
        current_node = parent

    return path[::-1]  # Reverse to get start-to-goal order


def plot_node(node, color, marker='o', size=3):
    """Plot a single node on the map."""
    plt.plot(node[1], node[0], color=color, marker=marker, markersize=size)
    plt.draw()
    plt.pause(0.001)


def plot_edge(from_node, to_node, color='cyan', linewidth=0.5):
    """Plot an edge between two nodes."""
    plt.plot([from_node[1], to_node[1]], [from_node[0], to_node[0]],
             color=color, linewidth=linewidth, alpha=0.6)
    plt.draw()
    plt.pause(0.001)


def rrt_search(map, start, goal, max_iterations=5000, step_size=10, goal_tolerance=15):
    """
    RRT (Rapidly-exploring Random Tree) path planning algorithm.

    Args:
        map: The environment map
        start: Starting position (i, j)
        goal: Goal position (i, j)
        max_iterations: Maximum number of iterations
        step_size: Maximum distance to extend the tree in each iteration
        goal_tolerance: Distance threshold to consider goal reached

    Returns:
        List of nodes forming the path from start to goal, or empty list if no path found
    """
    # Initialize tree with start node
    tree = {start: None}

    plot_node(start, 'blue', marker='o', size=8)

    for iteration in range(max_iterations):
        # Sample a random point (with bias toward goal occasionally)
        if np.random.rand() < 0.3:  # 10% chance to sample the goal
            sampled_point = goal
        else:
            sampled_point = sample_random_point(map.shape)

        # Find the nearest node in the tree
        nearest_node = find_nearest_node(tree, sampled_point)

        # Steer from nearest_node toward sampled_point
        new_node = steer(nearest_node, sampled_point, step_size)

        # Check if new_node is already in the tree
        if new_node in tree:
            continue

        # Add new_node to the tree
        tree[new_node] = nearest_node

        # Visualize the new node and edge
        plot_node(new_node, 'green', marker='.', size=2)
        plot_edge(nearest_node, new_node, color='cyan', linewidth=0.5)

        # Check if we've reached the goal
        if is_goal_reached(new_node, goal, goal_tolerance):
            print(f"Goal reached in {iteration + 1} iterations!")
            return reconstruct_path(tree, start, new_node)

    print(f"Max iterations ({max_iterations}) reached without finding goal")
    return []


def main():
    rows = 200
    cols = 300
    map = np.ones((rows, cols)) * 255

    start = (100, 150)
    goal = (np.random.randint(0, rows), np.random.randint(0, cols))

    print(f"Start: {start}, Goal: {goal}")

    # Set up the plot
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(12, 8))
    plt.imshow(map, cmap='gray')
    plt.plot(start[1], start[0], 'bo', markersize=10, label='Start')
    plt.plot(goal[1], goal[0], 'y*', markersize=15, label='Goal')
    plt.legend()
    plt.title('RRT Path Planning')
    plt.draw()

    # Run RRT
    path = rrt_search(map, start, goal)

    # Plot the final path in red
    if path:
        for i in range(len(path) - 1):
            plot_edge(path[i], path[i + 1], color='red', linewidth=2)

        for node in path:
            plt.plot(node[1], node[0], 'r.', markersize=5)

        plt.draw()
        print(f"Path found with {len(path)} nodes")
    else:
        print("No path found!")

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the window open


if __name__ == "__main__":
    main()

"""
Collision checking mini-project for path planning.
Implements functions to check if paths are obstacle-free on grid maps.
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_nd, random_shapes


def generate_random_map(height=200, width=300, min_shapes=5, max_shapes=20):
    """Generate a random map with obstacles using scikit-image."""
    map_array, labels = random_shapes(
        (height, width),
        min_shapes=min_shapes,
        max_shapes=max_shapes,
        num_channels=1
    )
    return map_array


def is_navigable(map_array, i, j):
    """
    Check if a pixel at position (i, j) is navigable free space.

    Args:
        map_array: The map array
        i: Row index
        j: Column index

    Returns:
        True if navigable (white/255), False if obstacle
    """
    # Check bounds
    if i < 0 or i >= map_array.shape[0] or j < 0 or j >= map_array.shape[1]:
        return False

    # Free space is marked as 255 (white), obstacles are other values
    return map_array[i, j] == 255


def is_path_open(map_array, point_a, point_b):
    """
    Check if the straight-line path from point_a to point_b is obstacle-free.

    Args:
        map_array: The map array where 255 = free space, other values = obstacles
        point_a: Starting point (i, j)
        point_b: Ending point (i, j)

    Returns:
        True if path is obstacle-free, False otherwise
    """
    # First check if the endpoint is navigable
    if not is_navigable(map_array, point_b[0], point_b[1]):
        return False

    # Get all discrete points along the line from a to b
    line_points = line_nd(point_a, point_b, integer=True)

    # Check each point along the line
    for i in range(len(line_points[0])):
        point_i = line_points[0][i]
        point_j = line_points[1][i]

        if not is_navigable(map_array, point_i, point_j):
            return False

    return True


def visualize_path_check(map_array, point_a, point_b):
    """
    Visualize a path check on the map.
    Shows the path in white if open, red if blocked.
    """
    plt.figure(figsize=(10, 7))
    plt.imshow(map_array, cmap='gray')

    # Get line points
    line_points = line_nd(point_a, point_b, integer=True)

    # Check if path is open
    path_is_open = is_path_open(map_array, point_a, point_b)

    # Plot the line points
    if path_is_open:
        plt.plot(line_points[1], line_points[0], 'w.', markersize=3, label='Open path')
    else:
        # Find where collision occurs
        collision_index = None
        for i in range(len(line_points[0])):
            point_i = line_points[0][i]
            point_j = line_points[1][i]
            if not is_navigable(map_array, point_i, point_j):
                collision_index = i
                break

        # Plot free part in white, collision part in red
        if collision_index is not None and collision_index > 0:
            plt.plot(line_points[1][:collision_index], line_points[0][:collision_index],
                    'w.', markersize=3)
            plt.plot(line_points[1][collision_index:], line_points[0][collision_index:],
                    'r.', markersize=3, label='Collision')
        else:
            plt.plot(line_points[1], line_points[0], 'r.', markersize=3, label='Collision')

    # Plot start and end points
    plt.plot(point_a[1], point_a[0], 'bo', markersize=10, label='Start')
    plt.plot(point_b[1], point_b[0], 'y*', markersize=15, label='Goal')

    plt.legend()
    plt.title(f'Path Check: {"OPEN" if path_is_open else "BLOCKED"}')
    plt.tight_layout()
    plt.show()


def main():
    """Test the collision checking functions."""
    # Generate a random map
    print("Generating random map...")
    map_array = generate_random_map(height=200, width=300, min_shapes=5, max_shapes=20)

    # Test case 1: Path from top-left to bottom-right
    point_a = (10, 10)
    point_b = (190, 290)

    print(f"\nTest 1: Path from {point_a} to {point_b}")
    print(f"Result: {'OPEN' if is_path_open(map_array, point_a, point_b) else 'BLOCKED'}")
    visualize_path_check(map_array, point_a, point_b)

    # Test case 2: Another random path
    point_c = (50, 50)
    point_d = (150, 250)

    print(f"\nTest 2: Path from {point_c} to {point_d}")
    print(f"Result: {'OPEN' if is_path_open(map_array, point_c, point_d) else 'BLOCKED'}")
    visualize_path_check(map_array, point_c, point_d)


if __name__ == "__main__":
    main()

from collections import deque

import matplotlib.pyplot as plt
import numpy as np

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def get_neighbours(map, i, j) -> list[tuple]:
    cutoff_prob = 0.3

    neighbours = []

    for direction in DIRECTIONS:
        current_i = i + direction[0]
        current_j = j + direction[1]
        if 0 <= current_i < len(map) and 0 <= current_j < len(
                map[0]):  # check that we're still within map boundaries
            if map[current_i][current_j] <= cutoff_prob:
                neighbours.append((current_i, current_j))

    return neighbours


def get_shortest_path(map, start, goal):
    queue = deque([start])
    visited_nodes = {start}
    parent = {start: None}

    while queue:
        current_node = queue.popleft()
        refresh_map(current_node)

        if current_node == goal:
            # Reconstruct path from goal to start
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = parent[current_node]
            return path[::-1]  # Reverse to get start-to-goal order

        neighbours = get_neighbours(map, current_node[0], current_node[1])
        for neighbour in neighbours:
            if neighbour not in visited_nodes:
                visited_nodes.add(neighbour)
                parent[neighbour] = current_node
                queue.append(neighbour)

    return []  # No path found


def refresh_map(vertex):
    plt.plot(vertex[1], vertex[0], 'g.', markersize=3)
    plt.draw()
    plt.pause(0.01)


def main():
    rows = 20
    cols = 30
    map = np.random.rand(rows, cols) <0.1

    goal_x = np.random.randint(rows)
    goal_y = np.random.randint(cols)
    goal = (goal_x, goal_y)
    start = (0, 0)
    print(f"Start: {start}, Goal: {goal}")

    plt.ion()  # turns 'interactive mode' on BEFORE showing anything
    plt.imshow(map)  # shows the map
    plt.plot(start[1], start[0], 'bo', markersize=10)  # blue dot at start
    plt.plot(goal[1], goal[0], 'y*', markersize=15)  # yellow asterisk at goal
    plt.draw()

    path = get_shortest_path(map, start, goal)

    # Plot the final path in red
    if path:
        for p in path:
            plt.plot(p[1], p[0], 'r.', markersize=8)
        plt.draw()
        print(f"Path found with {len(path)} steps")
    else:
        print("No path found!")

    plt.ioff()  # turn off interactive mode
    plt.show()  # keep the window open


if __name__ == "__main__":
    main()

from collections import defaultdict
import heapq

import matplotlib.pyplot as plt
import numpy as np

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]


def get_neighbours(map, i, j) -> list[tuple[int, tuple[int, int]]]:
    cutoff_prob = 0.3

    neighbours = []

    for direction in DIRECTIONS:
        current_i = i + direction[0]
        current_j = j + direction[1]
        if 0 <= current_i < len(map) and 0 <= current_j < len(map[0]):  # check that we're still within map boundaries
            if map[current_i][current_j] <= cutoff_prob:
                move_cost = 1
                if abs(current_i) == abs(current_j):  # diagonal, has different move cost
                    move_cost = np.sqrt(2)
                neighbours.append((move_cost, (current_i, current_j)))

    return neighbours


def heuristic(node, goal):
    """Euclidean distance - admissible heuristic (never overestimates)."""
    return np.sqrt((goal[0] - node[0])**2 + (goal[1] - node[1])**2)


def get_shortest_path(map, start, goal):
    distances = defaultdict(lambda: float("inf"))  # g(n): actual cost from start
    distances[start] = 0
    parent = {start: None}
    visited = set()

    # Priority queue: (f, node) where f = g + h
    pq = [(heuristic(start, goal), start)]
    plot_node(start, 'green')  # queued

    while pq:
        _, current_node = heapq.heappop(pq)

        if current_node in visited:
            continue

        visited.add(current_node)
        plot_node(current_node, 'yellow')  # explored

        if current_node == goal:
            # Reconstruct path from goal to start
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = parent[current_node]
            return path[::-1]  # Reverse to get start-to-goal order

        neighbours = get_neighbours(map, current_node[0], current_node[1])
        for move_cost, neighbour in neighbours:
            if neighbour not in visited:
                new_g = distances[current_node] + move_cost
                if new_g < distances[neighbour]:
                    distances[neighbour] = new_g
                    parent[neighbour] = current_node
                    f = new_g + heuristic(neighbour, goal)
                    heapq.heappush(pq, (f, neighbour))
                    plot_node(neighbour, 'green')  # queued

    return []  # No path found


def plot_node(vertex, color, marker='o', size=3):
    plt.plot(vertex[1], vertex[0], color=color, marker=marker, markersize=size)
    plt.draw()
    plt.pause(0.001)


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

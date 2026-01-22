# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python robotics project for the Webots simulator, implementing autonomous robot control using behavior trees and RRT path planning. The main project simulates a TIAGo mobile manipulator performing a jar collection task.

## Running Code

All Python scripts run directly with Python 3.11. No build system is configured.

```bash
# Activate virtual environment
source venv/bin/activate

# Standalone learning projects
python behaviour-trees.py      # Behavior tree examples
python robotics.py             # A* pathfinding
python robotics-rrt.py         # RRT implementation
python collision_checking.py   # Collision detection

# Webots controllers (require Webots simulator running)
# These are loaded by Webots, not run directly from command line
```

## Key Dependencies

- **py_trees**: Behavior tree framework
- **ikpy**: Inverse kinematics solver
- **numpy/scipy**: Numerical computing
- **matplotlib**: Visualization
- **skimage**: Image processing for mapping

## Architecture

### webots-project/ - Main Robotics System

The robot controller uses a behavior tree architecture with these modules:

| Module | Purpose |
|--------|---------|
| `behaviourtree.py` | Main controller, tree structure, Blackboard for inter-behavior communication |
| `mapping.py` | LIDAR → occupancy grid → configuration space (cspace.npy) |
| `planning.py` | RRT path planning with world↔map coordinate conversion |
| `navigation.py` | Waypoint following, heading control (TurnToHeading behavior) |
| `recognition.py` | Camera-based object detection, color matching |
| `jointcontrol.py` | Arm joint control, grasping behaviors |
| `inverse_kinematics.py` | IK using ikpy with tiago_urdf.urdf model |

### Coordinate Systems

```
World frame: x ∈ [-2.15, 2.15], y ∈ [-3.92, 1.66]
Map frame:   200×300 pixel grid (stored as [row, col] = [y, x])

Conversion functions in planning.py:
  world2map(x_world, y_world) → [y_map, x_map]
  map2world(y_map, x_map) → (x_world, y_world)
```

### Behavior Tree Pattern

Behaviors return `py_trees.common.Status.{SUCCESS, FAILURE, RUNNING}`. The Blackboard class in `behaviourtree.py` provides key-value storage for sharing data between behaviors (robot reference, detected objects, paths).

### Standalone Learning Projects (root directory)

- `behaviour-trees.py`: Basic BT implementation (Sequence, Selector, Action, Repeat, Parallel)
- `robotics.py`: A* pathfinding with visualization
- `robotics-rrt.py`: RRT algorithm implementation
- `collision_checking.py`: Line-of-sight collision detection on grid maps

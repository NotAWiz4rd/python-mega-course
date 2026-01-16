"""
Behaviour Tree main controller for the Webots robot.

This module defines the behaviour tree structure that controls the robot's
overall behaviour. The tree implements the following workflow:
1. Check if a map exists, if not, map the environment while navigating
2. Plan a path to the lower left corner using RRT
3. Navigate to the lower left corner
4. Plan a path to the sink using RRT
5. Navigate to the sink
"""

from os.path import exists

import numpy as np
import py_trees
from py_trees.composites import Sequence, Parallel, Selector

from navigation import Navigation
from mapping import Mapping
from planning import Planning

from controller import Robot, Supervisor


# Create the Robot instance (Supervisor allows access to simulation features)
robot = Supervisor()

# Get simulation timestep
timestep = int(robot.getBasicTimeStep())

# Predefined waypoints for the mapping phase (navigate around the table)
waypoints = [
    (0.614, -0.19),
    (0.77, -0.94),
    (0.37, -3.04),
    (-1.14, -3.39),
    (-1.53, -3.39),
    (-1.8, -1.46),
    (-1.44, 0.38),
    (0, 0)
]


class DoesMapExist(py_trees.behaviour.Behaviour):
    """
    Condition behaviour that checks if the configuration space map exists.

    This allows the behaviour tree to skip the mapping phase if a map
    has already been created from a previous run.
    """

    def __init__(self, name: str):
        """
        Initialize the condition behaviour.

        Args:
            name: Name of this behaviour node
        """
        super(DoesMapExist, self).__init__(name)

    def update(self) -> py_trees.common.Status:
        """
        Check if the configuration space file exists.

        Returns:
            SUCCESS if map file exists, FAILURE otherwise
        """
        file_exists = exists('cspace.npy')
        if file_exists:
            print("DoesMapExist: Map exists, skipping mapping phase")
            return py_trees.common.Status.SUCCESS
        else:
            print("DoesMapExist: Map does not exist, will create map")
            return py_trees.common.Status.FAILURE


class Blackboard:
    """
    Simple blackboard implementation for sharing data between behaviours.

    The blackboard pattern allows behaviours to communicate without
    direct coupling, making the behaviour tree more modular.
    """

    def __init__(self):
        """Initialize empty blackboard storage."""
        self.data = {}

    def write(self, key, value):
        """
        Store a value on the blackboard.

        Args:
            key: String identifier for the data
            value: Data to store
        """
        self.data[key] = value

    def read(self, key):
        """
        Retrieve a value from the blackboard.

        Args:
            key: String identifier for the data

        Returns:
            The stored value, or None if key doesn't exist
        """
        return self.data.get(key)


# Create shared blackboard and store robot reference
blackboard = Blackboard()
blackboard.write('robot', robot)

# Store mapping waypoints (forward and reverse for complete coverage)
blackboard.write('waypoints', np.concatenate((waypoints, np.flip(waypoints, 0)), axis=0))

# Build the behaviour tree structure matching the workflow diagram:
#
# Sequence "Main"
# ├── Selector "Does map exist?"
# │   ├── DoesMapExist "Test for Map"
# │   └── Parallel "Mapping" (SuccessOnOne)
# │       ├── Mapping "Map the environment"
# │       └── Navigation "Move around the table"
# ├── Planning "Compute path to lower left corner"
# ├── Navigation "Move to lower left corner"
# ├── Planning "Compute path to sink"
# └── Navigation "Move to sink"

tree = Sequence("Main", memory=True, children=[
    # First, check if map exists or create one
    Selector("Does map exist?", memory=True, children=[
        DoesMapExist("Test for Map"),
        Parallel(
            "Mapping",
            policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
            children=[
                Mapping("Map the environment", blackboard),
                Navigation("Move around the table", blackboard)
            ]
        )
    ]),
    # Plan and execute path to lower left corner
    Planning("Compute path to lower left corner", blackboard, (-1.46, -3.12)),
    Navigation("Move to lower left corner", blackboard),
    # Plan and execute path to sink
    Planning("Compute path to sink", blackboard, (-0.88, 0.09)),
    Navigation("Move to sink", blackboard)
])

# Set up all behaviours in the tree (calls setup() on each)
tree.setup_with_descendants()

# Main simulation loop
while robot.step(timestep) != -1:
    # Tick the behaviour tree once per simulation step
    tree.tick_once()

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

from controller import Robot, Supervisor


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


from os.path import exists

import numpy as np
import py_tree
import py_trees
from py_trees.composites import Sequence, Parallel, Selector

from navigation import Navigation
from mapping import Mapping
from planning import Planning

from controller import Robot, Supervisor

# create the Robot instance.
robot = Supervisor()

timestep = int(robot.getBasicTimeStep())

waypoints = [(0.614. -0.19), (0.77, -0.94), (0.37, -3.04), (-1.14, -3.39), (-1.53, -3.39),
             (-1.8, -1.46), (-1.44, 0.38), (0, 0)]


class DoesMapExist(py_tree.behaviour.Behaviour):
    def update(self) -> py_trees.common.Status:
        file_exists = exists('cspace.npy')
        if file_exists:
            print("Map exists")
            return py_trees.common.Status.SUCCESS
        else:
            print("Map does not exist")
            return py_trees.common.Status.FAILURE


class Blackboard:
    def __init__(self):
        self.data = {}

    def write(self, key, value):
        self.data[key] = value

    def read(self, key):
        return self.data.get(key)


blackboard = Blackboard()
blackboard.write('robot', robot)
blackboard.write('waypoints', np.concatenate((waypoints, np.flip(waypoints, 0)), axis=0))

tree = Sequence("Main", children= [
    Selector("Does map exist?", children=
    DoesMapExist("Test for Map"), Parallel("Mapping", policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
                                           children=[Mapping("Map the environment", blackboard), Navigation("Move around the table", blackboard)])
], memory=True),
Planning("Compute path to lower left corner", blackboard, (-1.46, -3.12)),
Navigation("Move to lower left corner", blackboard),
Planning("Compute path to the sink", blackboard, (-0.88, 0.09)),
Navigation("Move to the sink", blackboard)])

tree.setup_with_descendants()




while robot.step(timestep) != -1:
    tree.tick_once()
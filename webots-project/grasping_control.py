"""
Jar Collection Robot Controller

This controller uses a behavior tree to:
1. Scan the environment to detect colored jars using camera recognition
2. Navigate to each jar in sequence
3. Pick up each jar using IK-based arm control
4. Transport and deposit jars on the table
"""

import py_trees
from controller import Robot, Supervisor

# Table location where jars will be deposited (x, y world coordinates)
TABLE_LOCATION = (0.0, -1.5)
TABLE_DROP_POSITION = (0.5, -1.5)

class DynamicPlanning(py_trees.behaviour.Behaviour):
    """
    Planning behaviour that reads the goal from blackboard.

    Uses approach_position key for navigation target.
    """

    def __init__(self, name: str, position_key: str):
        super(DynamicPlanning, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.position_key = position_key
        self.robot = self.blackboard.get('robot')
        self.path_computed = False

    def setup(self):
        from planning import Planning
        self.timestep = int(self.robot.getBasicTimeStep())
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

    def initialise(self):
        self.path_computed = False

    def update(self):
        if self.path_computed:
            return py_trees.common.Status.SUCCESS

        import numpy as np
        from planning import RRT, world2map, map2world

        goal = self.blackboard.get(self.position_key)
        if goal is None:
            print(f"DynamicPlanning: No goal found at '{self.position_key}'")
            return py_trees.common.Status.FAILURE

        try:
            cspace = np.load('cspace.npy')
        except FileNotFoundError:
            print("DynamicPlanning: cspace.npy not found")
            return py_trees.common.Status.FAILURE

        x_world = self.gps.getValues()[0]
        y_world = self.gps.getValues()[1]

        start_map = world2map(x_world, y_world)
        goal_map = world2map(goal[0], goal[1])

        print(f"DynamicPlanning: Planning from ({x_world:.2f}, {y_world:.2f}) to ({goal[0]:.2f}, {goal[1]:.2f})")

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
            print("DynamicPlanning: No path found")
            return py_trees.common.Status.FAILURE

        waypoints = [map2world(p[0], p[1]) for p in path]
        self.blackboard.set('waypoints', waypoints)

        print(f"DynamicPlanning: Found path with {len(waypoints)} waypoints")
        self.path_computed = True
        return py_trees.common.Status.SUCCESS


class TurnToTarget(py_trees.behaviour.Behaviour):
    """Turn to face the current target position from blackboard."""

    def __init__(self, name: str, position_key: str):
        super(TurnToTarget, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.position_key = position_key
        self.robot = self.blackboard.get('robot')
        self.turn_speed = 1.5
        self.target_heading = None

    def setup(self):
        import numpy as np
        self.timestep = int(self.robot.getBasicTimeStep())
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)
        self.left_motor = self.robot.getDevice('wheel_left_joint')
        self.right_motor = self.robot.getDevice('wheel_right_joint')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

    def initialise(self):
        import numpy as np
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        target = self.blackboard.get(self.position_key)
        if target is None:
            self.target_heading = None
            return

        robot_x = self.gps.getValues()[0]
        robot_y = self.gps.getValues()[1]

        dx = target[0] - robot_x
        dy = target[1] - robot_y
        self.target_heading = np.arctan2(dy, dx)
        print(f"TurnToTarget: Turning to face ({target[0]:.2f}, {target[1]:.2f})")

    def update(self):
        import numpy as np

        if self.target_heading is None:
            return py_trees.common.Status.FAILURE

        current_heading = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])
        angle_error = self.target_heading - current_heading

        if angle_error > np.pi:
            angle_error -= 2 * np.pi
        elif angle_error < -np.pi:
            angle_error += 2 * np.pi

        if abs(angle_error) < 0.035:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            print(f"TurnToTarget: Heading achieved")
            return py_trees.common.Status.SUCCESS

        turn_velocity = np.clip(2.0 * angle_error, -self.turn_speed, self.turn_speed)
        self.left_motor.setVelocity(-turn_velocity)
        self.right_motor.setVelocity(turn_velocity)

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)


class StoreTablePosition(py_trees.behaviour.Behaviour):
    """Store the table position on the blackboard for navigation."""

    def __init__(self, name: str, table_position: tuple):
        super(StoreTablePosition, self).__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()
        self.table_position = table_position

    def update(self):
        self.blackboard.set('approach_position', self.table_position)
        print(f"StoreTablePosition: Set approach to ({self.table_position[0]:.2f}, {self.table_position[1]:.2f})")
        return py_trees.common.Status.SUCCESS


class PrintStatus(py_trees.behaviour.Behaviour):
    """Simple behavior to print status messages."""

    def __init__(self, name: str, message: str):
        super(PrintStatus, self).__init__(name)
        self.message = message

    def update(self):
        print(f"=== {self.message} ===")
        return py_trees.common.Status.SUCCESS



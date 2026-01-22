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
from py_trees.composites import Sequence, Selector
from py_trees.decorators import Repeat

from behaviourtree import Blackboard
from navigation import Navigation, TurnToHeading, DriveForward
from planning import Planning
from jointcontrol import (
    PositionJoints, GripWithForce, ReleaseGripper,
    ARM_CARRY, HAND_OPEN
)
from recognition import (
    ScanForObjects, SelectNextTarget, HasMoreTargets,
    GetTargetApproachPosition
)
from inverse_kinematics import MoveArmToPosition, compute_ik

# Create the Robot instance
robot = Supervisor()

# Get the time step of the current world
timestep = int(robot.getBasicTimeStep())

# Enable camera with recognition
camera = robot.getDevice("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)

# GPS for position feedback
gps = robot.getDevice('gps')
gps.enable(timestep)

# Compass for heading feedback
compass = robot.getDevice('compass')
compass.enable(timestep)

# Differential drive motors
left_motor = robot.getDevice('wheel_left_joint')
right_motor = robot.getDevice('wheel_right_joint')

# Robot joint definitions
robot_joints = {
    'torso_lift_joint': 0.15,
    'arm_1_joint': 0.71,
    'arm_2_joint': 1.02,
    'arm_3_joint': -2.815,
    'arm_4_joint': 1.011,
    'arm_5_joint': 0,
    'arm_6_joint': 0,
    'arm_7_joint': 0,
    'gripper_left_finger_joint': 0.045,
    'gripper_right_finger_joint': 0.045,
    'head_1_joint': 0,
    'head_2_joint': -0.3  # Look slightly down
}

joint_motors = {}
joint_sensors = {}

for joint_key in robot_joints.keys():
    joint_motors[joint_key] = robot.getDevice(joint_key)
    if "finger" in joint_key:
        if "left" in joint_key:
            sensor_key = "gripper_left_sensor_finger_joint"
        else:
            sensor_key = "gripper_right_sensor_finger_joint"

        joint_sensors[joint_key] = robot.getDevice(sensor_key)
        joint_sensors[joint_key].enable(timestep)
    else:
        sensor_key = joint_key + "_sensor"
        joint_sensors[joint_key] = robot.getDevice(sensor_key)
        joint_sensors[joint_key].enable(timestep)

joint_motors['gripper_left_finger_joint'].enableForceFeedback(timestep)
joint_motors['gripper_right_finger_joint'].enableForceFeedback(timestep)

# Initialize all joint motors to safe starting position
for joint_key in robot_joints.keys():
    joint_motors[joint_key].setPosition(robot_joints.get(joint_key))

# Create shared blackboard and store robot reference
blackboard = Blackboard()
blackboard.write('robot', robot)

# Table location where jars will be deposited (x, y world coordinates)
TABLE_LOCATION = (0.0, -1.5)
TABLE_DROP_POSITION = (0.5, -1.5)

# Arm positions for IK-based grasping
# These are relative positions in the robot's local frame
ARM_APPROACH_POSITION = {
    'torso_lift_joint': 0.25,
    'arm_1_joint': 1.4,
    'arm_2_joint': -0.2,
    'arm_3_joint': -2.0,
    'arm_4_joint': 0.5,
    'arm_5_joint': 0,
    'arm_6_joint': 0,
    'arm_7_joint': 0,
    'head_1_joint': 0,
    'head_2_joint': -0.3,
}

ARM_GRASP_POSITION = {
    'torso_lift_joint': 0.25,
    'arm_1_joint': 1.4,
    'arm_2_joint': -0.4,
    'arm_3_joint': -2.0,
    'arm_4_joint': 0.5,
    'arm_5_joint': 0,
    'arm_6_joint': 0,
    'arm_7_joint': 0,
}

ARM_PLACE_POSITION = {
    'torso_lift_joint': 0.2,
    'arm_1_joint': 1.5,
    'arm_2_joint': -0.1,
    'arm_3_joint': -2.0,
    'arm_4_joint': 1.0,
    'arm_5_joint': 0,
    'arm_6_joint': 0.3,
    'arm_7_joint': 0,
}


class DynamicPlanning(py_trees.behaviour.Behaviour):
    """
    Planning behaviour that reads the goal from blackboard.

    Uses approach_position key for navigation target.
    """

    def __init__(self, name: str, blackboard: Blackboard, position_key: str):
        super(DynamicPlanning, self).__init__(name)
        self.blackboard = blackboard
        self.position_key = position_key
        self.robot = blackboard.read('robot')
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

        goal = self.blackboard.read(self.position_key)
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
        self.blackboard.write('waypoints', waypoints)

        print(f"DynamicPlanning: Found path with {len(waypoints)} waypoints")
        self.path_computed = True
        return py_trees.common.Status.SUCCESS


class TurnToTarget(py_trees.behaviour.Behaviour):
    """Turn to face the current target position from blackboard."""

    def __init__(self, name: str, blackboard: Blackboard, position_key: str):
        super(TurnToTarget, self).__init__(name)
        self.blackboard = blackboard
        self.position_key = position_key
        self.robot = blackboard.read('robot')
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

        target = self.blackboard.read(self.position_key)
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

    def __init__(self, name: str, blackboard: Blackboard, table_position: tuple):
        super(StoreTablePosition, self).__init__(name)
        self.blackboard = blackboard
        self.table_position = table_position

    def update(self):
        self.blackboard.write('approach_position', self.table_position)
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


# Grasp and transport sequence - only runs if grasp succeeds
grasp_and_deliver = Sequence("Grasp and deliver", memory=True, children=[
    # Grasp the jar
    GripWithForce("Grip jar", blackboard),

    # Lift and retract arm
    PositionJoints("Lift with jar", ARM_APPROACH_POSITION, blackboard),
    PositionJoints("Carry position", ARM_CARRY, blackboard),

    PrintStatus("Jar collected", "Jar collected, transporting to table"),

    # Navigate to table
    StoreTablePosition("Set table target", blackboard, TABLE_LOCATION),
    DynamicPlanning("Plan to table", blackboard, 'approach_position'),
    Navigation("Navigate to table", blackboard),

    # Turn to face the table drop point
    TurnToTarget("Face table", blackboard, 'approach_position'),

    # Place the jar on the table
    PositionJoints("Extend to place", ARM_PLACE_POSITION, blackboard),
    ReleaseGripper("Release jar", blackboard),

    # Retract arm
    PositionJoints("Retract arm", ARM_CARRY, blackboard),

    PrintStatus("Jar placed", "Jar placed on table"),
])

# Recovery sequence when grasp fails - retract arm and continue to next jar
grasp_failed_recovery = Sequence("Handle failed grasp", memory=True, children=[
    PrintStatus("Grasp failed", "Failed to grasp jar, moving to next target"),
    PositionJoints("Retract after fail", ARM_CARRY, blackboard),
])

# Selector tries grasp first, falls back to recovery on failure
# This ensures the single_jar_sequence always succeeds (unless SelectNextTarget fails)
attempt_grasp = Selector("Attempt grasp", memory=False, children=[
    grasp_and_deliver,
    grasp_failed_recovery,
])

# Build the single jar collection sequence
# This sequence handles: approach, attempt grasp, and either deliver or skip
single_jar_sequence = Sequence("Collect Single Jar", memory=True, children=[
    # Select the next target jar (FAILURE here = no more jars = stop loop)
    SelectNextTarget("Select next jar", blackboard),

    # Compute approach position for this jar (0.6m to stay clear of obstacles in cspace)
    GetTargetApproachPosition("Compute approach", blackboard, approach_distance=0.4),

    # Plan and navigate to approach position
    DynamicPlanning("Plan to jar", blackboard, 'approach_position'),
    Navigation("Navigate to jar", blackboard),

    # Turn to face the jar, then drive closer (path planner can't get us all the way)
    TurnToTarget("Face jar", blackboard, 'grasp_target_position'),
    DriveForward("Drive closer to jar", blackboard, distance=0.25),

    # Prepare arm for grasping
    PositionJoints("Open gripper", HAND_OPEN, blackboard),
    PositionJoints("Approach position", ARM_APPROACH_POSITION, blackboard),
    PositionJoints("Lower to grasp", ARM_GRASP_POSITION, blackboard),

    # Try to grasp and deliver - if grasp fails, recovers and continues
    attempt_grasp,
])


# Main behavior tree
# 1. Scan environment for jars
# 2. Repeat the jar collection sequence until no jars remain
tree = Sequence("Main - Jar Collection", memory=True, children=[
    PrintStatus("Start", "Starting jar collection mission"),

    # Phase 1: Scan environment to find all jars
    # Define the specific RGB colors for our jars (values 0-1)
    ScanForObjects("Scan for jars", blackboard),

    # Phase 2: Collect jars one by one
    # The Repeat decorator will keep running until the child fails
    # (which happens when SelectNextTarget has no more jars)
    Selector("Collect all jars", memory=False, children=[
        Repeat("Repeat collection", child=single_jar_sequence, num_success=-1),
        PrintStatus("Done", "All jars collected!")
    ]),
])

# Set up all behaviours in the tree
tree.setup_with_descendants()

print("=== Jar Collection Robot Initialized ===")
print("Behavior tree ready. Starting main loop...")

# Main simulation loop
while robot.step(timestep) != -1:
    # Tick the behaviour tree once per simulation step
    tree.tick_once()

    # Check tree status
    if tree.status == py_trees.common.Status.SUCCESS:
        print("=== Behaviour tree completed successfully! ===")
        break
    elif tree.status == py_trees.common.Status.FAILURE:
        print("=== Behaviour tree failed! Restarting... ===")
        tree.setup_with_descendants()

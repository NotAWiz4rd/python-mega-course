import py_trees
from controller import Robot, Supervisor
from py_trees.composites import Sequence

from behaviourtree import Blackboard
from navigation import Navigation
from planning import Planning
from jointcontrol import (
    PositionJoints, GripWithForce, ReleaseGripper,
    ARM_STRAIGHT_FORWARD, ARM_APPROACH_COUNTER, ARM_GRASP_POSITION,
    ARM_CARRY, ARM_PLACE_DOWN, HAND_OPEN
)

# create the Robot instance.
robot = Supervisor()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# GPS for position feedback
gps = robot.getDevice('gps')
gps.enable(timestep)

# Compass for heading feedback
compass = robot.getDevice('compass')
compass.enable(timestep)

# Differential drive motors
left_motor = robot.getDevice('wheel_left_joint')
right_motor = robot.getDevice('wheel_right_joint')

# safe positions
robot_joints = {
    'torso_lift_joint': 0.35,
    'arm_1_joint': 0.71,
    'arm_2_joint': 1.02,
    'arm_3_joint': -2.815,
    'arm_4_joint': 1.011,
    'arm_5_joint': 0,
    'arm_6_joint': 0,
    'arm_7_joint': 0,
    'gripper_left_finger_joint': 0,
    'gripper_right_finger_joint': 0,
    'head_1_joint': 0,
    'head_2_joint': 0}

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

# init all joint motors to initial safe position
for joint_key in robot_joints.keys():
    joint_motors[joint_key].setPosition(robot_joints.get(joint_key))

# Create shared blackboard and store robot reference
blackboard = Blackboard()
blackboard.write('robot', robot)

tree = Sequence("Main", memory=True, children=[
    # Phase 1: Navigate to counter
    Planning("Compute path to kitchen counter", blackboard, (1, -0.35)),
    Navigation("Move to kitchen counter", blackboard),

    # Phase 2: Position arm and open gripper for grasping
    PositionJoints("Open gripper", HAND_OPEN, blackboard),
    PositionJoints("Approach position", ARM_APPROACH_COUNTER, blackboard),
    PositionJoints("Lower to grasp position", ARM_GRASP_POSITION, blackboard),

    # Phase 3: Grip the glass using force feedback
    GripWithForce("Grip glass with force control", blackboard),

    # Phase 4: Lift and retract arm with glass
    PositionJoints("Lift with glass", ARM_APPROACH_COUNTER, blackboard),
    PositionJoints("Retract arm for carrying", ARM_CARRY, blackboard),

    # TODO: Future phases for transport and placement
    # Planning("Compute path to table", blackboard, (target_x, target_y)),
    # Navigation("Move to table", blackboard),
    # PositionJoints("Extend arm to place", ARM_PLACE_DOWN, blackboard),
    # ReleaseGripper("Release glass", blackboard),
    # PositionJoints("Retract arm", ARM_CARRY, blackboard),
])

# Set up all behaviours in the tree (calls setup() on each)
tree.setup_with_descendants()

while robot.step(timestep) != -1:
    # Tick the behaviour tree once per simulation step
    tree.tick_once()

    # Stop when the tree completes successfully
    if tree.status == py_trees.common.Status.SUCCESS:
        print("Behaviour tree completed successfully!")
        break

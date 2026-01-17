import py_trees
from controller import Robot, Supervisor
from py_trees.composites import Sequence

from behaviourtree import Blackboard
from navigation import Navigation
from planning import Planning
from jointcontrol import PositionJoints, ARM_STRAIGHT_FORWARD

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
    # Plan and execute path to lower left corner
    Planning("Compute path to kitchen counter 1st jar", blackboard, (1, -0.35)),
    Navigation("Move to kitchen counter 1st jar", blackboard),
    PositionJoints("Straight forward arm", ARM_STRAIGHT_FORWARD, blackboard)
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

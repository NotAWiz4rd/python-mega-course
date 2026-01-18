"""
Inverse Kinematics module using ikpy for the TIAGo robot arm.

This module provides IK-based arm control for precise end-effector positioning.
Uses ikpy library to compute joint angles from target positions.
"""

import numpy as np
import py_trees
from ikpy.chain import Chain
from ikpy.link import URDFLink

from behaviourtree import Blackboard


def create_tiago_arm_chain():
    """
    Create the kinematic chain for the TIAGo robot arm.

    The TIAGo arm has 7 DOF plus a torso lift joint.
    This chain models the arm from base to gripper.

    Joint order:
    1. torso_lift_joint - prismatic (vertical lift)
    2. arm_1_joint - shoulder yaw
    3. arm_2_joint - shoulder pitch
    4. arm_3_joint - shoulder roll
    5. arm_4_joint - elbow pitch
    6. arm_5_joint - forearm roll
    7. arm_6_joint - wrist pitch
    8. arm_7_joint - wrist roll

    Returns:
        ikpy Chain object representing the arm kinematics
    """
    # TIAGo arm link lengths (approximate, based on robot specs)
    # These values are tuned for the Webots TIAGo model

    chain = Chain(name='tiago_arm', links=[
        # Base link (fixed)
        URDFLink(
            name="base",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=None,
            joint_type="fixed"
        ),
        # Torso lift - vertical prismatic joint
        URDFLink(
            name="torso_lift_joint",
            origin_translation=[0.0, 0.0, 0.60],  # Base height to torso
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],  # Moves along Z
            joint_type="prismatic",
            bounds=(0.0, 0.35)
        ),
        # Shoulder horizontal offset and arm_1 (yaw)
        URDFLink(
            name="arm_1_joint",
            origin_translation=[0.157, 0.0, 0.19],  # Torso to shoulder
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],  # Rotation around Z (yaw)
            bounds=(-1.18, 1.57)
        ),
        # arm_2 (shoulder pitch)
        URDFLink(
            name="arm_2_joint",
            origin_translation=[0.125, 0.0155, 0.0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],  # Rotation around Y (pitch)
            bounds=(-1.18, 1.57)
        ),
        # arm_3 (shoulder roll)
        URDFLink(
            name="arm_3_joint",
            origin_translation=[0.0, 0.0, -0.02],
            origin_orientation=[0, 0, 0],
            rotation=[1, 0, 0],  # Rotation around X (roll)
            bounds=(-3.14, 3.14)
        ),
        # arm_4 (elbow pitch)
        URDFLink(
            name="arm_4_joint",
            origin_translation=[0.02, 0.0, -0.222],  # Upper arm length
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],  # Rotation around Y (pitch)
            bounds=(0.0, 2.29)
        ),
        # arm_5 (forearm roll)
        URDFLink(
            name="arm_5_joint",
            origin_translation=[0.0, 0.0, 0.0],
            origin_orientation=[0, 0, 0],
            rotation=[1, 0, 0],  # Rotation around X (roll)
            bounds=(-2.07, 2.07)
        ),
        # arm_6 (wrist pitch)
        URDFLink(
            name="arm_6_joint",
            origin_translation=[0.0, 0.0, -0.15],  # Forearm length
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],  # Rotation around Y (pitch)
            bounds=(-1.39, 1.39)
        ),
        # arm_7 (wrist roll)
        URDFLink(
            name="arm_7_joint",
            origin_translation=[0.0, 0.0, 0.0],
            origin_orientation=[0, 0, 0],
            rotation=[1, 0, 0],  # Rotation around X (roll)
            bounds=(-2.07, 2.07)
        ),
        # End effector (gripper)
        URDFLink(
            name="gripper",
            origin_translation=[0.0, 0.0, -0.12],  # Wrist to gripper tip
            origin_orientation=[0, 0, 0],
            rotation=None,
            joint_type="fixed"
        ),
    ])

    return chain


# Global chain instance
TIAGO_ARM_CHAIN = None


def get_arm_chain():
    """Get or create the TIAGo arm chain (singleton)."""
    global TIAGO_ARM_CHAIN
    if TIAGO_ARM_CHAIN is None:
        TIAGO_ARM_CHAIN = create_tiago_arm_chain()
    return TIAGO_ARM_CHAIN


def compute_ik(target_position, target_orientation=None, initial_position=None):
    """
    Compute inverse kinematics for a target end-effector position.

    Args:
        target_position: [x, y, z] target position in robot base frame
        target_orientation: Optional 3x3 rotation matrix for orientation
        initial_position: Optional initial joint angles for optimization

    Returns:
        Dictionary mapping joint names to angles, or None if IK failed
    """
    chain = get_arm_chain()

    # Set up initial position if not provided
    if initial_position is None:
        # Default arm configuration
        initial_position = [0] * len(chain.links)

    # Compute IK
    try:
        if target_orientation is not None:
            joint_angles = chain.inverse_kinematics(
                target_position=target_position,
                target_orientation=target_orientation,
                initial_position=initial_position,
                orientation_mode="all"
            )
        else:
            joint_angles = chain.inverse_kinematics(
                target_position=target_position,
                initial_position=initial_position
            )

        # Map joint angles to joint names
        # Skip the base link and gripper link
        joint_names = [
            'torso_lift_joint',
            'arm_1_joint',
            'arm_2_joint',
            'arm_3_joint',
            'arm_4_joint',
            'arm_5_joint',
            'arm_6_joint',
            'arm_7_joint'
        ]

        # joint_angles[0] is base (fixed), joint_angles[1:9] are the arm joints
        result = {}
        for i, name in enumerate(joint_names):
            result[name] = joint_angles[i + 1]  # +1 to skip base link

        return result

    except Exception as e:
        print(f"IK computation failed: {e}")
        return None


def compute_fk(joint_angles):
    """
    Compute forward kinematics to get end-effector position.

    Args:
        joint_angles: Dictionary mapping joint names to angles, or list of angles

    Returns:
        4x4 transformation matrix of end-effector pose
    """
    chain = get_arm_chain()

    if isinstance(joint_angles, dict):
        # Convert dict to list
        angles = [0]  # base link
        angles.append(joint_angles.get('torso_lift_joint', 0.15))
        angles.append(joint_angles.get('arm_1_joint', 0))
        angles.append(joint_angles.get('arm_2_joint', 0))
        angles.append(joint_angles.get('arm_3_joint', 0))
        angles.append(joint_angles.get('arm_4_joint', 0))
        angles.append(joint_angles.get('arm_5_joint', 0))
        angles.append(joint_angles.get('arm_6_joint', 0))
        angles.append(joint_angles.get('arm_7_joint', 0))
        angles.append(0)  # gripper (fixed)
    else:
        angles = joint_angles

    return chain.forward_kinematics(angles)


class MoveArmToPosition(py_trees.behaviour.Behaviour):
    """
    Behaviour that moves the arm to a target position using IK.

    Computes inverse kinematics for the target position and moves
    all arm joints to the computed angles.
    """

    POSITION_TOLERANCE = 0.02  # radians

    def __init__(self, name: str, blackboard: Blackboard, target_position=None,
                 position_key: str = None, height_offset: float = 0.0):
        """
        Initialize the IK movement behaviour.

        Args:
            name: Behaviour name
            blackboard: Shared blackboard
            target_position: [x, y, z] target in robot base frame, or None to read from blackboard
            position_key: Blackboard key to read target from (if target_position is None)
            height_offset: Additional Z offset to add to target (for approach/grasp phases)
        """
        super(MoveArmToPosition, self).__init__(name)
        self.blackboard = blackboard
        self.fixed_target = target_position
        self.position_key = position_key
        self.height_offset = height_offset
        self.robot = blackboard.read('robot')
        self.target_joints = None
        self.joint_motors = {}
        self.joint_sensors = {}

    def setup(self):
        self.timestep = int(self.robot.getBasicTimeStep())

        # Get all arm joint motors and sensors
        joint_names = [
            'torso_lift_joint',
            'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint',
            'arm_5_joint', 'arm_6_joint', 'arm_7_joint'
        ]

        for name in joint_names:
            self.joint_motors[name] = self.robot.getDevice(name)
            sensor_name = name + "_sensor"
            self.joint_sensors[name] = self.robot.getDevice(sensor_name)
            self.joint_sensors[name].enable(self.timestep)

    def initialise(self):
        """Compute IK and set joint targets."""
        # Get target position
        if self.fixed_target is not None:
            target = self.fixed_target
        elif self.position_key:
            target = self.blackboard.read(self.position_key)
            if target is None:
                print(f"MoveArmToPosition: No target found at key '{self.position_key}'")
                self.target_joints = None
                return
        else:
            print("MoveArmToPosition: No target specified")
            self.target_joints = None
            return

        # Apply height offset
        if isinstance(target, tuple):
            target = list(target)
        target_with_offset = [target[0], target[1], target[2] + self.height_offset]

        print(f"MoveArmToPosition: Computing IK for target {target_with_offset}")

        # Get current joint angles for initial guess
        current_angles = {}
        for name, sensor in self.joint_sensors.items():
            current_angles[name] = sensor.getValue()

        # Convert to list for IK
        initial = [0]  # base
        initial.append(current_angles.get('torso_lift_joint', 0.15))
        for i in range(1, 8):
            initial.append(current_angles.get(f'arm_{i}_joint', 0))
        initial.append(0)  # gripper

        # Compute IK
        self.target_joints = compute_ik(target_with_offset, initial_position=initial)

        if self.target_joints is None:
            print("MoveArmToPosition: IK failed to find solution")
            return

        print("MoveArmToPosition: IK solution found:")
        for name, angle in self.target_joints.items():
            print(f"  {name}: {np.degrees(angle):.1f}°")

        # Command motors to target positions
        for name, angle in self.target_joints.items():
            if name in self.joint_motors:
                self.joint_motors[name].setPosition(angle)

    def update(self):
        """Wait for arm to reach target position."""
        if self.target_joints is None:
            return py_trees.common.Status.FAILURE

        # Check if all joints have reached their targets
        all_reached = True
        for name, target in self.target_joints.items():
            if name in self.joint_sensors:
                current = self.joint_sensors[name].getValue()
                error = abs(target - current)
                if error > self.POSITION_TOLERANCE:
                    all_reached = False
                    break

        if all_reached:
            print("MoveArmToPosition: Arm reached target position")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class MoveArmToRelativePosition(py_trees.behaviour.Behaviour):
    """
    Moves the arm to a position relative to the robot's current pose.

    Uses GPS and compass to transform world coordinates to robot frame,
    then computes IK.
    """

    POSITION_TOLERANCE = 0.02

    def __init__(self, name: str, blackboard: Blackboard, world_position_key: str,
                 approach_height: float = 0.15, grasp_height: float = 0.0):
        """
        Args:
            name: Behaviour name
            blackboard: Shared blackboard
            world_position_key: Key to read [x, y, z] world position from blackboard
            approach_height: Height above target for approach phase
            grasp_height: Height at target for grasping
        """
        super(MoveArmToRelativePosition, self).__init__(name)
        self.blackboard = blackboard
        self.world_position_key = world_position_key
        self.approach_height = approach_height
        self.grasp_height = grasp_height
        self.robot = blackboard.read('robot')
        self.target_joints = None
        self.phase = 'approach'  # 'approach' or 'grasp'

    def setup(self):
        self.timestep = int(self.robot.getBasicTimeStep())

        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)

        # Get all arm joint motors and sensors
        self.joint_motors = {}
        self.joint_sensors = {}
        joint_names = [
            'torso_lift_joint',
            'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint',
            'arm_5_joint', 'arm_6_joint', 'arm_7_joint'
        ]

        for name in joint_names:
            self.joint_motors[name] = self.robot.getDevice(name)
            sensor_name = name + "_sensor"
            self.joint_sensors[name] = self.robot.getDevice(sensor_name)
            self.joint_sensors[name].enable(self.timestep)

    def _world_to_robot_frame(self, world_pos):
        """Transform world position to robot base frame."""
        robot_x = self.gps.getValues()[0]
        robot_y = self.gps.getValues()[1]
        robot_z = self.gps.getValues()[2]
        theta = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])

        # Translate to robot origin
        dx = world_pos[0] - robot_x
        dy = world_pos[1] - robot_y
        dz = world_pos[2] - robot_z

        # Rotate to robot frame
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        local_x = dx * cos_t - dy * sin_t
        local_y = dx * sin_t + dy * cos_t
        local_z = dz

        return [local_x, local_y, local_z]

    def initialise(self):
        self.target_joints = None
        self.phase = 'approach'

        # Get world position from blackboard
        world_pos = self.blackboard.read(self.world_position_key)
        if world_pos is None:
            print(f"MoveArmToRelativePosition: No position at key '{self.world_position_key}'")
            return

        # Transform to robot frame
        local_pos = self._world_to_robot_frame(world_pos)
        local_pos[2] += self.approach_height  # Add approach height

        print(f"MoveArmToRelativePosition: World pos {world_pos} -> Local pos {local_pos}")

        # Compute IK
        self.target_joints = compute_ik(local_pos)

        if self.target_joints is None:
            print("MoveArmToRelativePosition: IK failed")
            return

        # Command motors
        for name, angle in self.target_joints.items():
            if name in self.joint_motors:
                self.joint_motors[name].setPosition(angle)

    def update(self):
        if self.target_joints is None:
            return py_trees.common.Status.FAILURE

        # Check positions
        all_reached = True
        for name, target in self.target_joints.items():
            if name in self.joint_sensors:
                current = self.joint_sensors[name].getValue()
                if abs(target - current) > self.POSITION_TOLERANCE:
                    all_reached = False
                    break

        if all_reached:
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class IKGraspSequence(py_trees.behaviour.Behaviour):
    """
    Complete IK-based grasping sequence.

    This behavior handles the full approach -> lower -> grasp sequence
    using inverse kinematics to position the arm based on the target's
    world position.
    """

    POSITION_TOLERANCE = 0.02

    def __init__(self, name: str, blackboard: Blackboard,
                 target_position_key: str = 'grasp_target_position',
                 approach_offset: float = 0.10,
                 grasp_offset: float = 0.02):
        """
        Args:
            name: Behaviour name
            blackboard: Shared blackboard
            target_position_key: Blackboard key for [x, y, z] target position
            approach_offset: Height above target for approach (meters)
            grasp_offset: Height above target for grasp (meters)
        """
        super(IKGraspSequence, self).__init__(name)
        self.blackboard = blackboard
        self.target_position_key = target_position_key
        self.approach_offset = approach_offset
        self.grasp_offset = grasp_offset
        self.robot = blackboard.read('robot')
        self.phase = 'init'  # init -> approach -> lower -> done
        self.target_joints = None

    def setup(self):
        self.timestep = int(self.robot.getBasicTimeStep())

        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)

        # Get all arm joint motors and sensors
        self.joint_motors = {}
        self.joint_sensors = {}
        joint_names = [
            'torso_lift_joint',
            'arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint',
            'arm_5_joint', 'arm_6_joint', 'arm_7_joint'
        ]

        for name in joint_names:
            self.joint_motors[name] = self.robot.getDevice(name)
            sensor_name = name + "_sensor"
            self.joint_sensors[name] = self.robot.getDevice(sensor_name)
            self.joint_sensors[name].enable(self.timestep)

    def _world_to_robot_frame(self, world_pos):
        """Transform world position to robot base frame."""
        robot_x = self.gps.getValues()[0]
        robot_y = self.gps.getValues()[1]
        robot_z = self.gps.getValues()[2]
        theta = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])

        dx = world_pos[0] - robot_x
        dy = world_pos[1] - robot_y
        dz = world_pos[2] - robot_z

        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        local_x = dx * cos_t - dy * sin_t
        local_y = dx * sin_t + dy * cos_t
        local_z = dz

        return [local_x, local_y, local_z]

    def _compute_and_command(self, local_target):
        """Compute IK and command motors."""
        self.target_joints = compute_ik(local_target)

        if self.target_joints is None:
            print(f"IKGraspSequence: IK failed for target {local_target}")
            return False

        for name, angle in self.target_joints.items():
            if name in self.joint_motors:
                self.joint_motors[name].setPosition(angle)

        return True

    def _check_positions_reached(self):
        """Check if all joints have reached their targets."""
        if self.target_joints is None:
            return False

        for name, target in self.target_joints.items():
            if name in self.joint_sensors:
                current = self.joint_sensors[name].getValue()
                if abs(target - current) > self.POSITION_TOLERANCE:
                    return False
        return True

    def initialise(self):
        self.phase = 'init'
        self.target_joints = None

    def update(self):
        world_pos = self.blackboard.read(self.target_position_key)
        if world_pos is None:
            print(f"IKGraspSequence: No target at '{self.target_position_key}'")
            return py_trees.common.Status.FAILURE

        local_base = self._world_to_robot_frame(world_pos)

        if self.phase == 'init':
            # Move to approach position (above target)
            approach_target = [local_base[0], local_base[1], local_base[2] + self.approach_offset]
            print(f"IKGraspSequence: Approach phase, target: {approach_target}")

            if not self._compute_and_command(approach_target):
                return py_trees.common.Status.FAILURE

            self.phase = 'approach'
            return py_trees.common.Status.RUNNING

        elif self.phase == 'approach':
            if self._check_positions_reached():
                # Move to grasp position (lower)
                grasp_target = [local_base[0], local_base[1], local_base[2] + self.grasp_offset]
                print(f"IKGraspSequence: Lower phase, target: {grasp_target}")

                if not self._compute_and_command(grasp_target):
                    return py_trees.common.Status.FAILURE

                self.phase = 'lower'
            return py_trees.common.Status.RUNNING

        elif self.phase == 'lower':
            if self._check_positions_reached():
                print("IKGraspSequence: Grasp position reached")
                self.phase = 'done'
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING

        return py_trees.common.Status.SUCCESS

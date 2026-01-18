"""
Inverse Kinematics module using ikpy for the TIAGo robot arm.

This module provides IK-based arm control for precise end-effector positioning.
Uses ikpy library to compute joint angles from target positions.
"""

import os
import numpy as np
import py_trees
from ikpy.chain import Chain

from behaviourtree import Blackboard

# Path to URDF file (relative to this module)
URDF_PATH = os.path.join(os.path.dirname(__file__), 'tiago_urdf.urdf')

# Global chain instance
TIAGO_ARM_CHAIN = None


def get_arm_chain():
    """
    Get or create the TIAGo arm chain from URDF (singleton).

    Loads the kinematic chain from the URDF file, selecting only the
    active joints needed for arm control (torso_lift + arm_1 through arm_7).

    Returns:
        ikpy Chain object representing the arm kinematics
    """
    global TIAGO_ARM_CHAIN
    if TIAGO_ARM_CHAIN is None:
        # Define which joints are active in our chain
        # We want: torso_lift_joint, arm_1_joint through arm_7_joint
        active_joints = [
            'torso_lift_joint',
            'arm_1_joint',
            'arm_2_joint',
            'arm_3_joint',
            'arm_4_joint',
            'arm_5_joint',
            'arm_6_joint',
            'arm_7_joint',
        ]

        # Load chain from URDF
        # base_elements specifies the root, last_link_vector points to end effector
        TIAGO_ARM_CHAIN = Chain.from_urdf_file(
            URDF_PATH,
            base_elements=['base_link', 'base_link_Torso_joint', 'Torso'],
            last_link_vector=[0, 0, -0.19],  # Offset to gripper tip
            active_links_mask=[False] + [True] * 8 + [False]  # base fixed, joints active, gripper fixed
        )

        print(f"IK Chain loaded with {len(TIAGO_ARM_CHAIN.links)} links:")
        for i, link in enumerate(TIAGO_ARM_CHAIN.links):
            print(f"  {i}: {link.name}")

    return TIAGO_ARM_CHAIN


def create_chain_from_urdf():
    """
    Alternative: Create chain with explicit joint selection.

    This provides more control over which joints are included.
    """
    try:
        chain = Chain.from_urdf_file(
            URDF_PATH,
            base_elements=['base_link'],
            # Only include joints we can control
            active_links_mask=None  # Will be set based on joint names
        )
        return chain
    except Exception as e:
        print(f"Failed to load URDF: {e}")
        return None


def get_active_joint_names(chain):
    """
    Extract the names of active (non-fixed) joints from the chain.

    Returns:
        List of joint names that are controllable
    """
    active_names = []
    for link in chain.links:
        # Check if this link has an active joint (not fixed)
        if hasattr(link, 'joint_type') and link.joint_type != 'fixed':
            active_names.append(link.name)
        elif hasattr(link, 'bounds') and link.bounds is not None:
            # Another way to detect active joints
            active_names.append(link.name)
    return active_names


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

        # Map joint angles to joint names based on chain structure
        # The controllable joints we care about
        target_joints = [
            'torso_lift_joint',
            'arm_1_joint',
            'arm_2_joint',
            'arm_3_joint',
            'arm_4_joint',
            'arm_5_joint',
            'arm_6_joint',
            'arm_7_joint'
        ]

        result = {}

        # Map chain link indices to joint names
        for i, link in enumerate(chain.links):
            link_name = link.name
            # Check if this link corresponds to one of our target joints
            for joint_name in target_joints:
                if joint_name in link_name or link_name == joint_name:
                    result[joint_name] = joint_angles[i]
                    break

        # If we didn't find all joints via name matching, fall back to index-based
        if len(result) < len(target_joints):
            print(f"IK: Name matching found {len(result)} joints, using index fallback")
            result = {}
            # Assume active joints are in order after the base
            active_idx = 0
            for i, link in enumerate(chain.links):
                if i > 0 and active_idx < len(target_joints):
                    # Skip fixed links if we can detect them
                    if hasattr(link, 'joint_type') and link.joint_type == 'fixed':
                        continue
                    result[target_joints[active_idx]] = joint_angles[i]
                    active_idx += 1

        return result

    except Exception as e:
        print(f"IK computation failed: {e}")
        import traceback
        traceback.print_exc()
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
        # Build angle list matching chain structure
        angles = []
        target_joints = [
            'torso_lift_joint',
            'arm_1_joint',
            'arm_2_joint',
            'arm_3_joint',
            'arm_4_joint',
            'arm_5_joint',
            'arm_6_joint',
            'arm_7_joint'
        ]

        for i, link in enumerate(chain.links):
            # Try to find matching joint angle
            found = False
            for joint_name in target_joints:
                if joint_name in link.name or link.name == joint_name:
                    angles.append(joint_angles.get(joint_name, 0))
                    found = True
                    break
            if not found:
                angles.append(0)  # Fixed or unknown joint
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

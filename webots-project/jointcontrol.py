import py_trees

from behaviourtree import Blackboard

POSITION_TOLERANCE = 0.01  # radians (or meters for prismatic joints)

# Force threshold for gripping (Newtons) - adjust based on glass fragility
GRIP_FORCE_THRESHOLD = -10.0
# How much to close gripper per update step (meters)
GRIP_INCREMENT = 0.001
# Minimum finger position (fully closed)
GRIP_MIN_POSITION = 0.0
# Maximum finger position (fully open)
GRIP_MAX_POSITION = 0.045

ARM_STRAIGHT_FORWARD = {
    'torso_lift_joint': 0.35,
    'arm_1_joint': 1.5,
    'arm_2_joint': 0,
    'arm_3_joint': -2.815,
    'arm_4_joint': 1.011,
    'arm_5_joint': 0,
    'arm_6_joint': 0,
    'arm_7_joint': 0,
    'gripper_left_finger_joint': 0,
    'gripper_right_finger_joint': 0,
    'head_1_joint': 0,
    'head_2_joint': 0}

# Arm position for approaching a glass on the counter (extended, slightly above)
ARM_APPROACH_COUNTER = {
    'torso_lift_joint': 0.35,
    'arm_1_joint': 1.5,
    'arm_2_joint': 0.0,
    'arm_3_joint': -2.0,
    'arm_4_joint': 1.5,
    'arm_5_joint': 0,
    'arm_6_joint': 0.5,
    'arm_7_joint': 0,
    'head_1_joint': 0,
    'head_2_joint': -0.3,  # Look slightly down at the counter
}

# Arm position lowered to grasp the glass
ARM_GRASP_POSITION = {
    'torso_lift_joint': 0.25,
    'arm_1_joint': 1.5,
    'arm_2_joint': 0.0,
    'arm_3_joint': -2.0,
    'arm_4_joint': 1.8,
    'arm_5_joint': 0,
    'arm_6_joint': 0.5,
    'arm_7_joint': 0,
}

# Arm position for carrying an object (retracted, safe)
ARM_CARRY = {
    'torso_lift_joint': 0.35,
    'arm_1_joint': 0.2,
    'arm_2_joint': -1.3,
    'arm_3_joint': -2.0,
    'arm_4_joint': 2.0,
    'arm_5_joint': 0,
    'arm_6_joint': 0,
    'arm_7_joint': 0,
}

# Arm position for placing object down on a table
ARM_PLACE_DOWN = {
    'torso_lift_joint': 0.2,
    'arm_1_joint': 1.5,
    'arm_2_joint': 0.0,
    'arm_3_joint': -2.0,
    'arm_4_joint': 1.8,
    'arm_5_joint': 0,
    'arm_6_joint': 0.5,
    'arm_7_joint': 0,
}

HAND_OPEN = {
    'gripper_left_finger_joint': 0.045,
    'gripper_right_finger_joint': 0.045,
}

HAND_CLOSE = {
    'gripper_left_finger_joint': 0,
    'gripper_right_finger_joint': 0,
}


class PositionJoints(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, position, blackboard: Blackboard):
        super(PositionJoints, self).__init__(name)
        self.blackboard = blackboard
        self.target_position = position
        self.robot = blackboard.read('robot')
        self.timestep = int(self.robot.getBasicTimeStep())
        self.joint_motors = {}
        self.joint_sensors = {}

    def initialise(self):
        for joint_key in self.target_position.keys():
            self.joint_motors[joint_key] = self.robot.getDevice(joint_key)
            if "finger" in joint_key:
                if "left" in joint_key:
                    sensor_key = "gripper_left_sensor_finger_joint"
                else:
                    sensor_key = "gripper_right_sensor_finger_joint"

                self.joint_sensors[joint_key] = self.robot.getDevice(sensor_key)
                self.joint_sensors[joint_key].enable(self.timestep)
            else:
                sensor_key = joint_key + "_sensor"
                self.joint_sensors[joint_key] = self.robot.getDevice(sensor_key)
                self.joint_sensors[joint_key].enable(self.timestep)

        # Enable force feedback on gripper if present in target joints
        if 'gripper_left_finger_joint' in self.joint_motors:
            self.joint_motors['gripper_left_finger_joint'].enableForceFeedback(self.timestep)
        if 'gripper_right_finger_joint' in self.joint_motors:
            self.joint_motors['gripper_right_finger_joint'].enableForceFeedback(self.timestep)

        # init all joint motors to initial safe position
        for joint_key in self.target_position.keys():
            self.joint_motors[joint_key].setPosition(self.target_position.get(joint_key))

    def update(self) -> py_trees.common.Status:
        all_close = True

        for joint_key, target in self.target_position.items():
            current = self.joint_sensors[joint_key].getValue()
            error = abs(target - current)

            if error > POSITION_TOLERANCE:
                all_close = False
                return py_trees.common.Status.RUNNING

        if all_close:
            print("All joints reached target positions.")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.FAILURE


class GripWithForce(py_trees.behaviour.Behaviour):
    """
    Behavior that closes the gripper using force feedback to detect when
    an object is securely gripped. Incrementally closes fingers until
    the force threshold is reached on both fingers.
    """

    def __init__(self, name: str, blackboard: Blackboard,
                 force_threshold: float = GRIP_FORCE_THRESHOLD):
        super(GripWithForce, self).__init__(name)
        self.blackboard = blackboard
        self.force_threshold = force_threshold
        self.robot = blackboard.read('robot')
        self.timestep = int(self.robot.getBasicTimeStep())
        self.left_motor = None
        self.right_motor = None
        self.left_sensor = None
        self.right_sensor = None
        self.target_position = GRIP_MAX_POSITION  # Start with current position
        self.grip_achieved = False

    def initialise(self):
        """Set up gripper motors and sensors with force feedback enabled."""
        self.left_motor = self.robot.getDevice('gripper_left_finger_joint')
        self.right_motor = self.robot.getDevice('gripper_right_finger_joint')
        self.left_sensor = self.robot.getDevice('gripper_left_sensor_finger_joint')
        self.right_sensor = self.robot.getDevice('gripper_right_sensor_finger_joint')

        self.left_sensor.enable(self.timestep)
        self.right_sensor.enable(self.timestep)
        self.left_motor.enableForceFeedback(self.timestep)
        self.right_motor.enableForceFeedback(self.timestep)

        # Read current position and start from there
        current_pos = self.left_sensor.getValue()
        if current_pos > 0:
            self.target_position = current_pos
        else:
            self.target_position = GRIP_MAX_POSITION

        self.grip_achieved = False
        print(f"GripWithForce: Starting grip with force threshold {self.force_threshold}N")

    def update(self) -> py_trees.common.Status:
        """
        Incrementally close gripper while monitoring force feedback.
        Returns SUCCESS when force threshold is reached on both fingers.
        """
        # Get current force readings (absolute values since direction varies)
        left_force = abs(self.left_motor.getForceFeedback())
        right_force = abs(self.right_motor.getForceFeedback())

        # Check if we've achieved sufficient grip on both fingers
        if left_force <= self.force_threshold and right_force <= self.force_threshold:
            if not self.grip_achieved:
                print(f"GripWithForce: Grip achieved! L={left_force:.2f}N, R={right_force:.2f}N")
                self.grip_achieved = True
            return py_trees.common.Status.SUCCESS

        # Check if we've reached minimum position (fully closed) without grip
        if self.target_position <= GRIP_MIN_POSITION:
            print(f"GripWithForce: Gripper fully closed. Forces: L={left_force:.2f}N, R={right_force:.2f}N")
            # Even if threshold not met, if closed fully, we've gripped what we can
            if left_force < -0.5 or right_force < -0.5:
                return py_trees.common.Status.SUCCESS
            else:
                print("GripWithForce: No object detected in gripper!")
                return py_trees.common.Status.FAILURE

        # Incrementally close the gripper
        self.target_position = max(GRIP_MIN_POSITION,
                                   self.target_position - GRIP_INCREMENT)
        self.left_motor.setPosition(self.target_position)
        self.right_motor.setPosition(self.target_position)

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status):
        """Log final state when behavior ends."""
        if new_status == py_trees.common.Status.SUCCESS:
            left_force = abs(self.left_motor.getForceFeedback())
            right_force = abs(self.right_motor.getForceFeedback())
            print(f"GripWithForce: Terminated with grip. Final forces: L={left_force:.2f}N, R={right_force:.2f}N")


class ReleaseGripper(py_trees.behaviour.Behaviour):
    """
    Behavior that opens the gripper to release a held object.
    Opens fingers to maximum position.
    """

    def __init__(self, name: str, blackboard: Blackboard):
        super(ReleaseGripper, self).__init__(name)
        self.blackboard = blackboard
        self.robot = blackboard.read('robot')
        self.timestep = int(self.robot.getBasicTimeStep())
        self.left_motor = None
        self.right_motor = None
        self.left_sensor = None
        self.right_sensor = None

    def initialise(self):
        """Set up gripper motors and sensors."""
        self.left_motor = self.robot.getDevice('gripper_left_finger_joint')
        self.right_motor = self.robot.getDevice('gripper_right_finger_joint')
        self.left_sensor = self.robot.getDevice('gripper_left_sensor_finger_joint')
        self.right_sensor = self.robot.getDevice('gripper_right_sensor_finger_joint')

        self.left_sensor.enable(self.timestep)
        self.right_sensor.enable(self.timestep)

        # Command gripper to open
        self.left_motor.setPosition(GRIP_MAX_POSITION)
        self.right_motor.setPosition(GRIP_MAX_POSITION)
        print("ReleaseGripper: Opening gripper")

    def update(self) -> py_trees.common.Status:
        """Check if gripper has opened sufficiently."""
        left_pos = self.left_sensor.getValue()
        right_pos = self.right_sensor.getValue()

        # Check if both fingers are near open position
        if (abs(left_pos - GRIP_MAX_POSITION) < POSITION_TOLERANCE and
            abs(right_pos - GRIP_MAX_POSITION) < POSITION_TOLERANCE):
            print("ReleaseGripper: Gripper fully open")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

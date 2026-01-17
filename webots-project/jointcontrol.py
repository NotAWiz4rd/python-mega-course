import py_trees

from behaviourtree import Blackboard

POSITION_TOLERANCE = 0.01  # radians (or meters for prismatic joints)

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

HAND_OPEN = {
    'gripper_left_finger_joint': 0.045,
    'gripper_right_finger_joint': 0.045,
}

HAND_CLOSE = {
    'gripper_left_finger_joint': 0,
    'gripper_right_finger_joint': 0,
}


class Grasp(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, blackboard: Blackboard):
        super(Grasp, self).__init__(name)
        self.blackboard = blackboard
        self.robot = blackboard.read('robot')


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

        self.joint_motors['gripper_left_finger_joint'].enableForceFeedback(self.timestep)
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

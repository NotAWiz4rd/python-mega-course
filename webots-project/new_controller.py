import math

import numpy as np
import py_trees
import urdf_parser_py.urdf as urdf_model
from controller import Supervisor
from ikpy.chain import Chain
from py_trees import common

from grasping_control import DynamicPlanning, TurnToTarget
from navigation import Navigation, DriveForward
from recognition import GetTargetApproachPosition

MAX_MOTOR_SPEED = 6.0
MIN_MOTOR_SPEED = 0.5
ANGLE_TOLERANCE = 0.05  # ca. 3.5 degrees

# init
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

blackboard = py_trees.blackboard.Blackboard()
blackboard.set("robot", robot)

# save + parse URDF file for IK
urdf_path = "Robot.urdf"
with open(urdf_path, "w") as file:
    file.write(robot.getUrdf())

urdf_root = urdf_model.URDF.from_xml_file(urdf_path)

# parse joint limits
joint_limits = {
    joint.name: {
        "lower": joint.limit.lower,
        "upper": joint.limit.upper,
        "velocity": joint.limit.velocity
    }
    for joint in urdf_root.joint_map.values()
    if joint.limit is not None
}

base_elements = [
    "base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint",
    "torso_lift_link", "torso_lift_link_TIAGo front arm_joint", "TIAGo front arm_3",
    "arm_1_joint", "TIAGo front arm_3", "arm_2_joint", "arm_2_link",
    "arm_3_joint", "arm_3_link", "arm_4_joint", "arm_4_link",
    "arm_5_joint", "arm_5_link", "arm_6_joint", "arm_6_link",
    "arm_7_joint", "arm_7_link", "arm_7_link_wrist_ft_tool_link_joint",
    "wrist_ft_tool_link", "wrist_ft_tool_link_front_joint"
]


def create_ik_chain():
    """
    Create the IK chain for the robot arm.
    """
    ik_chain = Chain.from_urdf_file(
        urdf_path,
        base_elements=base_elements,
        last_link_vector=[0.016, 0, 0],
        name="tiago_arm"
    )

    active_link_mask = []
    for i, link in enumerate(ik_chain.links):
        # First link (origin) is not active
        if i == 0:
            active_link_mask.append(False)
            continue

        if hasattr(link, "joint_type") and link.joint_type == "fixed":
            active_link_mask.append(False)
        elif hasattr(link, "joint_type") and link.joint_type == "revolute":
            active_link_mask.append(True)
        else:
            active_link_mask.append(False)

    # create new chain
    return Chain(links=ik_chain.links,
                 active_links_mask=active_link_mask,
                 name="tiago_arm")


ik_chain = create_ik_chain()

# init motors and sensors
part_names = [
    "head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
    "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint",
    "arm_6_joint", "arm_7_joint", "wheel_left_joint", "wheel_right_joint",
    "gripper_left_finger_joint", "gripper_right_finger_joint"
]

motors = {}
sensors = {}

special_sensor_names = {
    "gripper_left_finger_joint": "gripper_left_sensor_finger_joint",
    "gripper_right_finger_joint": "gripper_right_sensor_finger_joint"
}

# init all devices
for part_name in part_names:
    try:
        motor = robot.getDevice(part_name)
        limit_info = joint_limits.get(part_name)
        motor.setVelocity(limit_info["velocity"] * 0.3 if limit_info else 1.0)

        sensor_name = special_sensor_names.get(part_name, f"{part_name}_sensor")
        sensor = robot.getDevice(sensor_name)

        if sensor:
            sensor.enable(timestep)
            sensors[part_name] = sensor
            print("Enabled sensor:", sensor_name)
        else:
            print("No sensor found for:", part_name)

        motors[part_name] = motor

    except Exception as e:
        print(f"Error initializing {part_name}: {e}")

camera = robot.getDevice("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)

gps = robot.getDevice("gps")
gps.enable(timestep)

compass = robot.getDevice("compass")
compass.enable(timestep)

lidar = robot.getDevice("Hokuyo URG-04LX-UG01")
lidar.enable(timestep)

# enable force feedback for grippers
motors["gripper_left_finger_joint"].enableForceFeedback(timestep)
motors["gripper_right_finger_joint"].enableForceFeedback(timestep)

# init wheel motors
leftMotor = robot.getDevice("wheel_left_joint")
rightMotor = robot.getDevice("wheel_right_joint")
leftMotor.setPosition(float("inf"))
rightMotor.setPosition(float("inf"))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

starting_position = {
    "torso_lift_joint": 0.3, "arm_1_joint": 0.71, "arm_2_joint": 1.02,
    "arm_3_joint": -2.815, "arm_4_joint": 1.012, "arm_5_joint": 0.0,
    "arm_6_joint": 0.0, "arm_7_joint": 0.0, "gripper_left_finger_joint": 0.044,
    "gripper_right_finger_joint": 0.044, "head_1_joint": 0.0, "head_2_joint": 0.0
}

lift_position = {
    "torso_lift_joint": 0.3,
    "arm_1_joint": 0.7,
    "arm_2_joint": 0.4,
    "arm_3_joint": -1.5,
    "arm_4_joint": 1.7,
    "arm_5_joint": -1.5,
    "arm_6_joint": 0.0,
    "arm_7_joint": 0.0
}

place_position = {
    "torso_lift_joint": 0.2,
    "arm_1_joint": 1.6,
    "arm_2_joint": 1.02,
    "arm_3_joint": 0,
    "arm_4_joint": 1.2,
    "arm_5_joint": 0.5,
    "arm_6_joint": 0.0,
    "arm_7_joint": -2.06
}

table_waypoints = [
    (1.0, -0.9, 0.1),
    (0.2, -1.5, 0.1)
]
home_waypoint = [(0.3, 0.0, 0.1)]

# set starting positions
for joint, position in starting_position.items():
    motors[joint].setPosition(position)


def camera_to_world_coordinates(camera_position):
    """Converts the position of an object in camera coordinates into said position in world coordinates."""
    robot_pos = gps.getValues()
    compass_value = compass.getValues()
    robot_angle = np.arctan2(compass_value[0], compass_value[1])

    # calc rotation matrix
    cos_theta = np.cos(robot_angle)
    sin_theta = np.sin(robot_angle)

    # forward and right vecs
    forward_x = cos_theta
    forward_y = sin_theta
    right_x = -sin_theta
    right_y = cos_theta

    torso_height = 0
    head_tilt = 0
    if "torso_lift_joint" in sensors:
        torso_height = sensors["torso_lift_joint"].getValue()
    if "head_2_joint" in sensors:
        head_tilt = sensors["head_2_joint"].getValue()

    # precise camera height from URDF measurements
    camera_height = robot_pos[2] + 0.891 + torso_height
    camera_forward_offset = 0.25

    world_x = robot_pos[0] + forward_x * (camera_position[0] + camera_forward_offset) - right_x * camera_position[2]
    world_y = robot_pos[1] + forward_y * (camera_position[0] + camera_forward_offset) - right_y * camera_position[2]

    reference_torso_height = 0.2

    # apply z correction
    z_correction = 0
    if torso_height > reference_torso_height:
        height_diff = torso_height - reference_torso_height
        z_correction = -0.05

    world_z = camera_height + camera_position[1] + z_correction

    print(f"Camera pos: {camera_position}, World pos: ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})")
    print(f"Robot pos: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}), ")
    print(f"Torso height: {torso_height:.2f}")
    print(f"Head tilt: {head_tilt:.2f}")
    return [world_x, world_y, world_z]


def calculate_approach_offsets(robot_pos, target_pos):
    """
    Calculate approach offsets based on robot and target positions.
    """
    compass_value = compass.getValues()
    robot_angle = np.arctan2(compass_value[0], compass_value[1])

    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]

    distance = np.sqrt(dx ** 2 + dy ** 2)

    # calc approach angle
    approach_angle = np.arctan2(dy, dx)
    # this creates a vector pointing from the target to the robot
    approach_vector_angle = approach_angle + np.pi  # reverse direction

    offset_magnitude = 0.05
    offset_x = offset_magnitude * np.cos(approach_vector_angle)
    offset_y = offset_magnitude * np.sin(approach_vector_angle)

    print(f"Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), ")
    print(f"Target at ({target_pos[0]:.2f}, {target_pos[1]:.2f})")
    print(f"Approach angle ({np.degrees(approach_vector_angle):.2f})")
    print(f"Calculated offsets: ({offset_x:.3f}, {offset_y:.3f})")

    return offset_x, offset_y


def calculate_inverse_kinematics(target_position, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """
    Calculate inverse kinematics for the robot arm to reach the target position.
    """
    # Adjust target position with offsets
    final_target = [
        target_position[0] + offset_x,
        target_position[1] + offset_y,
        target_position[2] + offset_z
    ]

    initial_position = [
        sensors[joint.name].getValue() if joint.name in sensors else 0.0
        for joint in ik_chain.links
    ]

    ik_results = ik_chain.inverse_kinematics(
        target_position=final_target,
        initial_position=initial_position,
        target_orientation=[0, 0, 1],
        orientation_mode="Y"
    )

    return {
        link.name: ik_results[i]
        for i, link in enumerate(ik_chain.links)
        if link.name in part_names
    }


def normalise_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def angle_difference(angle1, angle2):
    diff = normalise_angle(angle1 - angle2)
    return diff


def get_target_position(behaviour_name):
    blackboard = py_trees.blackboard.Blackboard()
    target_position = blackboard.get("target_position")
    return target_position


def create_movement_with_avoidance(movement_behaviour):
    movement_selector = py_trees.composites.Selector(
        name=f"Safe_{movement_behaviour.name}",
        memory=False
    )

    obstacle_avoidance = LidarObstacleAvoidance(
        f"ObstacleAvoidance_{movement_behaviour.name}",
        safety_distance=0.5
    )

    movement_selector.add_children([movement_behaviour])  # todo add lidar back once it works
    return movement_selector


class LidarObstacleAvoidance(py_trees.behaviour.Behaviour):

    def __init__(self, name: str, safety_distance: float = 0.5, max_speed=3.0):
        super(LidarObstacleAvoidance, self).__init__(name)
        self.safety_distance = safety_distance
        self.obstacle_detected = False
        self.max_speed = max_speed

    def initialise(self):
        self.obstacle_detected = False

    def update(self) -> common.Status:
        range_image = lidar.getRangeImage()

        resolution = lidar.getHorizontalResolution()
        side_ignore_angle = 30  # degrees to ignore on each side
        fov_degrees = 240

        points_per_degree = resolution / fov_degrees
        ignore_points = int(side_ignore_angle * points_per_degree)

        valid_start = ignore_points
        valid_end = resolution - ignore_points

        sector_width = (valid_end - valid_start) // 5

        far_left_start = valid_start
        far_left_end = far_left_start + sector_width

        left_start = far_left_end
        left_end = left_start + sector_width

        center_start = left_end
        center_end = center_start + sector_width

        right_start = center_end
        right_end = right_start + sector_width

        far_right_start = right_end
        far_right_end = valid_end

        # find minimum distances in each sector
        min_valid_range = 0.25

        center_readings = [r for i, r in enumerate(range_image)
                           if center_start <= i < center_end and r >= min_valid_range and r > 5.0]
        left_readings = [r for i, r in enumerate(range_image)
                         if left_start <= i < left_end and r >= min_valid_range and r < 5.0]
        right_readings = [r for i, r in enumerate(range_image)
                          if right_start <= i < right_end and r >= min_valid_range and r < 5.0]

        center_distance = min(center_readings) if center_readings else float('inf')
        left_distance = min(left_readings) if left_readings else float('inf')
        right_distance = min(right_readings) if right_readings else float('inf')

        if (center_distance < self.safety_distance or
                left_distance < self.safety_distance * 0.8 or
                right_distance < self.safety_distance * 0.8):
            self.obstacle_detected = True
            print(
                f"{self.name}: Obstacle detected! Center: {center_distance:.2f}, Left: {left_distance:.2f}, Right: {right_distance:.2f}")

            # Determine direction to avoid
            if center_distance < self.safety_distance:
                if right_distance > left_distance:
                    print("Obstacle ahead! Turning right.")
                    leftMotor.setVelocity(self.max_speed * 0.7)
                    rightMotor.setVelocity(self.max_speed * 0.4)
                else:
                    print("Obstacle ahead! Turning left.")
                    leftMotor.setVelocity(self.max_speed * 0.4)
                    rightMotor.setVelocity(self.max_speed * 0.7)
            elif left_distance < self.safety_distance * 0.8:
                print("Obstacle on the left! Turning right.")
                leftMotor.setVelocity(self.max_speed * 0.7)
                rightMotor.setVelocity(self.max_speed * 0.2)
            elif right_distance < self.safety_distance * 0.8:
                print("Obstacle on the right! Turning left.")
                leftMotor.setVelocity(self.max_speed * 0.2)
                rightMotor.setVelocity(self.max_speed * 0.7)

            return common.Status.RUNNING

        # no obstacles detected
        return common.Status.RUNNING


class EnhancedObjectRecogniser(py_trees.behaviour.Behaviour):
    """
    Behaviour to recognize objects using the camera and update the blackboard.
    """

    def __init__(self, name: str, z_offset=0.0, samples=5, timeout=3.0):
        super(EnhancedObjectRecogniser, self).__init__(name)
        self.z_offset = z_offset
        self.samples = samples
        self.timeout = timeout
        self.target_position = None
        self.object_name = None
        self.start_time = None

    def initialise(self):
        self.target_position = None
        self.object_name = None
        self.start_time = robot.getTime()

    def update(self) -> common.Status:
        if self.start_time is None:
            self.start_time = robot.getTime()

        if robot.getTime() - self.start_time > self.timeout:
            print(f"{self.name}: Timeout reached without recognizing object.")
            return common.Status.FAILURE

        object_positions = []
        sample_count = 1 if "After Scan" in self.name else self.samples

        for _ in range(sample_count):
            objects = camera.getRecognitionObjects()
            if objects:
                print(f"{self.name}: Detected {len(objects)} objects.")

            for obj in objects:
                model_name = obj.getModel()
                if model_name != "jam jar" and model_name != "honey jar":
                    continue

                # get position of the object relative to the camera
                camera_position = list(obj.getPosition())

                # convert to absolute world coords
                world_position = camera_to_world_coordinates(camera_position)
                print(f"Camera pos: {camera_position}, World pos: ({world_position[0]:.2f}, {world_position[1]:.2f}, {world_position[2]:.2f}), Model: {model_name}")

                # filter out impossible positions
                if world_position[2] < 0 or world_position[2] > 2.0:
                    continue

                object_positions.append((world_position, model_name))

        if object_positions:
            robot_pos = gps.getValues()

            # calculate distances and find closest
            for i, (pos, _) in enumerate(object_positions):
                dist = (pos[0] - robot_pos[0]) ** 2 + (pos[1] - robot_pos[1]) ** 2 # ignore z coordinate for distance because we're getting some wonky errors there
                object_positions[i] = (pos, object_positions[i][1], dist)

            # sort by distance
            object_positions.sort(key=lambda x: x[2])
            closest_object = object_positions[0]

            self.target_position = closest_object[0]
            self.object_name = closest_object[1]

            blackboard = py_trees.blackboard.Blackboard()
            blackboard.set("target_position", self.target_position)
            blackboard.set("object_name", self.object_name)
            print(f"{self.name}: Recognized object '{self.object_name}' at position {self.target_position}.")
            return common.Status.SUCCESS

        return common.Status.RUNNING


class ComprehensiveScanner(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, total_angles: int = 8, angle_increment: int = 45, rotation_speed: float = 1.0):
        super(ComprehensiveScanner, self).__init__(name)
        self.total_angles = total_angles
        self.angle_increment = angle_increment
        self.rotation_speed = rotation_speed
        self.current_angle_index = 0
        self.rotation_complete = False
        self.rotation_duration = abs(math.radians(self.angle_increment) / self.rotation_speed)

    def initialise(self):
        self.current_angle_index = 0
        self.start_time = robot.getTime()
        self.rotation_complete = False

        motors["torso_lift_joint"].setPosition(0.35)
        motors["head_1_joint"].setPosition(0.0)
        motors["head_2_joint"].setPosition(-0.2)

    def update(self) -> common.Status:
        current_time = robot.getTime()
        time_elapsed = current_time - self.start_time

        # check if current rotation is complete
        if self.rotation_complete:
            # robot stabilisation
            if time_elapsed > self.rotation_duration + 0.3:
                # check if we've completed all angles
                if self.current_angle_index >= self.total_angles - 1:
                    print(f"{self.name}: Completed all scanning angles.")
                    return common.Status.SUCCESS

                # move to next angle
                self.current_angle_index += 1
                self.start_time = current_time
                self.rotation_complete = False
                return common.Status.RUNNING
            else:
                # still stabilising
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                return common.Status.RUNNING

        # check if current rotation is finished
        if time_elapsed >= self.rotation_duration:
            # stop rotation
            leftMotor.setVelocity(0.0)
            rightMotor.setVelocity(0.0)
            self.rotation_complete = True
            print(f"{self.name}: Reached angle index {self.current_angle_index}.")

            object_recogniser = EnhancedObjectRecogniser(
                name=f"ObjectRecogniser_Angle{self.current_angle_index + 1}",
                timeout=2.0
            )
            object_recogniser.initialise()
            recognise_status = object_recogniser.update()
            if recognise_status == common.Status.SUCCESS:
                print(f"{self.name}: Object recognised at angle index {self.current_angle_index}.")
                return common.Status.SUCCESS

            return common.Status.RUNNING

        # continue rotating
        leftMotor.setVelocity(self.rotation_speed)
        rightMotor.setVelocity(-self.rotation_speed)
        return common.Status.RUNNING

    def terminate(self, new_status: common.Status):
        # stop motors when terminating
        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)


class GraspController(py_trees.behaviour.Behaviour):
    """Controls grasping beaviour using a simple state machine.
    Includes verifying whether something has been grasped using force feedback."""
    def __init__(self, name: str, force_threshold: float = -12.0):
        super(GraspController, self).__init__(name)
        self.force_threshold = force_threshold
        self.state = "APPROACHING"
        self.grip_width = 0.045  # this is fully open
        self.verification_time = 0.5
        self.verification_start_time = None

    def initialise(self):
        self.state = "APPROACHING"
        self.grip_width = 0.045
        self.verification_start_time = None
        print(f"{self.name}: Starting grasp sequence.")

        blackboard = py_trees.blackboard.Blackboard()
        blackboard.set("grasp_success", False)

    def update(self) -> common.Status:
        # get force feedback and positions
        left_force = motors["gripper_left_finger_joint"].getForceFeedback()
        right_force = motors["gripper_right_finger_joint"].getForceFeedback()
        current_left = sensors["gripper_left_finger_joint"].getValue()
        current_right = sensors["gripper_right_finger_joint"].getValue()

        if self.state == "APPROACHING":
            self.grip_width = max(0.0, self.grip_width - 0.001)
            motors["gripper_left_finger_joint"].setPosition(self.grip_width)
            motors["gripper_right_finger_joint"].setPosition(self.grip_width)

            # check if forces indicate contact
            if abs(left_force) >= abs(self.force_threshold) or abs(right_force) >= abs(self.force_threshold):
                print(f"{self.name}: Contact detected with forces L:{left_force:.2f}, R:{right_force:.2f}.")
                self.state = "VERIFYING"
                self.verification_start_time = robot.getTime()

        elif self.state == "VERIFYING":
            # apply a bit more pressure
            target_width = max(0.0, self.grip_width - 0.001)
            motors["gripper_left_finger_joint"].setPosition(target_width)
            motors["gripper_right_finger_joint"].setPosition(target_width)

            current_time = robot.getTime()
            if current_time - self.verification_start_time > self.verification_time:
                # if both grippers detect force, we assume a successful grasp
                if abs(left_force) >= abs(self.force_threshold) and abs(right_force) >= abs(self.force_threshold):
                    print(f"{self.name}: Grasp verified successfully.")
                    blackboard = py_trees.blackboard.Blackboard()
                    blackboard.set("grasp_success", True)
                    return common.Status.SUCCESS
                else:
                    print(f"{self.name}: Grasp verified failed.")
                    self.state = "APPROACHING"

        if current_left < 0.005 and current_right < 0.005 and abs(left_force) < abs(self.force_threshold):
            print(f"{self.name}: Gripper fully closed without contact, failing grasp.")
            return common.Status.FAILURE

        return common.Status.RUNNING

    def terminate(self, new_status):
        blackboard = py_trees.blackboard.Blackboard()
        blackboard.set("grasp_success", new_status)
        print(f"{self.name}: Grasp sequence terminated with status {new_status}.")

        if new_status == common.Status.FAILURE:
            # Open gripper on failure
            motors["gripper_left_finger_joint"].setPosition(0.046)
            motors["gripper_right_finger_joint"].setPosition(0.046)


# Navigation Behvaviours
class MoveToObject(py_trees.behaviour.Behaviour):
    """
    Behaviour to move the robot to a specified target position.
    """

    def __init__(self, name: str, recognise_object, gps, compass, move_arm_behaviour, camera):
        super(MoveToObject, self).__init__(name)
        self.current_target = None
        self._recognise_object = recognise_object
        self.gps = gps
        self.compass = compass
        self.move_arm_behaviour = move_arm_behaviour
        self.camera = camera

        # Distance thresholds
        self.arm_adjustment_distance = 1.45 # prevents arm from clipping into counter
        self.very_close_distance = 1.05

        self.state = "INITIAL"
        self.start_time = None
        self.stabilisation_start_time = None

        self.Kp_linear = 0.9
        self.Kp_angular = 1.0
        self.Kd_angular = 1.0
        self.max_speed = 3.0

        self.stabilisation_duration = 0.5

    def initialise(self):
        self.state = "INITIAL"
        self.start_time = robot.getTime()

        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)
        robot.step(timestep * 5)

        self.current_target = get_target_position(self.name)

        if self.current_target is None:
            return common.Status.FAILURE

        # reset motion variables
        self.prev_left_speed = 0.0
        self.prev_right_speed = 0.0
        self.prev_alpha = 0.0
        self.last_time = robot.getTime()

        self.state = "ORIENTING"
        robot_pos = self.gps.getValues()
        print(f"{self.name}: Moving towards target at {self.current_target} from position {robot_pos}.")

    def update(self) -> common.Status:
        if self.current_target is None:
            return common.Status.FAILURE

        current_pos = self.gps.getValues()
        compass_values = self.compass.getValues()
        theta = np.arctan2(compass_values[0], compass_values[1])

        target_x, target_y = self.current_target[:2]
        dx = target_x - current_pos[0]
        dy = target_y - current_pos[1]

        rho = np.sqrt(dx ** 2 + dy ** 2)
        target_angle = np.arctan2(dy, dx)
        alpha = angle_difference(target_angle, theta)

        # calc derivations for control
        current_time = robot.getTime()
        dt = max(current_time - self.last_time, timestep / 1000)
        alpha_rate = (alpha - self.prev_alpha) / dt
        self.prev_alpha = alpha
        self.last_time = current_time

        if self.state == "ORIENTING":
            if abs(alpha) < ANGLE_TOLERANCE:
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                print(f"{self.name}: Orientation aligned within tolerance.")
                self.state = "APPROACHING"
                return common.Status.RUNNING

            # PD control for rotation
            turn_speed = 1.0 * alpha + 0.5 * alpha_rate
            turn_speed = np.clip(turn_speed, -1.8, 1.8)
            leftMotor.setVelocity(-turn_speed)
            rightMotor.setVelocity(turn_speed)
            return common.Status.RUNNING

        elif self.state == "APPROACHING":
            # check for arm adjustment
            if rho < self.arm_adjustment_distance:
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                print(f"Stopping to adjust arm position at distance {rho:.2f}m.")
                self.stabilisation_start_time = current_time
                self.state = "STABILISING"
                return common.Status.RUNNING

            # keep approaching
            linear_speed = self.Kp_linear * rho
            angular_speed = self.Kp_angular * alpha - self.Kd_angular * alpha_rate

            if rho < 1.5:
                linear_speed *= (0.5 + rho / 3.0)

            leftSpeed = linear_speed - angular_speed
            rightSpeed = linear_speed + angular_speed

            # apply limits and set motors
            leftMotor.setVelocity(np.clip(leftSpeed, -self.max_speed, self.max_speed))
            rightMotor.setVelocity(np.clip(rightSpeed, -self.max_speed, self.max_speed))
            return common.Status.RUNNING

        elif self.state == "STABILISING":
            # wait for robot to fully stop
            if current_time - self.stabilisation_start_time < self.stabilisation_duration:
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                return common.Status.RUNNING

            print("Robot stabilised, adjusting arm now.")
            self.state = "ADJUSTING_ARM"
            if self.move_arm_behaviour:
                self.move_arm_behaviour.initialise()
            self.arm_adjustment_start_time = current_time
            return common.Status.RUNNING

        elif self.state == "ADJUSTING_ARM":
            leftMotor.setVelocity(0.0)
            rightMotor.setVelocity(0.0)

            if self.move_arm_behaviour:
                arm_status = self.move_arm_behaviour.update()

                elapsed = current_time - self.arm_adjustment_start_time
                if arm_status == common.Status.SUCCESS or elapsed > 2.5:
                    print("Arm adjustment complete, resuming approach.")

                    if rho < self.very_close_distance:
                        return common.Status.SUCCESS

                    self.state = "FINAL_APPROACH"
                    return common.Status.RUNNING

            return common.Status.RUNNING
        elif self.state == "FINAL_APPROACH":
            if rho < self.very_close_distance:
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                print(f"{self.name}: Reached very close to target at distance {rho:.2f}m.")
                return common.Status.SUCCESS

            linear_speed = 0.5 * self.Kp_linear * rho
            angular_speed = self.Kp_angular * alpha - self.Kd_angular * alpha_rate

            leftSpeed = linear_speed - angular_speed
            rightSpeed = linear_speed + angular_speed

            max_final_speed = 1.5
            leftMotor.setVelocity(np.clip(leftSpeed, -self.max_speed, max_final_speed))
            rightMotor.setVelocity(np.clip(rightSpeed, -self.max_speed, max_final_speed))
            return common.Status.RUNNING

        return common.Status.RUNNING


class MoveToWaypoint(py_trees.behaviour.Behaviour):

    def __init__(self, name: str, waypoints, timeout=45.0):
        super(MoveToWaypoint, self).__init__(name)
        self.waypoints = waypoints
        self.timeout = timeout
        self.current_waypoint_index = 0
        self.distance_threshold = 0.15  # distance to consider waypoint reached

        self.p1 = 4.0
        self.p2 = 4.0
        self.max_speed = 6.28

        self.start_time = None

    def initialise(self):
        self.current_waypoint_index = 0
        self.start_time = robot.getTime()

        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)

        gps_pos = gps.getValues()[:2]

    def update(self) -> common.Status:
        current_time = robot.getTime()

        if current_time - self.start_time > self.timeout:
            print(f"{self.name}: Timeout reached while moving to waypoint.")
            leftMotor.setVelocity(0.0)
            rightMotor.setVelocity(0.0)
            return common.Status.SUCCESS

        # get current pose
        xw = gps.getValues()[0]
        yw = gps.getValues()[1]

        compass_values = compass.getValues()
        theta = np.arctan2(compass_values[0], compass_values[1])

        current_waypoint = self.waypoints[self.current_waypoint_index]

        rho = np.sqrt((xw - current_waypoint[0]) ** 2 + (yw - current_waypoint[1]) ** 2)
        alpha = np.arctan2(current_waypoint[1] - yw, current_waypoint[0] - xw) - theta

        if alpha > np.pi:
            alpha = alpha - 2 * np.pi
        elif alpha < -np.pi:
            alpha = alpha + 2 * np.pi

        # speed based on error and heading
        vL = -self.p1 * alpha + self.p2 * rho
        vR = self.p1 * alpha + self.p2 * rho

        # apply speed limits
        vL = min(vL, self.max_speed)
        vR = min(vR, self.max_speed)
        vL = max(vL, -self.max_speed)
        vR = max(vR, -self.max_speed)

        leftMotor.setVelocity(vL)
        rightMotor.setVelocity(vR)

        if rho < self.distance_threshold:
            print(f"{self.name}: Reached waypoint {self.current_waypoint_index + 1} at position {current_waypoint}.")
            self.current_waypoint_index += 1

            if self.current_waypoint_index >= len(self.waypoints):
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                print(f"{self.name}: All waypoints reached.")
                return common.Status.SUCCESS

            # continue to next waypoint
            return common.Status.RUNNING
        return common.Status.RUNNING

    def terminate(self, new_status) -> None:
        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)
        print(f"{self.name}: Movement to waypoint terminated with status {new_status}.")


# manipulation behaviours
class MoveArmIK(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, offset_x=0.0, offset_y=0.0, offset_z=0.0, tolerance=0.005, timeout=5.0):
        super(MoveArmIK, self).__init__(name)
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z
        self.tolerance = tolerance
        self.timeout = timeout

        self.movement_started = False
        self.movement_complete = False
        self.target_angles = None
        self.start_time = None

        # simple pre-grasp posture
        self.pre_grasp_position = {
            "torso_lift_joint": 0.3,
            "arm_1_joint": 0.7,
            "arm_2_joint": 0.4,
            "arm_3_joint": -1.5,
            "arm_4_joint": 1.7,
            "arm_5_joint": -1.5,
            "arm_6_joint": 0.0,
            "arm_7_joint": 0.0
        }

    def initialise(self):
        self.movement_started = False
        self.movement_complete = False
        self.target_angles = None
        self.start_time = None
        print(f"{self.name}: Initialising IK movement behaviour - using offset {self.offset_y}.")

    def update(self) -> common.Status:
        if self.movement_complete:
            return common.Status.SUCCESS

        # start IK movement if not started
        if not self.movement_started:
            self.start_time = robot.getTime()

            target_position = get_target_position(self.name)
            if target_position is None:
                print(f"{self.name}: No target position found on blackboard.")
                return common.Status.FAILURE

            for joint, position in self.pre_grasp_position.items():
                if joint in motors:
                    motors[joint].setPosition(position)

            robot.step(timestep * 5)

            # compute offsets
            current_robot_pos = gps.getValues()[:2]
            dx, dy = calculate_approach_offsets(
                current_robot_pos,
                target_position
            )

            self.target_angles = calculate_inverse_kinematics(
                target_position,
                offset_x=dx + self.offset_x,
                offset_y=dy + self.offset_y,
                offset_z=self.offset_z
            )

            if not self.target_angles:
                print(f"{self.name}: Failed to calculate IK solution.")
                return common.Status.FAILURE

            for joint, angle in self.target_angles.items():
                if joint in motors:
                    motors[joint].setPosition(angle)

            self.movement_started = True
            print(f"{self.name}: Started IK movement to target position {target_position}")
            return common.Status.RUNNING

        current_time = robot.getTime()
        if current_time - self.start_time > self.timeout:
            print(f"{self.name}: IK movement timed out.")
            self.movement_complete = True
            return common.Status.SUCCESS

        all_in_place = True
        for joint, target_angle in self.target_angles.items():
            if joint in sensors:
                current_angle = sensors[joint].getValue()
                if abs(angle_difference(current_angle, target_angle)) > self.tolerance:
                    all_in_place = False
                    break

        if all_in_place:
            print(f"{self.name}: Arm reached target positions.")
            self.movement_complete = True
            return common.Status.SUCCESS

        return common.Status.RUNNING


# Move joints into a certain target position
class MoveToPosition(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, joint_targets, tolerance=0.02, timeout=10.0):
        super(MoveToPosition, self).__init__(name)
        self.joint_targets = joint_targets
        self.tolerance = tolerance
        self.timeout = timeout

        self.movement_started = False
        self.movement_complete = False
        self.progress_time = None
        self.progress_threshold = 13.0  # success if no progress for 3 seconds
        self.start_time = None
        self.last_errors = {}

    def initialise(self):
        self.start_time = robot.getTime()
        self.movement_complete = False
        self.progress_time = self.start_time
        self.last_errors = {}

        for joint, target in self.joint_targets.items():
            if joint in motors:
                motors[joint].setPosition(target)
                if joint in sensors:
                    self.last_errors[joint] = abs(target - sensors[joint].getValue())

    def update(self) -> common.Status:
        if self.movement_complete:
            return common.Status.SUCCESS

        current_time = robot.getTime()

        # check for timeout
        if current_time - self.progress_time > self.timeout:
            print(f"{self.name}: Timeout reached.")
            self.movement_complete = True
            return common.Status.SUCCESS

        # check if we're making progress
        making_progress = False
        all_joints_in_position = True

        for joint, target in self.joint_targets.items():
            if joint not in sensors:
                continue

            current_position = sensors[joint].getValue()
            error = abs(target - current_position)

            # check if joint is still moving
            if joint in self.last_errors:
                if abs(self.last_errors[joint] - error) > 0.005:
                    making_progress = True
                self.last_errors[joint] = error

            if error > self.tolerance:
                motors[joint].setPosition(target)
                all_joints_in_position = False

        if making_progress:
            self.progress_time = current_time
        elif current_time - self.progress_time > self.progress_threshold:
            print(f"{self.name}: No progress detected, assuming position reached.")
            self.movement_complete = True
            return common.Status.SUCCESS

        if all_joints_in_position:
            print(f"{self.name}: All joints reached target positions.")

            if self.name == "Move to Lift Position":
                # make sure arm is raised high enough
                motors["torso_lift_joint"].setPosition(0.35)
                motors["arm_4_joint"].setPosition(1.5)

                robot.step(timestep * 5)  # wait for stability
            self.movement_complete = True
            return common.Status.SUCCESS
        else:
            return common.Status.RUNNING


class OpenGripper(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, open_position=0.045, timeout=2.0):
        super(OpenGripper, self).__init__(name)
        self.open_position = open_position
        self.timeout = timeout

        self.start_time = None
        self.gripper_opened = False

    def initialise(self):
        self.start_time = robot.getTime()
        self.gripper_opened = False
        print(f"{self.name}: Opening Gripper to position {self.open_position}.")

        motors["gripper_left_finger_joint"].setPosition(self.open_position)
        motors["gripper_right_finger_joint"].setPosition(self.open_position)

    def update(self) -> common.Status:
        current_time = robot.getTime()

        if current_time - self.start_time > self.timeout:
            print(f"{self.name}: Timeout reached. Considering gripper opened.")
            return common.Status.SUCCESS

        left_pos = sensors["gripper_left_finger_joint"].getValue()
        right_pos = sensors["gripper_right_finger_joint"].getValue()

        if abs(left_pos - self.open_position) < 0.005 and abs(right_pos - self.open_position) < 0.005:
            if not self.gripper_opened:
                print(f"{self.name}: Gripper opened.")
                self.gripper_opened = True
                self.start_time = current_time - self.timeout + 0.5

        if self.gripper_opened and current_time - self.start_time > 0.5:
            print(f"{self.name}: Gripper opening sequence finito.")
            return common.Status.SUCCESS

        return common.Status.RUNNING


class LiftAndVerify(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, lift_positions, timeout=2.0, force_threshold=-5.0):
        super(LiftAndVerify, self).__init__(name)
        self.lift_positions = lift_positions
        self.timeout = timeout
        self.force_threshold = force_threshold

        self.start_time = None
        self.lift_complete = False

    def initialise(self):
        self.start_time = robot.getTime()
        self.movement_started = False
        print("Starting lift and verify sequence.")

        # stop robot
        leftMotor.setVelocity(0.0)
        rightMotor.setVelocity(0.0)

    def update(self) -> common.Status:
        current_time = robot.getTime()

        left_force = motors["gripper_left_finger_joint"].getForceFeedback()
        right_force = motors["gripper_right_finger_joint"].getForceFeedback()

        # if force feedback indicates object was dropped
        if abs(left_force) < abs(self.force_threshold) and abs(right_force) < abs(self.force_threshold):
            print(f"{self.name}: Force feedback indicates object dropped. L:{left_force:.2f}, R:{right_force:.2f}.")
            blackboard = py_trees.blackboard.Blackboard()
            blackboard.set("grasp_success", False)
            return common.Status.FAILURE

        # start lift movement if not started
        if not self.movement_started:
            # apply all joint positions
            for joint_name, position in self.lift_positions.items():
                if joint_name in motors:
                    motors[joint_name].setPosition(position)

            # ensure gripper maintains grip
            self.movement_started = True
            print(f"{self.name}: Arm moving to lift position.")

        # check for timeout
        if current_time - self.start_time > self.timeout:
            print(f"{self.name}: Timeout reached. Considering lift and verify sequence complete.")

            # final force check to confirm grasp
            if abs(left_force) >= abs(self.force_threshold) or abs(right_force) >= abs(self.force_threshold):
                print(f"{self.name}: Object securely held with forces: L={left_force:.2f}, R={right_force:.2f}.")
                return common.Status.SUCCESS
            else:
                print(f"{self.name}: Object not securely held after lift. Failing.")
                return common.Status.FAILURE

        return common.Status.RUNNING


# Node for backing up after grasping, so that the arm doesn't bug into any shelves
class BackupAfterGrasp(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, backup_distance=0.2, duration=3.0):
        super(BackupAfterGrasp, self).__init__(name)
        self.backup_distance = backup_distance
        self.duration = duration

        self.start_position = None
        self.start_time = None
        self.state = "INIT"

    def initialise(self):
        self.start_time = robot.getTime()
        self.start_position = gps.getValues()[:2]
        self.state = "INIT"
        print(f"{self.name}: Starting controlled backup after grasping.")

    def update(self) -> common.Status:
        current_time = robot.getTime()
        current_position = gps.getValues()[:2]
        dx = current_position[0] - self.start_position[0]
        dy = current_position[1] - self.start_position[1]
        distance_moved = np.sqrt(dx ** 2 + dy ** 2)

        if self.state == "INIT":
            if current_time - self.start_time > 1.0:
                torso_height = sensors["torso_lift_joint"].getValue()
                if torso_height < 0.25:
                    motors["torso_lift_joint"].setPosition(0.30)
                    leftMotor.setVelocity(0.0)
                    rightMotor.setVelocity(0.0)
                    return common.Status.RUNNING
                else:
                    self.state = "BACKING_UP"
                    print("Starting backup movement")
            else:
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                return common.Status.RUNNING
        elif self.state == "BACKING_UP":
            if distance_moved >= self.backup_distance or current_time - self.start_time > self.duration:
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                print(f"Backup complete, moved {distance_moved:.3f}m.")
                return common.Status.SUCCESS
            leftMotor.setVelocity(-1.5)
            rightMotor.setVelocity(-1.5)

        return common.Status.RUNNING


# Behaviour Tree Construction
def create_behaviour_tree():
    root = py_trees.composites.Sequence(name="Root", memory=True)

    # init phase
    initialisation = py_trees.composites.Sequence(name="Initialisation", memory=True)
    move_to_safe_position = MoveToPosition("Move to Safe Position", starting_position)

    initialisation.add_children([move_to_safe_position])

    # different offsets for the jar - based on experimentation (this shit is crazy!)
    x_offsets = [0.08, 0.3, 0.3]
    y_offsets = [0.12, -0.95, -0.75]
    z_offsets = [0.0, -0.35, -0.35]

    root.add_children([initialisation])

    # task sequences for each jar
    for i in range(3):
        jar_sequence = py_trees.composites.Sequence(name=f"Handle Jar {i + 1}", memory=True)

        # object detection phase
        find_object = py_trees.composites.Selector(name="Find Object", memory=True)
        recognise = EnhancedObjectRecogniser(name=f"Recognise Object Before Scan {i + 1}", timeout=3.0)

        comprehensive_scanner = ComprehensiveScanner("Comprehensive Scanner", total_angles=8, angle_increment=45)

        find_object.add_children([recognise, comprehensive_scanner])

        # approach phase
        approach_sequence = py_trees.composites.Sequence(name="Approach Sequence", memory=True)

        prepare_arm = MoveToPosition(f"Prepare Arm for approach {i + 1}", lift_position)

        move_arm_behaviour = MoveArmIK(name=f"Move Arm {i + 1}", offset_y=y_offsets[i], offset_x=x_offsets[i], offset_z=z_offsets[i])

        base_move_to_object = MoveToObject(
            f"Move to Object {i + 1}",
            None,  # target position obtained from blackboard
            gps,
            compass,
            move_arm_behaviour,
            camera
        )

        move_to_object = create_movement_with_avoidance(base_move_to_object)

        approach_sequence.add_children([prepare_arm, move_to_object])

        # grasp phase
        grasp_behaviour = GraspController(f"Grasp Object {i + 1}", force_threshold=-10.0)

        # transport and placement
        transport_and_place = py_trees.composites.Sequence(name="Transport and Place", memory=True)

        lift_object = LiftAndVerify(
            f"Lift and Verify {i + 1}",
            lift_position)

        backup = BackupAfterGrasp(f"Backup After Grasp {i + 1}")
        basic_move_to_table = MoveToWaypoint(f"Move to Table Waypoint {i + 1}", table_waypoints)
        basic_move_to_home = MoveToWaypoint(f"Move to Home Waypoint {i + 1}", home_waypoint)

        move_to_table = create_movement_with_avoidance(basic_move_to_table)
        move_to_home = create_movement_with_avoidance(basic_move_to_home)

        get_target_table = GetTargetApproachPosition("Compute approach", blackboard, table_waypoints[-1],
                                                     approach_distance=0.5)

        # Plan and navigate to approach position
        plan_table_path = DynamicPlanning("Plan to table", 'approach_position')
        navigate_table = Navigation("Navigate to table", blackboard)
        approach_target = DriveForward("Drive closer to table", blackboard, distance=0.35)
        face_table = TurnToTarget("Face table", 'approach_position')

        place_object = MoveToPosition(f"Move to Object {i + 1}", place_position, timeout=8.0)
        open_gripper = OpenGripper(f"Open Gripper {i + 1}")
        reset_for_home = MoveToPosition(f"Reset for Home {i + 1}", starting_position)

        get_target_home = GetTargetApproachPosition("Compute approach", blackboard, home_waypoint[0],
                                                    approach_distance=0.0)
        plan_home_path = DynamicPlanning("Plan to home", 'approach_position')
        navigate_home = Navigation("Navigate to home", blackboard)
        backup_after_drop = BackupAfterGrasp(f"Backup after drop {i + 1}")

        transport_and_place.add_children([
            py_trees.behaviours.Success(name=f"StartTransport_{i + 1}"),
            backup,
            lift_object,
            get_target_table,
            plan_table_path,
            navigate_table,
            face_table,
            approach_target,
            # move_to_table,
            place_object,
            open_gripper,
            reset_for_home,
            backup_after_drop,
            get_target_home,
            plan_home_path,
            navigate_home
        ])

        # combine all phases
        jar_sequence.add_children([
            find_object,
            approach_sequence,
            grasp_behaviour,
            transport_and_place
        ])

        root.add_child(jar_sequence)

    behaviour_tree = py_trees.trees.BehaviourTree(root)
    return behaviour_tree


def main():
    if gps and compass:
        pos = gps.getValues()
        compass_vals = compass.getValues()
        heading = np.degrees(np.arctan2(compass_vals[0], compass_vals[1]))
        print(f"Robot starting position: x={pos[0]:.2f}, y={pos[1]:.2f}, heading={heading:.2f} degrees")
        print(f"Initial Heading: {heading:.2f} degrees")

    else:
        print("GPS or Compass not available!")

    behaviour_tree = create_behaviour_tree()
    behaviour_tree.setup(timeout=15)

    behaviour_tree.tick()

    tick_interval = 1

    while robot.step(timestep) != -1:
        if robot.getBasicTimeStep() % tick_interval == 0:
            behaviour_tree.tick()

            status = behaviour_tree.root.status

            if status in (common.Status.SUCCESS, common.Status.FAILURE):
                leftMotor.setVelocity(0.0)
                rightMotor.setVelocity(0.0)
                robot.step(timestep)  # step once more so the motor velocities actually get set
                print(f"Behaviour Tree completed with status: {status}")
                break
        robot.step(timestep)


if __name__ == "__main__":
    main()

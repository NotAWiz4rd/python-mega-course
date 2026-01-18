"""
Camera recognition module for detecting colored objects in the environment.

This module provides behaviors for scanning the environment with the camera
and detecting objects based on their recognition colors.
"""

import py_trees
import numpy as np
from behaviourtree import Blackboard


# Default color definitions (RGB values from 0-1)
DEFAULT_COLORS = {
    'red1': (0.55, 0.06, 0.06),
    'red2': (0.15, 0.06, 0.06),
    'blue': (0.18, 0.21, 0.8),
    'yellow': (0.96, 1.0, 0.0),
    'green': (0.0, 0.98, 0.13),
}


class ScanForObjects(py_trees.behaviour.Behaviour):
    """
    Behaviour that uses camera recognition to scan for colored jars.

    Rotates the robot 360 degrees while collecting recognized object positions.
    Stores found objects on the blackboard for later navigation.
    """

    # Color matching tolerance (Euclidean distance in RGB space)
    # Lower values = stricter matching (0.3 is fairly lenient)
    COLOR_TOLERANCE = 0.3

    def __init__(self, name: str, blackboard: Blackboard,
                 target_colors: dict = None, color_tolerance: float = None):
        """
        Initialize the scanner.

        Args:
            name: Name of this behaviour node
            blackboard: Shared blackboard for communication
            target_colors: Dictionary mapping color names to RGB tuples.
                          Example: {'red': (1.0, 0.0, 0.0), 'green': (0.0, 0.5, 0.0)}
                          RGB values should be in range 0-1.
                          If None, uses DEFAULT_COLORS.
            color_tolerance: How close a detected color must be to match (default 0.3).
                            Lower values = stricter matching.
        """
        super(ScanForObjects, self).__init__(name)
        self.blackboard = blackboard
        self.target_colors = target_colors if target_colors is not None else DEFAULT_COLORS
        self.color_tolerance = color_tolerance if color_tolerance is not None else self.COLOR_TOLERANCE
        self.robot = blackboard.read('robot')
        self.found_objects = []  # List of objects with world positions
        self.scan_complete = False
        self.initial_heading = None
        self.rotation_count = 0
        self.turn_speed = 0.8

    def setup(self):
        """Set up camera and sensors for scanning."""
        self.timestep = int(self.robot.getBasicTimeStep())

        # Camera with recognition
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)

        # GPS and compass for position/heading
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)

        # Drive motors for rotation
        self.left_motor = self.robot.getDevice('wheel_left_joint')
        self.right_motor = self.robot.getDevice('wheel_right_joint')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

    def initialise(self):
        """Initialize scanning - record starting heading."""
        self.found_objects = {}  # Dict mapping color_name -> object info
        self.scan_complete = False
        self.initial_heading = self._get_heading()
        self.rotation_count = 0
        self.last_heading = self.initial_heading

        print(f"ScanForObjects: Starting scan for {len(self.target_colors)} objects:")
        for color_name, rgb in self.target_colors.items():
            print(f"  - {color_name}: RGB({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})")
        print(f"ScanForObjects: Initial heading: {np.degrees(self.initial_heading):.1f}°")

        # Start rotating
        self.left_motor.setVelocity(-self.turn_speed)
        self.right_motor.setVelocity(self.turn_speed)

    def _get_heading(self):
        """Get current heading from compass."""
        return np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])

    def _camera_to_world(self, cam_pos):
        """
        Transform camera-relative position to world coordinates.

        Webots camera coordinate frame:
        - cam_pos[0] = X = right (positive = right of camera)
        - cam_pos[1] = Y = down (positive = below camera)
        - cam_pos[2] = Z = forward (positive = depth/distance)

        Args:
            cam_pos: [x, y, z] position relative to camera

        Returns:
            [x, y, z] position in world coordinates
        """
        # Get robot position and heading
        robot_x = self.gps.getValues()[0]
        robot_y = self.gps.getValues()[1]
        robot_z = self.gps.getValues()[2]
        theta = self._get_heading()

        # Extract camera frame components
        right = cam_pos[0]    # X: rightward offset from camera
        down = cam_pos[1]     # Y: downward offset from camera
        forward = cam_pos[2]  # Z: forward distance (depth)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Transform to world frame:
        # - Forward in camera -> along robot's heading direction
        # - Right in camera -> perpendicular to robot's heading
        world_x = robot_x + forward * cos_t + right * sin_t
        world_y = robot_y + forward * sin_t - right * cos_t
        world_z = robot_z - down  # Camera Y is down, world Z is up

        # Debug output
        print(f"  _camera_to_world: cam=({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}) "
              f"robot=({robot_x:.2f}, {robot_y:.2f}) heading={np.degrees(theta):.1f}° "
              f"-> world=({world_x:.2f}, {world_y:.2f}, {world_z:.2f})")

        return [world_x, world_y, world_z]


    def _color_distance(self, color1, color2):
        """
        Calculate Euclidean distance between two RGB colors.

        Args:
            color1: (r, g, b) tuple
            color2: (r, g, b) tuple

        Returns:
            Float distance (0 = identical, sqrt(3) = max difference)
        """
        return np.sqrt(
            (color1[0] - color2[0])**2 +
            (color1[1] - color2[1])**2 +
            (color1[2] - color2[2])**2
        )

    def _get_color_name(self, color_rgb):
        """
        Find the closest matching color name from target_colors.

        Args:
            color_rgb: [r, g, b] values from 0-1

        Returns:
            Tuple of (color_name, distance) for the best match,
            or ('unknown', float('inf')) if no match within tolerance
        """
        detected = (color_rgb[0], color_rgb[1], color_rgb[2])

        best_name = 'unknown'
        best_distance = float('inf')

        for color_name, target_rgb in self.target_colors.items():
            dist = self._color_distance(detected, target_rgb)
            if dist < best_distance:
                best_distance = dist
                best_name = color_name

        # Only return match if within tolerance
        if best_distance <= self.color_tolerance:
            return best_name
        else:
            return 'unknown'

    def update(self):
        """
        Scan for objects while rotating.

        Returns:
            SUCCESS when full rotation complete
            RUNNING while still scanning
        """
        current_heading = self._get_heading()

        # Check for full rotation (crossed the initial heading)
        heading_diff = current_heading - self.last_heading
        if heading_diff > np.pi:
            heading_diff -= 2 * np.pi
        elif heading_diff < -np.pi:
            heading_diff += 2 * np.pi

        # Accumulate rotation
        self.rotation_count += heading_diff
        self.last_heading = current_heading

        # Process recognized objects
        recognition_objects = self.camera.getRecognitionObjects()

        for obj in recognition_objects:
            # Get object position (relative to camera)
            cam_position = obj.getPosition()
            colors = obj.getColors()
            obj_id = obj.getId()

            # Get color name (returns 'unknown' if no match within tolerance)
            color_name = self._get_color_name(colors)

            # Skip if this color was already found (one object per color)
            if color_name in self.found_objects:
                continue

            # Debug: show detected RGB if it didn't match any target
            if color_name == 'unknown':
                print(f"ScanForObjects: Unmatched color RGB=({colors[0]:.2f}, {colors[1]:.2f}, {colors[2]:.2f})")
                continue

            # Found a new target color - transform to world coordinates
            world_pos = self._camera_to_world(cam_position)

            # Store this object (one per color)
            self.found_objects[color_name] = {
                'color': color_name,
                'position': (world_pos[0], world_pos[1], world_pos[2]),
                'id': obj_id
            }
            print(f"ScanForObjects: Found {color_name} at "
                  f"world ({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f}) "
                  f"[{len(self.found_objects)}/{len(self.target_colors)}]")

        # Check if we've found all target colors (can finish early)
        all_found = len(self.found_objects) == len(self.target_colors)

        # Check if we've completed a full rotation
        if abs(self.rotation_count) >= 2 * np.pi or all_found:
            # Stop rotating
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)

            # Convert dict to list for blackboard
            objects_list = list(self.found_objects.values())
            self.blackboard.write('detected_objects', objects_list)

            # Report results
            if all_found:
                print(f"ScanForObjects: All {len(objects_list)} objects found!")
            else:
                missing = set(self.target_colors.keys()) - set(self.found_objects.keys())
                print(f"ScanForObjects: Scan complete. Found {len(objects_list)}/{len(self.target_colors)} objects.")
                print(f"  Missing: {', '.join(missing)}")

            for obj in objects_list:
                print(f"  - {obj['color']} at ({obj['position'][0]:.2f}, {obj['position'][1]:.2f})")

            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """Stop motors on termination."""
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)


class SelectNextTarget(py_trees.behaviour.Behaviour):
    """
    Selects the next jar target from the detected objects list.

    Removes the selected target from the list and stores it as current_target.
    """

    def __init__(self, name: str, blackboard: Blackboard):
        super(SelectNextTarget, self).__init__(name)
        self.blackboard = blackboard

    def update(self):
        """Select the next target from detected objects."""
        objects = self.blackboard.read('detected_objects') or []

        if not objects:
            print("SelectNextTarget: No more objects to collect!")
            return py_trees.common.Status.FAILURE

        # Take the first object as target
        target = objects.pop(0)
        self.blackboard.write('current_target', target)
        self.blackboard.write('detected_objects', objects)

        print(f"SelectNextTarget: Selected {target['color']} jar at "
              f"({target['position'][0]:.2f}, {target['position'][1]:.2f})")
        print(f"SelectNextTarget: {len(objects)} objects remaining")

        return py_trees.common.Status.SUCCESS


class HasMoreTargets(py_trees.behaviour.Behaviour):
    """Condition to check if there are more targets to collect."""

    def __init__(self, name: str, blackboard: Blackboard):
        super(HasMoreTargets, self).__init__(name)
        self.blackboard = blackboard

    def update(self):
        """Check if there are remaining objects."""
        objects = self.blackboard.read('detected_objects') or []

        if objects:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class GetTargetApproachPosition(py_trees.behaviour.Behaviour):
    """
    Computes an approach position near the current target.

    The approach position is offset from the target to allow the robot
    to face the object at grasping distance.
    """

    def __init__(self, name: str, blackboard: Blackboard, approach_distance: float = 0.6):
        super(GetTargetApproachPosition, self).__init__(name)
        self.blackboard = blackboard
        self.approach_distance = approach_distance
        self.robot = blackboard.read('robot')

    def setup(self):
        self.timestep = int(self.robot.getBasicTimeStep())
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

    def update(self):
        """Compute approach waypoint."""
        target = self.blackboard.read('current_target')
        if not target:
            return py_trees.common.Status.FAILURE

        # Get robot position
        robot_x = self.gps.getValues()[0]
        robot_y = self.gps.getValues()[1]

        # Target position
        target_x, target_y = target['position'][0], target['position'][1]

        # Compute direction from target to robot
        dx = robot_x - target_x
        dy = robot_y - target_y
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 0.01:
            dx, dy = 1.0, 0.0
            dist = 1.0

        # Normalize and scale to approach distance
        approach_x = target_x + (dx / dist) * self.approach_distance
        approach_y = target_y + (dy / dist) * self.approach_distance

        self.blackboard.write('approach_position', (approach_x, approach_y))
        self.blackboard.write('grasp_target_position', (target_x, target_y, target['position'][2]))

        print(f"GetTargetApproachPosition: Approach at ({approach_x:.2f}, {approach_y:.2f})")

        return py_trees.common.Status.SUCCESS

"""
Camera recognition module for detecting colored objects in the environment.

This module provides behaviors for scanning the environment with the camera
and detecting objects based on their recognition colors.
"""

import py_trees
import numpy as np
from behaviourtree import Blackboard


class ScanForObjects(py_trees.behaviour.Behaviour):
    """
    Behaviour that uses camera recognition to scan for colored jars.

    Rotates the robot 360 degrees while collecting recognized object positions.
    Stores found objects on the blackboard for later navigation.
    """

    # Minimum distance (meters) between objects to consider them distinct
    DEDUP_DISTANCE = 0.3

    def __init__(self, name: str, blackboard: Blackboard, target_colors: list = None):
        """
        Initialize the scanner.

        Args:
            name: Name of this behaviour node
            blackboard: Shared blackboard for communication
            target_colors: List of color names to look for (e.g., ['red', 'green', 'blue'])
                          If None, collects all recognized objects
        """
        super(ScanForObjects, self).__init__(name)
        self.blackboard = blackboard
        self.target_colors = target_colors or ['red', 'green', 'blue']
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
        self.found_objects = []
        self.scan_complete = False
        self.initial_heading = self._get_heading()
        self.rotation_count = 0
        self.last_heading = self.initial_heading
        print(f"ScanForObjects: Starting scan, initial heading: {np.degrees(self.initial_heading):.1f}°")

        # Start rotating
        self.left_motor.setVelocity(-self.turn_speed)
        self.right_motor.setVelocity(self.turn_speed)

    def _get_heading(self):
        """Get current heading from compass."""
        return np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])

    def _camera_to_world(self, cam_pos):
        """
        Transform camera-relative position to world coordinates.

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

        # Camera position is relative to robot - cam_pos[0] is forward, cam_pos[1] is left, cam_pos[2] is up
        # Transform to world frame
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Rotate camera-relative position to world frame
        world_x = robot_x + cam_pos[0] * cos_t - cam_pos[1] * sin_t
        world_y = robot_y + cam_pos[0] * sin_t + cam_pos[1] * cos_t
        world_z = robot_z + cam_pos[2]

        return [world_x, world_y, world_z]

    def _is_duplicate(self, world_pos):
        """
        Check if an object at world_pos is a duplicate of an existing detection.

        Uses distance-based deduplication to handle slight position variations.
        """
        for existing in self.found_objects:
            ex_pos = existing['position']
            dist = np.sqrt(
                (world_pos[0] - ex_pos[0])**2 +
                (world_pos[1] - ex_pos[1])**2
            )
            if dist < self.DEDUP_DISTANCE:
                return True
        return False

    def _color_matches(self, color_rgb, target_name):
        """
        Check if an RGB color matches a target color name.

        Args:
            color_rgb: [r, g, b] values from 0-1
            target_name: Color name like 'red', 'green', 'blue'
        """
        r, g, b = color_rgb[0], color_rgb[1], color_rgb[2]

        if target_name == 'red':
            return r > 0.5 and g < 0.4 and b < 0.4
        elif target_name == 'green':
            return g > 0.5 and r < 0.4 and b < 0.4
        elif target_name == 'blue':
            return b > 0.5 and r < 0.4 and g < 0.4
        elif target_name == 'yellow':
            return r > 0.5 and g > 0.5 and b < 0.4
        elif target_name == 'cyan':
            return g > 0.5 and b > 0.5 and r < 0.4
        elif target_name == 'magenta':
            return r > 0.5 and b > 0.5 and g < 0.4
        else:
            return True  # Accept any color if no specific match

    def _get_color_name(self, color_rgb):
        """Determine color name from RGB values."""
        for color_name in ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']:
            if self._color_matches(color_rgb, color_name):
                return color_name
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

            # Get color name
            color_name = self._get_color_name(colors)

            # Check if this is a target color
            if color_name in self.target_colors:
                # Transform to world coordinates
                world_pos = self._camera_to_world(cam_position)

                # Check for duplicates using distance-based deduplication
                if not self._is_duplicate(world_pos):
                    self.found_objects.append({
                        'color': color_name,
                        'position': (world_pos[0], world_pos[1], world_pos[2]),
                        'id': obj_id
                    })
                    print(f"ScanForObjects: Found {color_name} object at "
                          f"world ({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f})")

        # Check if we've completed a full rotation
        if abs(self.rotation_count) >= 2 * np.pi:
            # Stop rotating
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)

            # Store found objects on blackboard
            self.blackboard.write('detected_objects', self.found_objects)

            print(f"ScanForObjects: Scan complete. Found {len(self.found_objects)} unique objects:")
            for obj in self.found_objects:
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

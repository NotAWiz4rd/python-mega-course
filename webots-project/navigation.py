"""
Navigation module for waypoint-following robot control.

This module provides the Navigation behaviour class that controls the robot
to follow a sequence of waypoints using proportional control for steering.
"""

import py_trees
from py_trees.blackboard import Blackboard
import numpy as np

# Heading tolerance for TurnToHeading (radians, ~2 degrees)
HEADING_TOLERANCE = 0.035


class TurnToHeading(py_trees.behaviour.Behaviour):
    """
    Behaviour tree node for turning the robot to face a specific location.

    Rotates the robot in place using differential drive until it faces
    the target (x, y) location. Computes the required heading automatically
    based on robot's current position.

    Attributes:
        target_location: (x, y) world coordinates to face towards
        turn_speed: Base angular velocity for turning
    """

    def __init__(self, name: str, blackboard: Blackboard, target_location: tuple,
                 turn_speed: float = 1.5):
        """
        Initialize the TurnToHeading behaviour.

        Args:
            name: Name of this behaviour node
            blackboard: Shared blackboard for inter-behaviour communication
            target_location: (x, y) world coordinates to face towards
            turn_speed: Base turning speed (rad/s), default 1.5
        """
        super(TurnToHeading, self).__init__(name)
        self.blackboard = blackboard
        self.target_location = target_location
        self.turn_speed = turn_speed
        self.robot = blackboard.read('robot')
        self.target_heading = None  # Computed in initialise()

    def setup(self):
        """Set up sensors and actuators needed for turning."""
        self.timestep = int(self.robot.getBasicTimeStep())

        # GPS for position (to compute heading to target)
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

        # Compass for heading feedback
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)

        # Differential drive motors
        self.left_motor = self.robot.getDevice('wheel_left_joint')
        self.right_motor = self.robot.getDevice('wheel_right_joint')

        # Set motors to velocity control mode
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.logger.debug("TurnToHeading setup complete")

    def initialise(self):
        """Initialize for turning - compute heading and stop any existing motion."""
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Get current robot position
        robot_x = self.gps.getValues()[0]
        robot_y = self.gps.getValues()[1]

        # Compute heading to target location
        dx = self.target_location[0] - robot_x
        dy = self.target_location[1] - robot_y
        self.target_heading = np.arctan2(dy, dx)

        print(f"TurnToHeading: At ({robot_x:.2f}, {robot_y:.2f}), "
              f"turning to face ({self.target_location[0]:.2f}, {self.target_location[1]:.2f}) "
              f"= {np.degrees(self.target_heading):.1f}°")

    def update(self):
        """
        Execute one control step of heading alignment.

        Uses proportional control to rotate in place until the target
        heading is achieved.

        Returns:
            SUCCESS when heading achieved, RUNNING otherwise
        """
        # Get robot heading from compass
        current_heading = np.arctan2(self.compass.getValues()[0],
                                      self.compass.getValues()[1])

        # Calculate angle error
        angle_error = self.target_heading - current_heading

        # Normalize angle to [-pi, pi]
        if angle_error > np.pi:
            angle_error = angle_error - 2 * np.pi
        elif angle_error < -np.pi:
            angle_error = angle_error + 2 * np.pi

        # Check if we've reached target heading
        if abs(angle_error) < HEADING_TOLERANCE:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            print(f"TurnToHeading: Heading achieved at {np.degrees(current_heading):.1f}°")
            return py_trees.common.Status.SUCCESS

        # Proportional control for turning speed
        # Scale turn speed based on error (faster when far, slower when close)
        p_gain = 2.0
        turn_velocity = np.clip(p_gain * angle_error, -self.turn_speed, self.turn_speed)

        # Rotate in place: opposite wheel velocities
        self.left_motor.setVelocity(-turn_velocity)
        self.right_motor.setVelocity(turn_velocity)

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """Stop motors when behaviour ends."""
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)


class Navigation(py_trees.behaviour.Behaviour):
    """
    Behaviour tree node for navigating the robot through waypoints.

    Uses a simple proportional controller to steer the robot towards
    each waypoint in sequence. The waypoints are read from the blackboard,
    typically set by the Planning behaviour.

    Attributes:
        blackboard: Shared data storage for communication between behaviours
        robot: Reference to the Webots robot/supervisor instance
        waypoints: List of (x, y) waypoints to navigate through
        index: Current waypoint index
    """

    def __init__(self, name: str, blackboard: Blackboard):
        """
        Initialize the Navigation behaviour.

        Args:
            name: Name of this behaviour node
            blackboard: Shared blackboard for inter-behaviour communication
        """
        super(Navigation, self).__init__(name)
        self.blackboard = blackboard
        self.robot = blackboard.read('robot')

    def setup(self):
        """
        Set up sensors and actuators needed for navigation.

        Initializes GPS for position, compass for heading, and motor
        controllers for differential drive.
        """
        self.timestep = int(self.robot.getBasicTimeStep())

        # GPS for position feedback
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

        # Compass for heading feedback
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)

        # Differential drive motors
        self.left_motor = self.robot.getDevice('wheel_left_joint')
        self.right_motor = self.robot.getDevice('wheel_right_joint')

        # Set motors to velocity control mode (infinite position)
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        # Visual marker for current target waypoint (Supervisor feature)
        self.marker = self.robot.getFromDef("marker").getField("translation")

        self.logger.debug("Navigation setup complete")

    def initialise(self):
        """
        Initialize for a new navigation run.

        Stops motors and resets waypoint index. Reads waypoints from blackboard.
        """
        # Stop motors initially
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Start at first waypoint
        self.index = 0

        # Get waypoints from blackboard (set by Planning behaviour)
        self.waypoints = self.blackboard.read('waypoints')

        self.logger.debug(f"Navigation initialised with {len(self.waypoints)} waypoints")

    def update(self):
        """
        Execute one control step of waypoint following.

        Uses proportional control to:
        1. Calculate distance (rho) and angle (alpha) to current waypoint
        2. Adjust wheel speeds to steer towards the waypoint
        3. Advance to next waypoint when close enough

        Returns:
            SUCCESS when all waypoints reached, RUNNING otherwise
        """
        self.logger.debug("Navigation update")

        # Get current robot position from GPS
        x_world = self.gps.getValues()[0]
        y_world = self.gps.getValues()[1]

        # Get robot heading from compass
        theta = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])

        # Calculate distance to current waypoint
        dx = self.waypoints[self.index][0] - x_world
        dy = self.waypoints[self.index][1] - y_world
        rho = np.sqrt(dx**2 + dy**2)

        # Calculate angle to waypoint relative to robot heading
        alpha = np.arctan2(dy, dx) - theta

        # Normalize angle to [-pi, pi]
        if alpha > np.pi:
            alpha = alpha - 2 * np.pi
        elif alpha < -np.pi:
            alpha = alpha + 2 * np.pi

        # Update visual marker to show current target
        self.marker.setSFVec3f([*self.waypoints[self.index], 0])

        # Proportional control gains
        p_rho = 4    # Forward speed gain based on distance
        p_alpha = 2  # Steering gain based on angle error

        # Base forward speed proportional to distance (capped)
        forward_speed = min(p_rho * rho, 6.28)

        # Differential steering based on angle error
        steering = p_alpha * alpha

        # Calculate individual wheel speeds
        speed_left = forward_speed - steering
        speed_right = forward_speed + steering

        # Clamp wheel speeds to motor limits (-6.28 to 6.28 rad/s)
        max_speed = 6.28
        speed_left = max(-max_speed, min(max_speed, speed_left))
        speed_right = max(-max_speed, min(max_speed, speed_right))

        # Apply velocities to motors
        self.left_motor.setVelocity(speed_left)
        self.right_motor.setVelocity(speed_right)

        # Check if waypoint reached (within threshold distance)
        waypoint_threshold = 0.4  # meters
        if rho < waypoint_threshold:
            print(f"Navigation: Reached waypoint {self.index} of {len(self.waypoints)}")
            self.index = self.index + 1

            # Check if all waypoints completed
            if self.index == len(self.waypoints):
                self.feedback_message = "All waypoints reached"
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.RUNNING
        else:
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Clean up when navigation ends.

        Stops the motors to ensure robot doesn't continue moving.

        Args:
            new_status: The status the behaviour is transitioning to
        """
        print("Navigation: Terminating navigation")

        # Stop motors
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

"""
Mapping module for building an occupancy grid from LIDAR sensor data.

This module provides the Mapping behaviour class that uses LIDAR to scan
the environment and builds a configuration space (cspace) for path planning.
"""

import numpy as np
import py_trees
from py_trees.blackboard import Blackboard
from scipy import signal
import matplotlib.pyplot as plt


def world2map(x_world, y_world):
    """
    Convert world coordinates to map/pixel coordinates.

    The world coordinate system maps to a 300x300 pixel grid:
    - World (-2.15, 1.66) -> Map (0, 0)     (top-left)
    - World (0.305, 0.25) -> Map (149, 149) (center)
    - World (2.15, -3.92) -> Map (299, 299) (bottom-right)

    Args:
        x_world: X coordinate in world frame
        y_world: Y coordinate in world frame

    Returns:
        List [x_map, y_map] with clamped pixel coordinates (0-299)
    """
    x_map = int((x_world + 2.15) / 4.3 * 299)
    y_map = int(-(y_world - 1.66) / 5.58 * 299)

    # Clamp x coordinate to valid map range
    if x_map < 0:
        x_map = 0
    elif x_map > 299:
        x_map = 299

    # Clamp y coordinate to valid map range
    if y_map < 0:
        y_map = 0
    elif y_map > 299:
        y_map = 299

    return [x_map, y_map]


class Mapping(py_trees.behaviour.Behaviour):
    """
    Behaviour tree node for mapping the environment using LIDAR.

    This behaviour drives the robot around the environment while using
    LIDAR to build an occupancy grid map. The map is then convolved with
    a kernel to create a configuration space (cspace) that accounts for
    the robot's size.

    Attributes:
        blackboard: Shared data storage for communication between behaviours
        robot: Reference to the Webots robot/supervisor instance
        has_run: Flag to track if mapping has executed at least once
    """

    def __init__(self, name: str, blackboard: Blackboard):
        """
        Initialize the Mapping behaviour.

        Args:
            name: Name of this behaviour node
            blackboard: Shared blackboard for inter-behaviour communication
        """
        super(Mapping, self).__init__(name)

        self.blackboard = blackboard
        self.has_run = False
        self.robot = blackboard.read('robot')

    def setup(self):
        """
        Set up all sensors required for mapping.

        Initializes GPS for localization, compass for orientation,
        LIDAR for obstacle detection, and display for visualization.
        """
        self.timestep = int(self.robot.getBasicTimeStep())

        # GPS for robot position
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)

        # Compass for robot orientation
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)

        # LIDAR for obstacle detection
        self.lidar = self.robot.getDevice('Hokuyo URG-04LX-UG01')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # Display for real-time map visualization
        self.display = self.robot.getDevice('display')

    def initialise(self):
        """
        Initialize the mapping process.

        Creates empty occupancy map and precomputes LIDAR beam angles.
        The angles are trimmed to remove edge beams that may have noise.
        """
        # Initialize empty occupancy grid (200x300 to match world dimensions)
        self.map = np.zeros((200, 300))

        # Precompute LIDAR beam angles (trimming noisy edge beams)
        # Hokuyo URG-04LX-UG01 has 667 readings over ~4.19 radian FOV
        self.angles = np.linspace(4.19 / 2, -4.19 / 2, 667)
        self.angles = self.angles[80:len(self.angles) - 80]  # Trim to 507 readings

    def update(self):
        """
        Process one timestep of mapping.

        Gets robot pose, reads LIDAR data, transforms points to world frame,
        and updates occupancy grid. Displays progress on robot's screen.

        Returns:
            RUNNING status (mapping continues until externally terminated)
        """
        self.has_run = True

        # Get current robot position from GPS
        x_world = self.gps.getValues()[0]
        y_world = self.gps.getValues()[1]

        # Get robot heading from compass (atan2 gives angle in radians)
        theta = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])

        # Draw robot position on display in red
        px, py = world2map(x_world, y_world)
        self.display.setColor(0xFF0000)
        self.display.drawPixel(px, py)

        # Build homogeneous transformation matrix from robot frame to world frame
        # This allows us to transform LIDAR points from robot-relative to world coordinates
        w_T_r = np.array([[np.cos(theta), -np.sin(theta), x_world],
                         [np.sin(theta), np.cos(theta), y_world],
                         [0, 0, 1]])

        # Get LIDAR range readings and trim edge beams
        ranges = np.array(self.lidar.getRangeImage())
        ranges = ranges[80:len(ranges) - 80]

        # Replace infinite readings (no obstacle) with large value
        ranges[ranges == np.inf] = 100

        # Convert polar LIDAR readings to Cartesian coordinates in robot frame
        # 0.202 is the LIDAR offset from robot center
        X_i = np.array([ranges * np.cos(self.angles) + 0.202,
                        ranges * np.sin(self.angles),
                        np.ones(len(ranges))])

        # Transform LIDAR points from robot frame to world frame
        D = w_T_r @ X_i

        # Update occupancy grid for each detected point
        for d in D.T:
            px, py = world2map(d[0], d[1])

            # Increment occupancy probability (capped at 1.0)
            self.map[px, py] += 0.01
            if self.map[px, py] > 1:
                self.map[px, py] = 1

            # Draw point on display (grayscale based on occupancy)
            v = int(self.map[px, py] * 255)
            colour = (v * 256**2 + v * 256 + v)
            self.display.setColor(int(colour))
            self.display.drawPixel(px, py)

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Clean up and save the configuration space when mapping ends.

        Convolves the occupancy grid with a kernel to create a configuration
        space that accounts for robot size. Displays the result and saves
        to 'cspace.npy' for use by the Planning behaviour.

        Args:
            new_status: The status the behaviour is transitioning to
        """
        if self.has_run:
            # Convolve with 26x26 kernel to expand obstacles by robot radius
            # This creates configuration space where a point robot can navigate
            cspace = signal.convolve2d(self.map, np.ones((26, 26)), mode='same')

            # Display the raw configuration space
            plt.figure(0)
            plt.imshow(cspace)
            plt.title('Configuration Space (raw)')
            plt.show()

            # Display thresholded configuration space (obstacles in white)
            plt.figure(1)
            plt.imshow(cspace > 0.9)
            plt.title('Configuration Space (thresholded)')
            plt.show()

            # Save configuration space for path planning
            np.save('cspace', cspace)
            print("Mapping: Configuration space saved to 'cspace.npy'")
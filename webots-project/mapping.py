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

    The world coordinate system maps to a 200x300 pixel grid:
    - World (-2.15, 1.66) -> Map (0, 0)     (top-left)
    - World (2.15, -3.92) -> Map (199, 299) (bottom-right)

    Args:
        x_world: X coordinate in world frame
        y_world: Y coordinate in world frame

    Returns:
        List [x_map, y_map] with clamped pixel coordinates
    """
    # Map dimensions: 200 (x) x 300 (y)
    x_map = int((x_world + 2.15) / 4.3 * 199)
    y_map = int(-(y_world - 1.66) / 5.58 * 299)

    # Clamp x coordinate to valid map range (0-199)
    if x_map < 0:
        x_map = 0
    elif x_map > 199:
        x_map = 199

    # Clamp y coordinate to valid map range (0-299)
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

        # Get LIDAR properties for angle computation
        self.lidar_resolution = self.lidar.getHorizontalResolution()
        self.lidar_fov = self.lidar.getFov()

        # Display for real-time map visualization
        self.display = self.robot.getDevice('display')

    def initialise(self):
        """
        Initialize the mapping process.

        Creates empty occupancy map and precomputes LIDAR beam angles.
        """
        # Reset run flag for this new mapping session
        self.has_run = False

        # Initialize empty occupancy grid (200x300 to match world dimensions)
        self.map = np.zeros((200, 300))

        # Precompute LIDAR beam angles using actual device properties
        self.angles = np.linspace(
            self.lidar_fov / 2,
            -self.lidar_fov / 2,
            self.lidar_resolution
        )

        # Re-enable LIDAR in case it was disabled
        self.lidar.enable(self.timestep)

        self.logger.debug("Mapping initialised")

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
        # Note: GPS/LIDAR are co-located at the front of the robot
        x_gps = self.gps.getValues()[0]
        y_gps = self.gps.getValues()[1]

        # Get robot heading from compass (atan2 gives angle in radians)
        theta = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])

        # Draw robot position on display in red
        px, py = world2map(x_gps, y_gps)
        self.display.setColor(0xFF0000)
        self.display.drawPixel(px, py)

        # Get LIDAR range readings
        ranges = np.array(self.lidar.getRangeImage())

        # Ignore noisy edge beams by setting them to infinity
        ranges[:80] = np.inf
        ranges[-80:] = np.inf

        # Replace infinite readings (no obstacle) with large value so they don't show on map
        ranges[ranges == np.inf] = 100

        # LIDAR is at GPS position (front of robot), so use GPS coords directly
        x_lidar, y_lidar = x_gps, y_gps

        # Build homogeneous transformation matrix from LIDAR frame to world frame
        lidar_to_world = np.array([
            [np.cos(theta), -np.sin(theta), x_lidar],
            [np.sin(theta), np.cos(theta), y_lidar],
            [0, 0, 1]
        ])

        # Convert polar LIDAR readings to Cartesian coordinates in LIDAR frame
        # No offset needed since we're using LIDAR position directly
        X_i = np.array([
            ranges * np.cos(self.angles),
            ranges * np.sin(self.angles),
            np.ones(len(self.angles))
        ])

        # Transform LIDAR points from LIDAR frame to world frame
        D = lidar_to_world @ X_i

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
            # Convolve with kernel to expand obstacles by robot radius
            # This creates configuration space where a point robot can navigate
            cspace = signal.convolve2d(self.map, np.ones((26, 26)), mode='same')

            # Save configuration space for path planning
            np.save('cspace', cspace)
            print("Mapping: Configuration space saved to 'cspace.npy'")

            # Display the configuration space (non-blocking)
            plt.figure(0)
            plt.imshow(cspace)
            plt.title('Configuration Space (raw)')
            plt.savefig('cspace_raw.png')  # Save to file instead of blocking

            plt.figure(1)
            plt.imshow(cspace > 0.9)
            plt.title('Configuration Space (thresholded)')
            plt.savefig('cspace_thresholded.png')  # Save to file instead of blocking

            # Close figures to free memory
            plt.close('all')

            # Reset flag so we don't save again if terminate is called multiple times
            self.has_run = False
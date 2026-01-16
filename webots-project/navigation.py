import py_trees
import numpy as np


class Navigation():
    def __init__():
        print("Nav")


    def setup(self):
        self.timestep = int(self.robot.getBasicTimeStep())
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)

        self.left_motor = self.robot.getDevice('wheel_left_joint')
        self.right_motor = self.robot.getDevice('wheel_right_joint')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.marker = self.robot.getFromDef("marker").getField("translation")

        self.logger.debug("Navigation set up")


    def initialise(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.index = 0

        self.logger.debug("Navigation initialised")
        self.waypoints = self.blackboard.read('waypoints')


    def update(self):
        self.debugger("Navigation update")

        x_world = self.gps.getValues()[0]
        y_world = self.gps.getValues()[1]
        theta = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])

        rho = np.sqrt((x_world - self.waypoints[self.index][0])**2 + (y_world - self.waypoints[self.index][1])**2)
        alpha = np.arctan2(self.waypoints[self.index][1] - y_world, self.waypoints[self.index][0] - x_world) - theta

        if alpha > np.pi:
            alpha = alpha - 2*np.pi
        elif alpha < -np.pi:
            alpha = alpha + 2*np.pi

        self.marker.setSFVec3f([*self.waypoints[self.index], 0])

        speed_left, speed_right = 6.28, 6.28

        p1 = 4
        p2 = 2

        speed_left = min(speed_left, 6.28)
        speed_left = max(speed_left, -6.28)
        speed_right = min(speed_right, 6.28)
        speed_right = max(speed_right, -6.28)

        self.left_motor.setVelocity(speed_left)
        self.right_motor.setVelocity(speed_right)

        if rho < 0.4:
            print("Reached ", self.index, len(self.waypoints))
            self.index = self.index + 1
            if self.index == len(self.waypoints):
                self.feedback_message = "Last waypoint reached"
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.RUNNING
        else:
            return py_trees.common.Status.RUNNING


    def terminate(self, new_status):
        print("Terminating navigation")


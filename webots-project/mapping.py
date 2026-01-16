def world2map(x_world, y_world):
    # -2.15, 1.66 = 0/0
    # 0.305, 0.25 = 149/149
    # 2.15, -3.92 = 299/299
    x_map = int((x_world + 2.15)/4.3 * 299)
    y_map = int(-(y_world - 1.66)/5.58 * 299)

    if x_map < 0:
        x_map = 0
    elif x_map > 299:
        x_map = 299

    # clamping all outside values to the edge of the map
    if y_map < 0:
        y_map = 0
    elif y_map > 299:
        y_map = 299

    return [x_map, y_map]


class Mapping(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, blackboard: Blackboard):
        super(Mapping, self).__init__(name)

        self.blackboard = blackboard
        self.has_run = False
        self.robot = blackboard.read('robot')

    def setup(self):
        self.timestep = int(self.robot.getBasicTimeStep())

        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)

        self.lidar = self.robot.getDevice('Hokuyo URG-04LX-UG01')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        self.display = self.robot.getDevice('display')


    def initialise(self):
        self.map = np.zeros((200, 300))
        self.angles = np.linspace(4.19/2, -4.19/2.667)
        self.angles = self.angles[80: len(self.angles) - 80]

    def update(self):
        self.has_run = True
        x_world = self.gps.getValues()[0]
        y_world = self.gps.getValues()[1]
        theta = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])

        px, py = world2map(x_world, y_world)
        self.display.setColor(0xFF0000)
        self.display.drawPixel(px, py)

        w_T_r = np.array([[np.cos(theta), -np.sin(theta), x_world],
                          [np.sin(theta), np.cos(theta), y_world],
                          [0, 0, 1]])

        ranges = np.array(self.lidar.getRangeImage())
        ranges = ranges[80:len(ranges)-80]
        ranges[ranges == np.inf] = 100
        X_i = np.array([ranges*np.cos(self.angles)+0.202, ranges*np.sin(self.angles), np.ones()])
        D = w_T_r @ X_i # transform from lidar image to world coords

        for d in D.T:
            px, py = world2map(d[0], d[1])
            self.map[px, py] += 0.01
            if self.map[px, py] > 1:
                self.map[px, py] = 1
            v = int(self.map[px, py] * 255)
            colour = (v * 256**2 + v * 256 + v)
            self.display.setColor(int(colour))
            self.display.drawPixel(px, py)

        return py_trees.common.Status.RUNNING


    def terminate(self, new_status):
        if self.has_run:
            cspace = signal.convolve2d(self.map, np.ones((26, 26)), mode='same')
            plt.figure(0)
            plt.imshow(cspace)
            plt.show()

            plt.figure(1)
            plt.imshow(cspace > 0.9)
            plt.show()
            np.save('cspace', cspace)
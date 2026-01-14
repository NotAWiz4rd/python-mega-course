class Node():
    def __init__(self, children: list = None, name: str = None):
        self.children = children or []
        self.name = name or self.__class__.__name__


class Sequence(Node):
    def execute(self) -> bool:
        print("Executing sequence:", self.name)
        for child in self.children:
            if not child.execute():
                return False
        return True


class Selector(Node):
    def execute(self) -> bool:
        print("Executing selector:", self.name)
        for child in self.children:
            if child.execute():
                return True
        return False


class Action(Node):
    def execute(self) -> bool:
        print("Executing action:", self.name)
        # Implement your action logic here
        # Return True if the action succeeded, False otherwise
        return True


class SenseIfPegAlreadyInGripper(Action):
    def execute(self) -> bool:
        print("Gripper not in hand!")
        return False


class LocatePeg(Action):
    pass


class OpenGripper(Action):
    pass


class MoveGripperToPeg(Action):
    pass


class CloseGripperAroundPeg(Action):
    pass


def main():
    # Example usage
    # Create the behavior tree
    tree = Selector([SenseIfPegAlreadyInGripper(),
                     Sequence([LocatePeg(), OpenGripper(),
                               MoveGripperToPeg(), CloseGripperAroundPeg()
                               ], "Get Peg")
                     ], "Pick square peg")

    # Execute the behavior tree
    print(tree.execute())


if __name__ == "__main__":
    main()

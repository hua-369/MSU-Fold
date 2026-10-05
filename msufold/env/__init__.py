class Action:
    """一次 pick-and-place 动作（像素坐标）。

    单臂步：1 个 pick + 1 个 place；双臂步：2 个 pick + 2 个 place。
    两者共用同一个共享热力图头，只是取峰个数不同。
    """

    def __init__(self, pick, place):
        self.pick = pick
        self.place = place

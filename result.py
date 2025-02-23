class Result:

    def __init__(self, points, count, x0, y0, min_y=None, success=False):
        self.points = points
        self.count = count
        self.x0 = x0
        self.y0 = y0
        self.min_y = min_y
        self.success = success

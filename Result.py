class Result:

    def __init__(self, points, x0, y0, bounds, count, diff, error, f, P):
        self.points = points
        self.bounds = bounds
        self.x0 = x0
        self.y0 = y0
        self.count = count
        self.diff = diff
        self.error = error
        self.P = P
        self.f = f

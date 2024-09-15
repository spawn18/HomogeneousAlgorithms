import math

class f1:
    a = 0.2
    b = 7

    @staticmethod
    def eval(x):
        return x+math.fabs(math.sin(x)*math.cos(x))

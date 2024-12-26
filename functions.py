import math

class f1:
    a = -1.5
    b = 11

    min_x = 10.0
    min_y = -29763.233333333344

    @staticmethod
    def eval(x):
        return x**6/6-52*x**5/25+39*x**4/80+71*x**3/10-79*x**2/20-x+0.1

class f2:
    a = 2.7
    b = 7.5

    min_x = 4.86490896
    min_y = -1.3213782438886077

    @staticmethod
    def eval(x):
        return math.sin(x)+math.sin(10*x)/3

class f3:
    a = -10
    b = 10

    min_x = -6.999998
    min_y = -13.666303261150684

    @staticmethod
    def eval(x):
        return -sum([k*math.sin(int((k+1)*x+k)) for k in range(1, 6)])

class f4:
    a = 1.9
    b = 3.9

    min_x = 2.8680339999999998
    min_y = -3.8504507088002193

    @staticmethod
    def eval(x):
        return -(16*x**2-24*x+5)*math.exp(-x)

class f5:
    a = 0
    b = 1.2

    min_x = 0.9660858
    min_y = -1.4890725386896007

    @staticmethod
    def eval(x):
        return (3*x-1.4)*math.sin(18*x)

class f6:
    a = -10
    b = 10

    min_x = 0.6795780000000011
    min_y = -0.8242393984752998

    @staticmethod
    def eval(x):
        return -(x+math.sin(x))*math.exp(-x**2)

class f7:
    a = 2.7
    b = 7.5

    min_x = 5.49634608
    min_y = -0.9542709402765928

    @staticmethod
    def eval(x):
        return math.sin(x)+math.sin(10*x)/3 + math.log(x) - 0.84*x + 3

class f8:
    a = -10
    b = 10

    min_x = -0.9999979999999997
    min_y = -15.0

    @staticmethod
    def eval(x):
        return -sum([k*math.cos(int((k+1)*x+k)) for k in range(1, 6)])

class f9:
    a = 3.1
    b = 20.4

    min_x = 5.15178173
    min_y = -1.1616649783297421

    @staticmethod
    def eval(x):
        return math.sin(x) + math.sin(2*x)/3

class f10:
    a = 0
    b = 10

    min_x = 7.978666
    min_y = -7.916727371587444

    @staticmethod
    def eval(x):
        return -x*math.sin(x)

class f11:
    a = -1.57
    b = 6.28

    min_x = 2.0943949150000005
    min_y = -1.4999999999999474

    @staticmethod
    def eval(x):
        return 2*math.cos(x)+math.cos(2*x)

class f12:
    a = 0
    b = 6.28

    min_x = 3.141592608
    min_y = -0.999999999999997

    @staticmethod
    def eval(x):
        return (math.sin(x))**3+(math.cos(x))**3

class f13:
    a = 0.001
    b = 0.99

    min_x = 0.7071067367
    min_y = -1.5874010519681965

    @staticmethod
    def eval(x):
        return -math.cbrt(x**2) + math.cbrt(x**2-1)

class f14:
    a = 0
    b = 4

    min_x = 0.2248804
    min_y = -0.7886853874086694

    @staticmethod
    def eval(x):
        return -math.exp(-x)*math.sin(2*math.pi*x)

class f15:
    a = -5
    b = 5

    min_x = 2.4142140000000003
    min_y = -0.03553390593270871

    @staticmethod
    def eval(x):
        return (x**2-5*x+6)/(x**2+1)

class f16:
    a = -3
    b = 3

    min_x = 1.5907169999999997
    min_y = 7.515924153082398

    @staticmethod
    def eval(x):
        return 2*(x-3)**2+math.exp(0.5*x**2)

class f17:
    a = -4
    b = 4

    min_x = -3.0
    min_y = 7.0

    @staticmethod
    def eval(x):
        return x**6-15*x**4+27*x**2+250

class f18:
    a = 0
    b = 6

    min_x = 2
    min_y = 0

    @staticmethod
    def eval(x):
        if x <= 3:
            return (x-2)**2
        else:
            return 2*math.log(x-2)+1

class f19:
    a = 0
    b = 0.65

    min_x = 0
    min_y = -1.0

    @staticmethod
    def eval(x):
        return -x+math.sin(3*x)-1

class f20:
    a = -10
    b = 10

    min_x = 1.1951360000000015
    min_y = -0.06349052893638496

    @staticmethod
    def eval(x):
        return (math.sin(x)-x)*math.exp(-x**2)


funcs = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20]
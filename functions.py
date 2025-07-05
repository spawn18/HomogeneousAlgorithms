import math
import numpy as np



class f1:
    bounds = [-1.5, 11.0]
    min_x = [10.0]
    min_y = -29763.233333333344

    @staticmethod
    def eval(x):
        return x**6/6-52*x**5/25+39*x**4/80+71*x**3/10-79*x**2/20-x+0.1

class f2:
    bounds = [2.7, 7.5]
    min_x = [5.14573536]
    min_y = -1.8995993491520842

    @staticmethod
    def eval(x):
        return np.sin(x)+np.sin(10*x/3)

class f3:
    bounds = [-10.0, 10.0]
    min_x = [-6.77458, -0.49139, 5.79179]
    min_y = -12.03124944216395

    @staticmethod
    def eval(x):
        return -sum([k*np.sin((k+1)*x+k) for k in range(1, 6)])

class f4:
    bounds = [1.9, 3.9]
    min_x = [2.8680339999999998]
    min_y = -3.8504507088002193

    @staticmethod
    def eval(x):
        return -(16*x**2-24*x+5)*np.exp(-x)

class f5:
    bounds = [0.0, 1.2]
    min_x = [0.9660858]
    min_y = -1.4890725386896007

    @staticmethod
    def eval(x):
        return (3*x-1.4)*np.sin(18*x)

class f6:
    bounds = [-10.0, 10.0]
    min_x = [0.6795780000000011]
    min_y = -0.8242393984752998

    @staticmethod
    def eval(x):
        return -(x+np.sin(x))*np.exp(-x**2)

class f7:
    bounds = [2.7, 7.5]
    min_x = [5.19977856]
    min_y = -1.6013075464941817

    @staticmethod
    def eval(x):
        return np.sin(x)+np.sin(10*x/3) + np.log(x) - 0.84*x + 3

class f8:
    bounds = [-10.0, 10.0]
    min_x = [-7.08351, -0.80032, 5.48286]
    min_y = -14.508007927187695

    @staticmethod
    def eval(x):
        return -sum([k*np.cos((k+1)*x+k) for k in range(1, 6)])

class f9:
    bounds = [3.1, 20.4]
    min_x = [17.039198199999998]
    min_y = -1.9059611187153978

    @staticmethod
    def eval(x):
        return np.sin(x) + np.sin(2*x/3)

class f10:
    bounds = [0.0, 10.0]
    min_x = [7.978666]
    min_y = -7.916727371587444

    @staticmethod
    def eval(x):
        return -x*np.sin(x)

class f11:
    bounds = [-1.57, 6.28]
    min_x = [2.0943949150000005]
    min_y = -1.5

    @staticmethod
    def eval(x):
        return 2*np.cos(x)+np.cos(2*x)

class f12:
    bounds = [0.0, 6.28]
    min_x = [math.pi, 3*math.pi/2]
    min_y = -1

    @staticmethod
    def eval(x):
        return np.sin(x)**3+np.cos(x)**3

class f13:
    bounds = [0.001, 0.99]
    min_x = [0.7071067367]
    min_y = -1.5874010519681965

    @staticmethod
    def eval(x):
        return -np.cbrt(x**2) + np.cbrt(x**2-1)

class f14:
    bounds = [0.0, 4.0]
    min_x = [0.2248804]
    min_y = -0.7886853874086694

    @staticmethod
    def eval(x):
        return -np.exp(-x)*np.sin(2*math.pi*x)

class f15:
    bounds = [-5.0, 5.0]
    min_x = [2.4142140000000003]
    min_y = -0.03553390593270871

    @staticmethod
    def eval(x):
        return (x**2-5*x+6)/(x**2+1)

class f16:
    bounds = [-3.0, 3.0]
    min_x = [1.5907169999999997]
    min_y = 7.515924153082398

    @staticmethod
    def eval(x):
        return 2*(x-3)**2+np.exp(0.5*x**2)

class f17:
    bounds = [-4.0, 4.0]
    min_x = [-3.0, 3.0]
    min_y = 7.0

    @staticmethod
    def eval(x):
        return x**6-15*x**4+27*x**2+250

class f18:
    bounds = [0.0, 6.0]
    min_x = [2.0]
    min_y = 0.0

    @staticmethod
    def eval(x):
        return np.piecewise(x, [x <= 3, x > 3], [lambda x: np.power((x-2), 2), lambda x: 2*np.log(x-2)+1])

class f19:
    bounds = [0.0, 0.65]
    min_x = [0.0]
    min_y = -1.0

    @staticmethod
    def eval(x):
        return -x+np.sin(3*x)-1

class f20:
    bounds = [-10.0, 10.0]
    min_x = [1.1951360000000015]
    min_y = -0.06349052893638496

    @staticmethod
    def eval(x):
        return (np.sin(x)-x)*np.exp(-x**2)


funcs = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20]


class f1s:
    bounds = [0.0, math.pi]
    min_x = [2.933735]
    min_y = -1.937583

    @staticmethod
    def eval(x):
        return sum([np.fabs(np.cos(k*x))*np.fabs(np.sin(k*x))*np.cos(x) for k in range(1,6)])

class f2s:
    bounds = [1, 5]
    min_x = [3*math.pi/2]
    min_y = 0.097755+550

    @staticmethod
    def eval(x):
        return (1/x)*np.fabs((1/np.sqrt(x))*np.cos(x))+(1/np.sqrt(x))*np.fabs((1/x)*np.sin(x))+550

class f3s:
    bounds = [-2.0+1372.0, 2.0+1372.0]
    min_x = [math.pi/2+1372]
    min_y = 0.0713225-18888.0

    @staticmethod
    def eval(x):
        return 1/(-np.log(np.fabs(np.cos(x-1372)) + 0.0001) + np.exp(x-1372))-18888

class f4s:
    bounds = [1.0, 10.0]
    min_x = [2*math.pi]
    min_y = -0.322712-14522

    @staticmethod
    def eval(x):
        return sum([1/np.sqrt(k)*np.log((20/k)*np.fabs(np.cos(k*x)*np.sin(x/k))+np.exp(np.fabs(np.sin(k*x))))*np.sin(np.sin(k)) for k in range(1,21)])-14522

class f5s:
    bounds = [50.0, 53.0]
    min_x = [51.86518]
    min_y = 0.065892

    @staticmethod
    def eval(x):
        return math.prod([np.exp(-np.fabs(np.sin(2**np.fabs(np.log(n))*x))) for n in range(1, 4)])

sfuncs = [f1s, f2s, f3s, f4s, f5s]

class f5_od1:
    bounds = [0.0, 1.2]
    min_x = [0.9660858]
    min_y = -1.4890725386896007+5

    @staticmethod
    def eval(x):
        return (3*x-1.4)*np.sin(18*x)+5

class f5_od2:
    bounds = [0.0, 1.2]
    min_x = [0.9660858]
    min_y = -1.4890725386896007-314

    @staticmethod
    def eval(x):
        return (3*x-1.4)*np.sin(18*x)-314

class f5_od3:
    bounds = [0.0, 1.2]
    min_x = [0.9660858]
    min_y = -1.4890725386896007+0.0003

    @staticmethod
    def eval(x):
        return (3*x-1.4)*np.sin(18*x)+0.0003

fod = [f5_od1, f5_od2, f5_od3]
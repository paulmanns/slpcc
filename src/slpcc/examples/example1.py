import numpy as np


class Example1:
    def __init__(self):
        self.l = np.array([0., 0.])
        self.u = np.array([np.Inf, np.Inf])
        self.x_init = np.array([2., 0.])
        self.n0 = 0
        self.n12 = 1
        self.opt_x = np.array([0., 1.])
        self.opt_f = self.fun(self.opt_x)

    def fun(self, x):
        return x[0]**3 - (x[1] - 0.5*x[1]**2)

    def grad(self, x):
        return np.array([3*x[0]**2, x[1] - 1.])

    def hess(self, x):
        return np.array([[6.*x[0], 0.], [0., 1.]])

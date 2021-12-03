import numpy as np


class Example6:
    def __init__(self):
        self.l = np.array([np.NINF, 2.])
        self.u = np.array([np.Inf,  3.])
        self.x_init = np.array([-0.3, 3.])
        self.n0 = 0
        self.n12 = 1
        self.opt_x = np.array([0., 2.])
        self.opt_f = self.fun(self.opt_x)

    def fun(self, x):
        return np.exp(-x[0])*(x[1] - 2.) + 0.0005*x[0]**2

    def grad(self, x):
        return np.array([-np.exp(-x[0])*(x[1] - 2.) + 0.001*x[0], np.exp(-x[0])])

    def hess(self, x):
        return np.array([
            [np.exp(-x[0])*(x[1] - 2.) + 0.001, -np.exp(-x[0])],
            [-np.exp(-x[0]), 0.]])

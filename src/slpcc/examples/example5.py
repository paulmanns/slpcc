import numpy as np


class Example5:
    def __init__(self):
        self.l = np.array([-2., np.NINF])
        self.u = np.array([ 1., np.Inf])
        self.x_init = np.array([-2., 5.])
        self.n0 = 0
        self.n12 = 1
        self.opt_x = np.array([-1., 0.])
        self.opt_f = self.fun(self.opt_x)

    def fun(self, x):
        return (x[0] + 0.5*x[0]**2) + 4*x[1]**4 + x[1]**2 + 0.01*x[1]**3

    def grad(self, x):
        return np.array([x[0] + 1., 12.*x[1]**3 + 2*x[1] + 0.03*x[1]**2])

    def hess(self, x):
        return np.array([[1., 0.], [0., 36.*x[0]**2 + 2. + 0.06*x[1]]])

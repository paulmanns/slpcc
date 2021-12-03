import numpy as np


class Jr2AL:
    def __init__(self):
        self.rho = 100.
        self.lam = 1.
        self.n0 = 1
        self.n12 = 1
        #                   x[0]    x[1]    x[2]
        #                     z1      z2      s1
        self.l = np.array([-500., 0., 0.])
        self.u = np.array([500., np.Inf, np.Inf])
        self.x_init = np.array([0., 0., 0.])
        self.opt_x = np.array([0.5, 0.5, 0.])
        self.opt_f = self.fun(self.opt_x)
        self.opt_g = np.array([0.,  0.,  1.])

    def fun(self, x):
        return (x[1] - 1.)**2 + x[0]**2 - self.lam*(x[2] - x[1] + x[0]) + .5*self.rho*(x[2] - x[1] + x[0])**2

    def grad(self, x):
        return np.array([
            2. * x[0] - self.lam + self.rho * (x[2] - x[1] + x[0]),
            2. * (x[1] - 1.) + self.lam - self.rho * (x[2] - x[1] + x[0]),
            -self.lam + self.rho * (x[2] - x[1] + x[0])
        ])

    def hess(self, x):
        return np.array([
            #         d/dx0,          d/dx1,          d/dx2
            [ 2. + self.rho,      -self.rho,       self.rho],
            [     -self.rho,  2. + self.rho,      -self.rho],
            [      self.rho,      -self.rho,       self.rho]
        ])

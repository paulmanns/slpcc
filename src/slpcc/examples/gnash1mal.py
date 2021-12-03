import numpy as np


class Gnash1MAL:
    def __init__(self):
        self.rho = 2.
        self.LamF1, self.LamF2, self.LamL1, self.LamL2 = 3.9375, -6.5, -0.25, 2.5
        self.n0, self.n12 = 4, 2
        #                     z[0]    z[1]    z[2]    z[3]    z[4]    z[5]    z[6]    z[7]
        #                     x[1]    x[2]    y[1]    y[2]    s[1]    s[2]    l[1]    l[2]
        self.l = np.array([     0.,     0.,  -500.,  -500.,     0.,     0.,     0.,     0.])
        self.u = np.array([    10.,    10.,   500.,   500., np.Inf, np.Inf, np.Inf, np.Inf])
        self.x_init = np.zeros(self.n0 + 2 * self.n12)
        self.opt_x = np.array([10., 0.56250006, 12.8749998, -2.31250023, 0., 0., 12.44791777, 9.53125077])
        self.opt_f = self.fun(self.opt_x)

        # not global minimizer:
        # xb_ref = np.array([10., 4.56526577, 10.13274313, 4.21128336, 0., 7.96128336, 0.53567482, 0.]);
        # has better objective value but this is what it should converge to using the current code

    def fun(self, z):
        return .5*((z[0] - z[2])**2 + (z[1] - z[3])**2) \
            + self.LamF1*(-34.   +   2.*z[2] + (8/3.)*z[3] + z[6]) \
            - self.LamF2*(-24.25 + 1.25*z[2] +     2.*z[3] + z[7]) \
            - self.LamL1*(z[4] + z[1] + z[2] - 15.) \
            + self.LamL2*(z[5] + z[0] - z[3] - 15.)  \
            + .5*self.rho*((-34.     +   2.*z[2] + (8/3.)*z[3] + z[6])**2
                           + (-24.25 + 1.25*z[2] +     2.*z[3] + z[7])**2
                           + (z[4] + z[1] + z[2] - 15.)**2
                           + (z[5] + z[0] - z[3] - 15.)**2)

    def grad(self, z):
        return np.array([
            z[0] - z[2] + self.LamL2 + self.rho*(z[5] + z[0] - z[3] - 15.),                # d/dz[0]
            z[1] - z[3] - self.LamL1 + self.rho*(z[4] + z[1] + z[2] - 15.),                # d/dz[1]
            -(z[0] - z[2]) + self.LamF1*2. - self.LamF2*1.25 - self.LamL1 + self.rho*(     # d/dz[2]
                (-34. + 2.*z[2] + (8/3)*z[3] + z[6])*2.                                    # ...
                + (-24.25 + 1.25*z[2] + 2.*z[3] + z[7])*1.25                               # ...
                + (z[4] + z[1] + z[2] - 15.)),                                             # ...
            -(z[1] - z[3]) + self.LamF1*(8/3.) - self.LamF2*2. - self.LamL2 + self.rho*(   # d/dz[3]
                (-34. + 2. * z[2] + (8/3.)*z[3] + z[6])*(8/3.)                             # ...
                + (-24.25 + 1.25*z[2] + 2.*z[3] + z[7])*2.                                 # ...
                - (z[5] + z[0] - z[3] - 15.)),                                             # ...
            - self.LamL1 + self.rho*(z[4] + z[1] + z[2] - 15.),                            # d/dz[4]
            self.LamL2 + self.rho*(z[5] + z[0] - z[3] - 15.),                              # d/dz[5]
            self.LamF1 + self.rho*(-34. + 2.*z[2] + (8/3.)*z[3] + z[6]),                   # d/dz[6]
            - self.LamF2 + self.rho*(-24.25 + 1.25*z[2] + 2.*z[3] + z[7])                  # d/dz[7]
        ])

    def hess(self, x):
        rho = self.rho
        return np.array([
            #    d/dz0,     d/dz1,           d/dz2,           d/dz3,  d/dz4,  d/dz5,    d/dz6,  d/dz7,
            [ 1. + rho,        0.,             -1.,            -rho,     0.,    rho,       0.,       0.],
            [       0.,  1. + rho,             rho,             -1.,    rho,     0.,       0.,       0.],
            [      -1.,       rho, 1. + 6.5625*rho,       47/6.*rho,    rho,     0.,   2.*rho, 5/4.*rho],
            [     -rho,       -1.,       47/6.*rho, 1. + 109/9.*rho,     0.,   -rho, 8/3.*rho,   2.*rho],
            [       0.,       rho,             rho,              0.,    rho,     0.,       0.,       0.],
            [      rho,        0.,              0.,            -rho,     0.,    rho,       0.,       0.],
            [       0.,        0.,          2.*rho,        8/3.*rho,     0.,     0.,      rho,       0.],
            [       0.,        0.,        5/4.*rho,          2.*rho,     0.,     0.,       0.,      rho]])


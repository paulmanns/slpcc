import numpy as np
import unittest

import slpcc.nlpcc as nlpcc

class TestNlpcc(unittest.TestCase):
    def test_compute_lpcc_cand(self):
        rho = 1.
        l0 = np.array([-1., 0.,  1., -1.,   1., 0.375])
        x0 = np.array([-1., 0.5, 1., -0.5,  2., 0.835])
        u0 = np.array([ 1., 1.,  4., -0.5,  2., 3.])
        g0 = np.array([ 1., 0., -3.,  0.5, -1., 1.75])

        x0_cmp = np.array([-1., 0.5, 2., -1., 2., 0.375])
        pred_lpcc_cmp = 0. + 0. + 3. + 0.25 + 0.805
        #              l1l2    l1u2     u1u2     u1l2     l1u1     l2u2
        l1 = np.array([-2.,    1.,      np.NINF, np.NINF, -1.,     np.NINF])
        x1 = np.array([-1.5,   1.,      -1.,     2.5,     0.5,     0.5])
        u1 = np.array([np.Inf, np.Inf,  2.,      2.5,     1.,      np.Inf])
        g1 = np.array([-1.5,   0.5,     -1.5,    1.,      -2.,     2.])
        l2 = np.array([3.,     np.NINF, np.NINF, -4.,     np.NINF, -0.5])
        x2 = np.array([3.,     4.,      1.,      -4.,     0.,      -0.5])
        u2 = np.array([np.Inf, 4.5,     1.,      np.Inf,  np.Inf,  0.5])
        g2 = np.array([-5.,    -1.,     -7,      -1.,     -0.5,    -0.5])
        cc_code = nlpcc.get_cc_code(l1, u1, l2, u2)

        self.assertTrue(((l0 <= x0) & (x0 <= u0)).all())
        self.assertTrue(((l1 <= x1) & (x1 <= u1)).all())
        self.assertTrue(((l2 <= x2) & (x2 <= u2)).all())

        x1_cmp = np.array([-2., 1.,  0., 2.5, 1., -0.5])
        x2_cmp = np.array([ 4., 4.5, 1., -3., 0.,  0.5])
        pred_lpcc_cmp += 4.25 + 0.5 + 1.5 + 1. + 1. + 2.5
        x_lpcc_cmp = np.hstack((x0_cmp, x1_cmp, x2_cmp))

        n0, n12 = x0.size, x1.size
        nx = n0 + 2 * n12
        mask_n0, mask_n1, mask_n2 = np.zeros(nx, dtype=bool), np.zeros(nx, dtype=bool), np.zeros(nx, dtype=bool)
        mask_n0[:n0] = True
        mask_n1[n0:n0 + n12] = True
        mask_n2[n0 + n12:] = True

        x = np.hstack((x0, x1, x2))
        self.assertTrue(nlpcc.x_feasible(x, mask_n0, mask_n1, mask_n2, l0, u0, l1, u1, l2, u2, cc_code))
        self.assertTrue(nlpcc.x_feasible(x_lpcc_cmp, mask_n0, mask_n1, mask_n2, l0, u0, l1, u1, l2, u2, cc_code))
        x_lpcc, pred_lpcc, ias0, ias1, ias2 = \
            nlpcc.compute_lpcc_cand(x0, x1, x2, g0, g1, g2, l0, l1, l2, u0, u1, u2, rho, cc_code)
        self.assertTrue((ias0 == np.array([False, True,  True,  False, False, False], dtype=bool)).all())
        self.assertTrue((ias1 == np.array([False, False,  True,  False, True,  True ], dtype=bool)).all())
        self.assertTrue((ias2 == np.array([True,  True,   False, True,  False, False], dtype=bool)).all())
        self.assertTrue(nlpcc.x_feasible(x_lpcc, mask_n0, mask_n1, mask_n2, l0, u0, l1, u1, l2, u2, cc_code))
        self.assertTrue(np.allclose(x_lpcc_cmp, x_lpcc))
        self.assertAlmostEqual(pred_lpcc, pred_lpcc_cmp)


class TestQpwlExamples(unittest.TestCase):
    def test_solve_simple_model_jr1al_bqp(self):
        import slpcc.examples.jr1al as jr1al
        ex = jr1al.Jr1AL()
        f, x, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init, ex.l, ex.u, ex.n0, ex.n12, use_bqp=True, disp=False)
        self.assertTrue(np.allclose(x, ex.opt_x))
        self.assertTrue(np.isclose(f, ex.opt_f))
        f, x, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init, ex.l, ex.u, ex.n0, ex.n12, use_bqp=True, use_qpwl=True, disp=False)
        self.assertTrue(np.allclose(x, ex.opt_x))
        self.assertTrue(np.isclose(f, ex.opt_f))

    def test_solve_simple_model_jr1al(self):
        import slpcc.examples.jr1al as jr1al
        ex = jr1al.Jr1AL()
        f, x, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init, ex.l, ex.u, ex.n0, ex.n12, disp=False)
        self.assertTrue(np.allclose(x, ex.opt_x))
        self.assertTrue(np.isclose(f, ex.opt_f))
        f, x, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init, ex.l, ex.u, ex.n0, ex.n12, use_qpwl=True, disp=False)
        self.assertTrue(np.allclose(x, ex.opt_x))
        self.assertTrue(np.isclose(f, ex.opt_f))

    def test_solve_simple_models_bqp(self):
        import slpcc.examples.gnash1mal as gnash1mal
        import slpcc.examples.gnash2mal as gnash2mal
        ex = gnash1mal.Gnash1MAL()

        fa, xa, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init,
            ex.l, ex.u, ex.n0, ex.n12, max_iter=2000, use_qpwl=True, use_bqp=True, disp=False)
        self.assertTrue(np.allclose(xa, ex.opt_x))
        self.assertTrue(np.allclose(fa, ex.opt_f))
        
        fb, xb, _, _, _, _ = nlpbreacc.solve(ex.fun, ex.grad, ex.hess, np.array([0.1, 0.2, 1e3, -1e5, 0.15, 1., 0.2, -1.]),
            ex.l, ex.u, ex.n0, ex.n12, max_iter=2000, use_qpwl=True, use_bqp=True, disp=False)
        self.assertTrue(np.allclose(xb, ex.opt_x))
        self.assertTrue(np.isclose(fb, ex.opt_f))

        ex = gnash2mal.Gnash2MAL()
        fa, xa, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init,
            ex.l, ex.u, ex.n0, ex.n12, max_iter=20000, use_qpwl=True, use_bqp=True, disp=False
        )
        fb, xb, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, np.array([0.1, 0.2, 1e3, -1e5, 0.15, 1., 0.2, -1.]),
            ex.l, ex.u, ex.n0, ex.n12, max_iter=20000, use_qpwl=True, use_bqp=True, disp=False
        )
        #TODO: Add assertions on return value accuracy etc.
        #self.assertTrue(np.allclose(xa, ex.opt_x))
        #self.assertTrue(np.allclose(xb, ex.opt_x))
        self.assertTrue(np.allclose(fa, ex.opt_f))
        self.assertTrue(np.isclose(fb, ex.opt_f))

    def test_solve_simple_models(self):
        import slpcc.examples.gnash1mal as gnash1mal
        import slpcc.examples.gnash2mal as gnash2mal

        ex = gnash1mal.Gnash1MAL()
        fa, xa, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init,
            ex.l, ex.u, ex.n0, ex.n12, max_iter=2000, use_qpwl=True, disp=False)
        self.assertTrue(np.allclose(xa, ex.opt_x))
        self.assertTrue(np.allclose(fa, ex.opt_f))

        fb, xb, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, np.array([0.1, 0.2, 1e3, -1e5, 0.15, 1., 0.2, -1.]),
            ex.l, ex.u, ex.n0, ex.n12, max_iter=2000, use_qpwl=True, disp=False)
        self.assertTrue(np.allclose(xb, ex.opt_x))
        self.assertTrue(np.isclose(fb, ex.opt_f))

        ex = gnash2mal.Gnash2MAL()
        fa, xa, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init,
            ex.l, ex.u, ex.n0, ex.n12, max_iter=20000, use_qpwl=True, disp=False
        )
        fb, xb, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, np.array([0.1, 0.2, 1e3, -1e5, 0.15, 1., 0.2, -1.]),
            ex.l, ex.u, ex.n0, ex.n12, max_iter=20000, use_qpwl=True, disp=False
        )
        #TODO: Add assertions on return value accuracy etc.
        #self.assertTrue(np.allclose(xa, ex.opt_x))
        #self.assertTrue(np.allclose(xb, ex.opt_x))
        self.assertTrue(np.allclose(fa, ex.opt_f))
        self.assertTrue(np.isclose(fb, ex.opt_f))        
        

class TestNlpccExamples(unittest.TestCase):
    def test_solve_simple_model_example1(self):
        import slpcc.examples.example1 as example1
        ex = example1.Example1()
        f, x, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init, ex.l, ex.u, ex.n0, ex.n12, disp=False)
        self.assertTrue(np.allclose(x, ex.opt_x))
        self.assertTrue(np.isclose(f, ex.opt_f))

    def test_solve_simple_model_example2(self):
        import slpcc.examples.example2 as example2
        ex = example2.Example2()
        f, x, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init, ex.l, ex.u, ex.n0, ex.n12, disp=False)
        self.assertTrue(np.allclose(x, ex.opt_x))
        self.assertTrue(np.isclose(f, ex.opt_f))        

    def test_solve_simple_model_example3(self):
        import slpcc.examples.example3 as example3
        ex = example3.Example3()
        f, x, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init, ex.l, ex.u, ex.n0, ex.n12, disp=False)
        self.assertTrue(np.allclose(x, ex.opt_x))
        self.assertTrue(np.isclose(f, ex.opt_f))            

    def test_solve_simple_model_example4(self):
        import slpcc.examples.example4 as example4
        ex = example4.Example4()
        f, x, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init, ex.l, ex.u, ex.n0, ex.n12, disp=False)
        self.assertTrue(np.allclose(x, ex.opt_x))
        self.assertTrue(np.isclose(f, ex.opt_f))     

    def test_solve_simple_model_example5(self):
        import slpcc.examples.example5 as example5
        ex = example5.Example5()
        f, x, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init, ex.l, ex.u, ex.n0, ex.n12, disp=False)
        self.assertTrue(np.allclose(x, ex.opt_x))
        self.assertTrue(np.isclose(f, ex.opt_f))   

    def test_solve_simple_model_example6(self):
        import slpcc.examples.example6 as example6
        ex = example6.Example6()
        f, x, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init, ex.l, ex.u, ex.n0, ex.n12, disp=False)
        self.assertTrue(np.allclose(x, ex.opt_x))
        self.assertTrue(np.isclose(f, ex.opt_f))       

    def test_solve_simple_model_jr1al(self):
        import slpcc.examples.jr1al as jr1al
        ex = jr1al.Jr1AL()
        f, x, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init, ex.l, ex.u, ex.n0, ex.n12, disp=False)
        self.assertTrue(np.allclose(x, ex.opt_x))
        self.assertTrue(np.isclose(f, ex.opt_f))

    def test_solve_simple_model_jr1al_bqp(self):
        import slpcc.examples.jr1al as jr1al
        ex = jr1al.Jr1AL()
        f, x, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init, ex.l, ex.u, ex.n0, ex.n12, use_bqp=True, disp=False)
        self.assertTrue(np.allclose(x, ex.opt_x))
        self.assertTrue(np.isclose(f, ex.opt_f))

    def test_solve_simple_model_jr2al(self):
        import slpcc.examples.jr2al as jr2al
        ex = jr2al.Jr2AL()
        f, x, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init, ex.l, ex.u, ex.n0, ex.n12, disp=False)
        self.assertTrue(np.allclose(x, ex.opt_x))
        self.assertTrue(np.isclose(f, ex.opt_f))

    def test_solve_simple_model_jr2al_bqp(self):
        import slpcc.examples.jr2al as jr2al
        ex = jr2al.Jr2AL()
        f, x, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init, ex.l, ex.u, ex.n0, ex.n12, use_bqp=True, disp=False)
        self.assertTrue(np.allclose(x, ex.opt_x))
        self.assertTrue(np.isclose(f, ex.opt_f))

    def test_solve_simple_models(self):
        import slpcc.examples.gnash1mal as gnash1mal
        import slpcc.examples.gnash2mal as gnash2mal
        ex = gnash1mal.Gnash1MAL()
        fa, xa, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init,
            ex.l, ex.u, ex.n0, ex.n12, max_iter=2000, disp=False)
        fb, xb, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, np.array([0.1, 0.2, 1e3, -1e5, 0.15, 1., 0.2, -1.]),
            ex.l, ex.u, ex.n0, ex.n12, max_iter=2000, disp=False)
        self.assertTrue(np.allclose(xa, ex.opt_x))
        self.assertTrue(np.allclose(fa, ex.opt_f))
        self.assertTrue(np.allclose(xa, xb))
        self.assertTrue(np.isclose(fa, fb))

        ex = gnash2mal.Gnash2MAL()
        fa, xa, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init,
            ex.l, ex.u, ex.n0, ex.n12, max_iter=10000, disp=False)
        fb, xb, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, np.array([0.1, 0.2, 1e3, -1e5, 0.15, 1., 0.2, -1.]),
            ex.l, ex.u, ex.n0, ex.n12, max_iter=10000, disp=False
        )
        #TODO: Add assertions on return value accuracy etc.
        #self.assertTrue(np.allclose(xa, ex.opt_x))
        #self.assertTrue(np.allclose(xa, xb))
        self.assertTrue(np.allclose(fa, ex.opt_f))
        self.assertTrue(np.isclose(fa, fb))

    def test_solve_simple_models_bqp(self):
        import slpcc.examples.gnash1mal as gnash1mal
        import slpcc.examples.gnash2mal as gnash2mal
        ex = gnash1mal.Gnash1MAL()
        fa, xa, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init,
            ex.l, ex.u, ex.n0, ex.n12, max_iter=2000,  use_bqp=True, disp=False)
        fb, xb, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, np.array([0.1, 0.2, 1e3, -1e5, 0.15, 1., 0.2, -1.]),
            ex.l, ex.u, ex.n0, ex.n12, max_iter=2000, use_bqp=True, disp=False)
        self.assertTrue(np.allclose(xa, ex.opt_x))
        self.assertTrue(np.allclose(fa, ex.opt_f))
        self.assertTrue(np.allclose(xa, xb))
        self.assertTrue(np.isclose(fa, fb))

        ex = gnash2mal.Gnash2MAL()
        fa, xa, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, ex.x_init,
            ex.l, ex.u, ex.n0, ex.n12, max_iter=10000, use_bqp=True, disp=False
        )
        fb, xb, _, _, _, _ = nlpcc.solve(ex.fun, ex.grad, ex.hess, np.array([0.1, 0.2, 1e3, -1e5, 0.15, 1., 0.2, -1.]),
            ex.l, ex.u, ex.n0, ex.n12, max_iter=10000, use_bqp=True, disp=False
        )
        #TODO: Add assertions on return value accuracy etc.
        #self.assertTrue(np.allclose(xa, ex.opt_x))
        #self.assertTrue(np.allclose(xa, xb))
        self.assertTrue(np.allclose(fa, ex.opt_f))
        self.assertTrue(np.isclose(fa, fb))        

if __name__ == '__main__':
    unittest.main()

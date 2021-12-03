import copy
import numpy as np
from enum import IntEnum

import slpcc.quadratic_cauchy_improvement as qpwl

class CCCode(IntEnum):
    L1_FINITE = 0
    U1_FINITE = 1
    L2_FINITE = 2
    U2_FINITE = 3


class ReturnCode(IntEnum):
    GRADIENT_TOLERANCE_REACHED = 0
    STEPSIZE_TOLERANCE_REACHED = 1
    ITERATION_LIMIT_REACHED = 2


def solve(
        fun, grad, hess,
        x_init,
        l, u, n0, n12,
        max_iter=100, gtol=1e-7, rho_bar=1.0, max_inner_iter=50,
        sigma=1e-1, use_qpwl=False, use_bqp=False,
        bqp_t_7_ok=False,
        disp=True):
    dispprint = print if disp else lambda *a, **k: None

    # Number of variables
    nx = n0 + 2*n12

    # Masks for $x_0$, $x_1$, $x_2$
    mask_n0, mask_n1, mask_n2 = np.zeros(nx, dtype=bool), np.zeros(nx, dtype=bool), np.zeros(nx, dtype=bool)
    
    # - The following three lines have to be replaced once we allow variables in different order
    mask_n0[:n0] = True
    mask_n1[n0:n0+n12] = True
    mask_n2[n0+n12:] = True
    mask_n12 = mask_n1 | mask_n2

    # Lower and upper bounds for $x_0$, $x_1$, $x_2$, $(x_1^T x_2^T)^T$
    l0, u0, l1, u1, l2, u2, l12, u12 = get_bounds(l, u, mask_n0, mask_n1, mask_n2, mask_n12)

    # Determine type of complementarity
    # (True indicates finite bound, False indicates infinite bound)
    cc_code = get_cc_code(l1, u1, l2, u2)

    # Check shapes and bound consistency
    check_shapes(n0, n12, nx, x_init, l0, u0, l1, u1, l2, u2, l12, u12)
    check_bound_consistency(l0, u0, l1, u1, l2, u2, cc_code)

    # Check (and repair) feasibility of initial guess
    if not x_feasible(x_init, mask_n0, mask_n1, mask_n2, l0, u0, l1, u1, l2, u2, cc_code):
        dispprint(r'Provided initial guess is not feasible. Compute max-norm-projection to feasible set.')
        # Compute one $\|\cdot\|_\infty$-projection on to the feasible set.
        # The $\|\cdot\|_\infty$-projection is not unique.
        # First, project to bounds.
        x0_init, x1_init, x2_init = project_to_bounds(x_init, l0, u0, l1, u1, l2, u2, mask_n0, mask_n1, mask_n2)
        # Second, after bounds are respected, project to complementarity.
        project_to_complementarity(x_init, x1_init, x2_init, l1, u1, l2, u2, mask_n1, mask_n2, cc_code)
        # Third, overwrite x_init with projection.
        x_init[mask_n0], x_init[mask_n1], x_init[mask_n2] = x0_init, x1_init, x2_init
    assert x_feasible(x_init, mask_n0, mask_n1, mask_n2, l0, u0, l1, u1, l2, u2, cc_code)

    # Setup initial iterate
    rho_bar_k = rho_bar
    rho_qp_bar = 2. * rho_bar
    rho_qp = rho_qp_bar
    x_k = copy.deepcopy(x_init)
    f_k, g_k = fun(x_k), grad(x_k)
    eps_k, eps_x0, eps_x12, g0, g1, g2, x0, x1, x2 = \
        get_iter_vars(x_k, g_k, mask_n0, mask_n1, mask_n2, l0, u0, l1, u1, l2, u2, cc_code)

    assert (not use_qpwl) or (
        cc_code[0].all() \
        and (not cc_code[1].any()) \
        and cc_code[2].all() \
        and (not cc_code[3].any())
    )
    
    # Setup printing
    dispprint('SI  It.  In. It.             fk     |xk - xk-1|       fk - fk-1          Rho n            eps')
    dispprint('---------------------------------------------------------------------------------------------')
    dispprint('SI = LA/CA/LR/LO (LPCC Step Accepted / Cauchy Step Accepted / REJECTED / Stationary)')
    dispprint('SI = BA/BR (Bound Optimization Accepted / Rejected) ')
    dispprint('---------------------------------------------------------------------------------------------')
    dispprint_iterate = lambda si, k, n, f, dx, df, r, e:\
        dispprint('%s %4d     %4d    %+.4e     %+.4e     %+.4e     %.4e     %.4e' % (si, k, n, f, dx, df, r, e))
    dispprint_iterate('--', 0, 0, f_k, -1, -1, 0, 1e10)

    # Setup tracking of workload
    k, cum_n_lpcc, cum_n_qp = 0, 0, 0

    # Return Code
    #    -1 = Error
    #     0 = B-stationary (predicted reduction is zero before TR contraction)
    #     1 = approximately B-stationary (max-norm of projected gradient below gtol)
    #     2 = TR contraction
    #     3 = Iteration limit reached
    return_code = 3

    # Compute new iterate until iteration limit or other termination criterion invoked
    for k in range(max_iter):
        # Prepare QP    
        if use_qpwl or use_bqp:
            A = hess(x_k)
            b = g_k - .5*(x_k.dot(A).transpose() + A.dot(x_k))

        # Reset trust region
        rho_n = rho_bar_k

        # Reduce trust region until sufficient decrease candidate is found.
        # If function is sufficiently regular, such a candidate is found eventually
        # (neglecting finite precision arithmetics).
        iterate_found, stationary, si_cand = False, False, 'LR'
        for n in range(max_inner_iter):
            cum_n_lpcc += 1
            x_lpcc, pred_lpcc, ias0, ias1, ias2 = \
                compute_lpcc_cand(x0, x1, x2, g0, g1, g2, l0, l1, l2, u0, u1, u2, rho_n, cc_code)

            if pred_lpcc == 0.:
                si_cand = 'LS'
                stationary = True
                break

            if use_qpwl:
                l_qpwl, u_qpwl = np.maximum(l, x_k - rho_n), np.minimum(u, x_k + rho_n)
                x_qpwl, mc, ias0_, ias1_, ias2_ = qpwl.solve_qpwl(A, b, x_k, g_k, mask_n0, mask_n12, l_qpwl, u_qpwl, cc_code, gtol)

                # QPWL disabled => always check LPCC step
                f_qpwl = fun(x_qpwl)
                #f_lpcc = fun(x_lpcc)
                ared_qpwl = f_k - f_qpwl
                if ared_qpwl >= sigma * pred_lpcc:
                    # Accept LPCC step
                    ias0, ias1, ias2 = ias0_, ias1_, ias2_
                    iterate_found = True
                    ared_cand = ared_qpwl
                    si_cand = 'QA'
                    x_cand, f_cand = x_qpwl, f_qpwl
                    break
            
            if not iterate_found:
                # QPWL disabled => always check LPCC step
                f_lpcc = fun(x_lpcc)
                ared_lpcc = f_k - f_lpcc
                if ared_lpcc >= sigma * pred_lpcc:
                    # Accept LPCC step
                    iterate_found = True
                    ared_cand = ared_lpcc
                    si_cand = 'LA'
                    x_cand, f_cand = x_lpcc, f_lpcc
                    break

            # Reduce trust region
            rho_n *= 0.5
        cum_n_lpcc += n

        if stationary:
            eps_k, eps_x0, eps_x12, g0, g1, g2, x0, x1, x2 = \
                get_iter_vars(x_k, g_k, mask_n0, mask_n1, mask_n2, l0, u0, l1, u1, l2, u2, cc_code)            
            dispprint_iterate(si_cand, k + 1, n + 1, f_k, 0., 0., rho_n, eps_k)
            return_code = 0
            break

        if iterate_found:
            ndx_k = np.linalg.norm(x_k - x_cand)
            x_k, f_k, ared_k = x_cand, f_cand, ared_cand
            g_k = grad(x_k)
            eps_k, eps_x0, eps_x12, g0, g1, g2, x0, x1, x2 = \
                get_iter_vars(x_k, g_k, mask_n0, mask_n1, mask_n2, l0, u0, l1, u1, l2, u2, cc_code)
        else:
            ndx_k, ared_k = 0., 0.
            
        dispprint_iterate(si_cand, k + 1, n + 1, f_k, ndx_k, ared_k, rho_n, eps_k)

        if not iterate_found:
            return_code = 2
            break
        
        if eps_k < gtol:
            return_code = 1
            break

        if use_bqp:
            si_cand, ared_k, ndx_k = 'BR', 0., 0.
            x_out, q_out, q_0, n_qp_it = run_bqp_solve(dispprint,x_k, A, g_k, f_k, ias0, ias1, ias2, l, u, n0, n12, mask_n0, mask_n12, rho_qp, bqp_t_7_ok)
            cum_n_qp += n_qp_it
            if q_0 - q_out > 0.:
                f_out = fun(x_out)
                ratio = (f_k - f_out) / (q_0 - q_out)
            else:
                ratio = 0.
            
            if ratio > 0.:
                si_cand = 'BA'
                ndx_k = np.linalg.norm(x_k - x_cand)
                x_k, f_k, ared_k = x_out, f_out, f_k - f_out
                g_k = grad(x_k)
                eps_k, eps_x0, eps_x12, g0, g1, g2, x0, x1, x2 = \
                    get_iter_vars(x_k, g_k, mask_n0, mask_n1, mask_n2, l0, u0, l1, u1, l2, u2, cc_code)

            dispprint_iterate(si_cand, k + 1, n_qp_it, f_k, ndx_k, ared_k, rho_qp, eps_k)

            if ratio >= 0.75 and q_0 - q_out > 0.:
                rho_qp = np.min([2.*rho_qp, rho_qp_bar])
            elif ratio < 0.25:
                rho_qp *= .25

            if rho_qp < 1e-8:
                rho_qp = rho_qp_bar

            if eps_k < gtol:
                return_code = 1
                break           
    
    k += 1
    if return_code == 3:
        k = max_iter

    # print(eps_k)

    # RETURN:
    #   function value
    #   final iterate
    #   number of outer iterations
    #   number of inner iterations
    #   number of BQP iterations
    #   return code
    return f_k, x_k, k, cum_n_lpcc, cum_n_qp, return_code


def run_bqp_solve(dispprint, x, Q, g, f, ias0, ias1, ias2, l, u, n0, n12, mask_n0, mask_n12, rho_bqp, bqp_t_7_ok):
    # Setup quadratic model
    f00 = f - x.dot(g) + .5 * x.dot(Q.dot(x))
    g00 = g - .5 * Q.dot(x) - .5 * Q.transpose().dot(x)
    Q00 = Q

    idx_n12 = np.sort(np.where(mask_n12 == True)[0])
    mc_bnd = np.hstack((np.sort(np.where(mask_n0 == True)[0]), idx_n12[:n12][ias1], idx_n12[n12:][ias2]))

    assert n0 + n12 == mc_bnd.shape[0]

    # Transformation: x = Z * y + z
    Z = np.zeros((x.shape[0], n0 + n12))
    Z[mc_bnd, range(n0 + n12)] = 1.
    z = copy.deepcopy(x)
    z[mc_bnd] = 0.
    Qr = Z.transpose().dot(Q00.dot(Z))
    gr = .5 * z.transpose().dot(Q00.dot(Z)) + .5 * z.transpose().dot(Q00.transpose().dot(Z)) + g00.transpose().dot(Z)
    fr = .5 * z.dot(Q00.dot(z)) + g00.dot(z)
    
    assert np.allclose(Qr, Qr.transpose())

    lr = np.maximum(l[mc_bnd], x[mc_bnd] - rho_bqp)
    ur = np.minimum(u[mc_bnd], x[mc_bnd] + rho_bqp)
    xr = copy.deepcopy(x[mc_bnd])

    x_out = copy.deepcopy(x)
    x_out[mc_bnd] = np.minimum(ur, np.maximum(lr, np.array(xr)))
    n_qp_it = 0
    
    q_out = .5 * x_out.dot(Q00.dot(x_out)) + g00.dot(x_out) + f00
    q_0 = .5 * x.dot(Q00.dot(x)) + g00.dot(x) + f00

    return x_out, q_out, q_0, n_qp_it


def get_iter_vars(x_k, g_k, mask_n0, mask_n1, mask_n2, l0, u0, l1, u1, l2, u2, cc_code):
    x0, x1, x2 = x_k[mask_n0], x_k[mask_n1], x_k[mask_n2]
    g0, g1, g2 = g_k[mask_n0], g_k[mask_n1], g_k[mask_n2]
    eps_x0 = compute_x0_stationarity(x0, g0, l0, u0)
    eps_x12 = compute_x1x2_stationarity(x1, x2, g1, g2, l1, u1, l2, u2, cc_code)
    eps_k = np.max([eps_x0, eps_x12])
    return eps_k, eps_x0, eps_x12, g0, g1, g2, x0, x1, x2


def compute_x1x2_stationarity(x1, x2, g1, g2, l1, u1, l2, u2, cc_code):
    assert x1.size == x2.size
    assert x1.size == g1.size
    assert x1.size == g2.size
    assert x1.size == l1.size
    assert x1.size == u1.size
    assert x1.size == l2.size
    assert x1.size == u2.size
    if x1.size == 0:
        return 0.

    l1l2 = cc_code[CCCode.L1_FINITE] & cc_code[CCCode.L2_FINITE]
    l1u2 = cc_code[CCCode.L1_FINITE] & cc_code[CCCode.U2_FINITE]
    u1u2 = cc_code[CCCode.U1_FINITE] & cc_code[CCCode.U2_FINITE]
    u1l2 = cc_code[CCCode.U1_FINITE] & cc_code[CCCode.L2_FINITE]
    mask_1_in1 = (l1l2 | l1u2 | u1u2 | u1l2) & (l1 < x1) & (x1 < u1)
    mask_1_in2 = (l1l2 | l1u2 | u1u2 | u1l2) & (l2 < x2) & (x2 < u2)
    mask_1_lb1 = (l1l2 | l1u2 | u1u2 | u1l2) & (l1 == x1)
    mask_1_lb2 = (l1l2 | l1u2 | u1u2 | u1l2) & (l2 == x2)
    mask_1_ub1 = (l1l2 | l1u2 | u1u2 | u1l2) & (x1 == u1)
    mask_1_ub2 = (l1l2 | l1u2 | u1u2 | u1l2) & (x2 == u2)
    assert not (mask_1_in1 & mask_1_in2).any()
    assert not (mask_1_lb1 & mask_1_ub1).any()
    assert not (mask_1_lb2 & mask_1_ub2).any()
    assert (mask_1_in1.astype(int) + mask_1_in2.astype(int) +
            mask_1_lb1.astype(int) + mask_1_lb2.astype(int) +
            mask_1_ub1.astype(int) + mask_1_ub2.astype(int)
            == (l1l2 | l1u2 | u1u2 | u1l2).astype(int) * 2).all()

    l1u1 = cc_code[CCCode.L1_FINITE] & cc_code[CCCode.U1_FINITE]
    l2u2 = cc_code[CCCode.L2_FINITE] & cc_code[CCCode.U2_FINITE]
    mask_2_in1 = (l1u1 & (l1 < x1) & (x1 < u1)) | (l2u2 & ((0 < x1) | (x1 < 0)))
    mask_2_in2 = (l1u1 & ((0 < x2) | (x2 < 0))) | (l2u2 & (l2 < x2) & (x2 < u2))
    mask_2_lb1 = (l1u1 & (l1 == x1)) | (l2u2 & (x1 == 0) & (l2 == x2))
    mask_2_lb2 = (l1u1 & (l1 == x1) & (x2 == 0)) | (l2u2 & (l2 == x2))
    mask_2_ub1 = (l1u1 & (x1 == u1)) | (l2u2 & (x1 == 0) & (x2 == u2))
    mask_2_ub2 = (l1u1 & (x1 == u1) & (x2 == 0)) | (l2u2 & (x2 == u2))
    assert not (mask_2_in1 & mask_2_in2).any()
    assert (l1u1 & (mask_2_in1 | mask_2_lb1 | mask_2_ub1) == l1u1).all()
    assert (l2u2 & (mask_2_in2 | mask_2_lb2 | mask_2_ub2) == l2u2).all()

    mask_in1 = mask_1_in1 | mask_2_in1
    mask_in2 = mask_1_in2 | mask_2_in2
    mask_lb1 = (mask_1_lb1 | mask_2_lb1) & (~mask_in2)
    mask_lb2 = (mask_1_lb2 | mask_2_lb2) & (~mask_in1)
    mask_ub1 = (mask_1_ub1 | mask_2_ub1) & (~mask_in2)
    mask_ub2 = (mask_1_ub2 | mask_2_ub2) & (~mask_in1)
    stat_in1 = np.max(np.abs(g1[mask_in1])) if mask_in1.any() else 0.
    stat_in2 = np.max(np.abs(g2[mask_in2])) if mask_in2.any() else 0.
    stat_lb1 = np.max(np.abs(np.minimum(g1[mask_lb1], np.zeros(np.sum(mask_lb1))))) if mask_lb1.any() else 0.
    stat_lb2 = np.max(np.abs(np.minimum(g2[mask_lb2], np.zeros(np.sum(mask_lb2))))) if mask_lb2.any() else 0.
    stat_ub1 = np.max(np.abs(np.maximum(g1[mask_ub1], np.zeros(np.sum(mask_ub1))))) if mask_ub1.any() else 0.
    stat_ub2 = np.max(np.abs(np.maximum(g2[mask_ub2], np.zeros(np.sum(mask_ub2))))) if mask_ub2.any() else 0.
    return stat_in1 + stat_in2 + stat_lb1 + stat_lb2 + stat_ub1 + stat_ub2


def compute_x0_stationarity(x0, g0, l0, u0):
    assert x0.size == g0.size
    assert x0.size == l0.size
    assert x0.size == u0.size
    if x0.size == 0:
        return 0.
    mask_in = (l0 < x0) & (x0 < u0)
    mask_lb = (l0 == x0)
    mask_ub = (x0 == u0)
    assert (mask_in | mask_lb | mask_ub).all()
    assert not (mask_in & mask_lb).any()
    assert not (mask_in & mask_ub).any()
    assert not (mask_lb & mask_ub).any()
    stat_in = np.max(np.abs(g0[mask_in])) if mask_in.any() else 0.
    stat_lb = np.max(np.abs(np.minimum(g0[mask_lb], np.zeros(np.sum(mask_lb))))) if mask_lb.any() else 0.
    stat_ub = np.max(np.abs(np.maximum(g0[mask_ub], np.zeros(np.sum(mask_ub))))) if mask_ub.any() else 0.
    return stat_in + stat_lb + stat_ub


def get_cc_code(l1, u1, l2, u2):
    cc_code = np.array([np.NINF < l1, u1 < np.Inf, np.NINF < l2, u2 < np.Inf])
    return cc_code


def compute_lpcc_cand(x0, x1, x2, g0, g1, g2, l0, l1, l2, u0, u1, u2, rho, cc_code):
    d0 = rho * -np.sign(g0)
    # Solve LP in coordinates of $x_0$
    x0_cand = np.maximum(l0, np.minimum(x0 + d0, u0))
    ias0 = (l0 < x0_cand) & (x0_cand < u0)

    # Solve LP in complementary coordinates
    x1_cand, x2_cand, ias1, ias2 = \
        compute_lpcc_x1x2_cand(x1, x2, g1, g2, rho, l1, u1, l2, u2, cc_code)

    pred_d0 = -g0.dot(x0_cand - x0)
    pred_d1 = -g1.dot(x1_cand - x1)
    pred_d2 = -g2.dot(x2_cand - x2)

    assert pred_d0 >= 0.
    assert pred_d1 + pred_d2 >= 0.
    return np.hstack((x0_cand, x1_cand, x2_cand)), pred_d0 + pred_d1 + pred_d2, ias0, ias1, ias2


def compute_lpcc_x1x2_cand(x1, x2, g1, g2, rho, l1, u1, l2, u2, cc_code):
    # Assumes that x1, x2 are feasible
    # Assumes that x1, x2 are exactly on the bound values if active
    n12 = x1.size

    # Compute and tabulate predicted reductions for
    # 0: (x1 + d, x2 + d) | 1: (x1 + d, x2    ) | 2: (x1 + d, x2 - d)
    # 3: (x1,     x2 + d) | 4: (x1,     x2    ) | 5: (x1,     x2 - d)
    # 6: (x1 - d, x2 + d) | 7: (x1 - d, x2    ) | 8: (x1 - d, x2 - d)
    pred_table = np.zeros((x1.size, 9))
    x1_pos, x2_pos = np.minimum(x1 + rho, u1), np.minimum(x2 + rho, u2)
    x1_neg, x2_neg = np.maximum(x1 - rho, l1), np.maximum(x2 - rho, l2)
    d1_pos, d2_pos, d1_neg, d2_neg = x1_pos - x1, x2_pos - x2, x1_neg - x1, x2_neg - x2
    
    pred_table[:, 0] = -g1 * d1_pos - g2 * d2_pos
    pred_table[:, 1] = -g1 * d1_pos
    pred_table[:, 2] = -g1 * d1_pos - g2 * d2_neg
    pred_table[:, 3] = -g2 * d2_pos
    pred_table[:, 4] = 0.
    pred_table[:, 5] = -g2 * d2_neg
    pred_table[:, 6] = -g1 * d1_neg - g2 * d2_pos
    pred_table[:, 7] = -g1 * d1_neg
    pred_table[:, 8] = -g1 * d1_neg - g2 * d2_neg

    # Check if x1_pos, x2_pos, x1_neg, x2_neg satisfy bounds
    assert (l1 <= x1_pos).all() and (x1_pos <= u1).all()
    assert (l2 <= x2_pos).all() and (x2_pos <= u2).all()
    assert (l1 <= x1_neg).all() and (x1_neg <= u1).all()
    assert (l2 <= x2_neg).all() and (x2_neg <= u2).all()

    l1l2 = cc_code[CCCode.L1_FINITE] & cc_code[CCCode.L2_FINITE]
    l1u2 = cc_code[CCCode.L1_FINITE] & cc_code[CCCode.U2_FINITE]
    u1u2 = cc_code[CCCode.U1_FINITE] & cc_code[CCCode.U2_FINITE]
    u1l2 = cc_code[CCCode.U1_FINITE] & cc_code[CCCode.L2_FINITE]
    l1u1 = cc_code[CCCode.L1_FINITE] & cc_code[CCCode.U1_FINITE]
    l2u2 = cc_code[CCCode.L2_FINITE] & cc_code[CCCode.U2_FINITE]

    # Set predicted reduction to negative infinity where l1l2 complementarity is present and violated
    pred_table[l1l2 & (x1_pos > l1) & (x2_pos > l2), 0] = np.NINF
    pred_table[l1l2 & (x1_pos > l1) & (x2     > l2), 1] = np.NINF
    pred_table[l1l2 & (x1_pos > l1) & (x2_neg > l2), 2] = np.NINF
    pred_table[l1l2 & (x1     > l1) & (x2_pos > l2), 3] = np.NINF
    pred_table[l1l2 & (x1     > l1) & (x2     > l2), 4] = np.NINF
    pred_table[l1l2 & (x1     > l1) & (x2_neg > l2), 5] = np.NINF
    pred_table[l1l2 & (x1_neg > l1) & (x2_pos > l2), 6] = np.NINF
    pred_table[l1l2 & (x1_neg > l1) & (x2     > l2), 7] = np.NINF
    pred_table[l1l2 & (x1_neg > l1) & (x2_neg > l2), 8] = np.NINF

    # Set predicted reduction to negative infinity where l1u2 complementarity is present and violated
    pred_table[l1u2 & (x1_pos > l1) & (u2 > x2_pos), 0] = np.NINF
    pred_table[l1u2 & (x1_pos > l1) & (u2 > x2    ), 1] = np.NINF
    pred_table[l1u2 & (x1_pos > l1) & (u2 > x2_neg), 2] = np.NINF
    pred_table[l1u2 & (x1     > l1) & (u2 > x2_pos), 3] = np.NINF
    pred_table[l1u2 & (x1     > l1) & (u2 > x2    ), 4] = np.NINF
    pred_table[l1u2 & (x1     > l1) & (u2 > x2_neg), 5] = np.NINF
    pred_table[l1u2 & (x1_neg > l1) & (u2 > x2_pos), 6] = np.NINF
    pred_table[l1u2 & (x1_neg > l1) & (u2 > x2    ), 7] = np.NINF
    pred_table[l1u2 & (x1_neg > l1) & (u2 > x2_neg), 8] = np.NINF

    # Set predicted reduction to negative infinity where u1u2 complementarity is present and violated
    pred_table[u1u2 & (u1 > x1_pos) & (u2 > x2_pos), 0] = np.NINF
    pred_table[u1u2 & (u1 > x1_pos) & (u2 > x2    ), 1] = np.NINF
    pred_table[u1u2 & (u1 > x1_pos) & (u2 > x2_neg), 2] = np.NINF
    pred_table[u1u2 & (u1 > x1    ) & (u2 > x2_pos), 3] = np.NINF
    pred_table[u1u2 & (u1 > x1    ) & (u2 > x2    ), 4] = np.NINF
    pred_table[u1u2 & (u1 > x1    ) & (u2 > x2_neg), 5] = np.NINF
    pred_table[u1u2 & (u1 > x1_neg) & (u2 > x2_pos), 6] = np.NINF
    pred_table[u1u2 & (u1 > x1_neg) & (u2 > x2    ), 7] = np.NINF
    pred_table[u1u2 & (u1 > x1_neg) & (u2 > x2_neg), 8] = np.NINF

    # Set predicted reduction to negative infinity where u1l2 complementarity is present and violated
    pred_table[u1l2 & (u1 > x1_pos) & (x2_pos > l2), 0] = np.NINF
    pred_table[u1l2 & (u1 > x1_pos) & (x2     > l2), 1] = np.NINF
    pred_table[u1l2 & (u1 > x1_pos) & (x2_neg > l2), 2] = np.NINF
    pred_table[u1l2 & (u1 > x1    ) & (x2_pos > l2), 3] = np.NINF
    pred_table[u1l2 & (u1 > x1    ) & (x2     > l2), 4] = np.NINF
    pred_table[u1l2 & (u1 > x1    ) & (x2_neg > l2), 5] = np.NINF
    pred_table[u1l2 & (u1 > x1_neg) & (x2_pos > l2), 6] = np.NINF
    pred_table[u1l2 & (u1 > x1_neg) & (x2     > l2), 7] = np.NINF
    pred_table[u1l2 & (u1 > x1_neg) & (x2_neg > l2), 8] = np.NINF

    amax_pred = np.argmax(pred_table, axis=1)
    assert (pred_table[range(pred_table.shape[0]), amax_pred] >= 0.).all()

    x1_cand, x2_cand = copy.deepcopy(x1), copy.deepcopy(x2)
    amax_pred_x1_pos = (amax_pred == 0) | (amax_pred == 1) | (amax_pred == 2)
    amax_pred_x1_neg = (amax_pred == 6) | (amax_pred == 7) | (amax_pred == 8)
    amax_pred_x2_pos = (amax_pred == 0) | (amax_pred == 3) | (amax_pred == 6)
    amax_pred_x2_neg = (amax_pred == 2) | (amax_pred == 5) | (amax_pred == 8)
    x1_cand[amax_pred_x1_pos] = x1_pos[amax_pred_x1_pos]
    x1_cand[amax_pred_x1_neg] = x1_neg[amax_pred_x1_neg]
    x2_cand[amax_pred_x2_pos] = x2_pos[amax_pred_x2_pos]
    x2_cand[amax_pred_x2_neg] = x2_neg[amax_pred_x2_neg]

    x1_cand_cc, x2_cand_cc = compute_lpcc_x1x2_cand_cc(x1, x2, g1, g2, l1, u1, l2, u2, l1u1, l2u2, rho)
    x1_cand[l1u1 | l2u2] = x1_cand_cc
    x2_cand[l1u1 | l2u2] = x2_cand_cc

    # Determine inactive set
    pred_higher_in_1 = (x1_cand - x1) * g1 <= (x2_cand - x2) * g2
    pred_higher_in_2 = np.logical_not(pred_higher_in_1)
    inactive_set_1, inactive_set_2 = np.full(n12, False), np.full(n12, False)

    # Inactive set cases -- one entry strictly inactive and one entry active
    inactive_set_1[(l1 < x1_cand) & (x1_cand < u1) & (~l2u2)] = True
    inactive_set_2[(l2 < x2_cand) & (x2_cand < u2) & (~l1u1)] = True
    inactive_set_1[l2u2 & (x1_cand != 0.)] = True
    inactive_set_2[l1u1 & (x2_cand != 0.)] = True

    # Inactive set cases -- two entries active
    inactive_set_1[((l1 == x1_cand) | (x1_cand == u1)) & ((l2 == x2_cand) | (x2_cand == u2)) & pred_higher_in_1] = True
    inactive_set_2[((l1 == x1_cand) | (x1_cand == u1)) & ((l2 == x2_cand) | (x2_cand == u2)) & pred_higher_in_2] = True
    inactive_set_1[l1u1 & ((l1 == x1_cand) | (x1_cand == u1)) & (x2_cand == 0.) & pred_higher_in_1] = True
    inactive_set_2[l1u1 & ((l1 == x1_cand) | (x1_cand == u1)) & (x2_cand == 0.) & pred_higher_in_2] = True
    inactive_set_1[l2u2 & (x1_cand == 0.) & ((l2 == x2_cand) | (x2_cand == u2)) & pred_higher_in_1] = True
    inactive_set_2[l2u2 & (x1_cand == 0.) & ((l2 == x2_cand) | (x2_cand == u2)) & pred_higher_in_2] = True

    assert (inactive_set_1 | inactive_set_2).all()
    assert not (inactive_set_1 & inactive_set_2).any()
    return x1_cand, x2_cand, inactive_set_1, inactive_set_2


def compute_lpcc_x1x2_cand_cc(x1, x2, g1, g2, l1, u1, l2, u2, l1u1, l2u2, rho):
    x1_cand = copy.deepcopy(x1)
    x2_cand = copy.deepcopy(x2)

    pred_table = np.full((x1_cand.shape[0], 9), np.NINF)

    x1_pos_to_0, x1_neg_to_0 = np.minimum(x1 + rho, 0.), np.maximum(x1 - rho, 0.)
    x1_pos_to_u, x1_neg_to_l = np.minimum(x1 + rho, u1), np.maximum(x1 - rho, l1)
    d1_pos_to_0, d1_neg_to_0 = x1_pos_to_0 - x1, x1_neg_to_0 - x1
    d1_pos_to_u, d1_neg_to_l = x1_pos_to_u - x1, x1_neg_to_l - x1

    x2_pos_to_0, x2_neg_to_0 = np.minimum(x2 + rho, 0.), np.maximum(x2 - rho, 0.)
    x2_pos_to_u, x2_neg_to_l = np.minimum(x2 + rho, u2), np.maximum(x2 - rho, l2)
    d2_pos_to_0, d2_neg_to_0 = x2_pos_to_0 - x2, x2_neg_to_0 - x2
    d2_pos_to_u, d2_neg_to_l = x2_pos_to_u - x2, x2_neg_to_l - x2

    # Case -Inf < l1 < x1 < u1 < Inf
    cond_l1u1_0 = l1u1 & (x2 < 0.)
    cond_l1u1_1 = l1u1 & (x2 < 0.) & (x2_pos_to_u >= 0.)
    cond_l1u1_2 = l1u1 & (  ((x2 < 0.) & (x2_pos_to_u >= 0.) & (x1_neg_to_l == l1))
                          | ((x2 == 0.) & (x1_neg_to_l == l1)))
    cond_l1u1_345 = l1u1 & (x2 == 0.)                     
    cond_l1u1_6 = l1u1 & (  ((x2 > 0.) & (x2_neg_to_l <= 0.) & (x1_pos_to_u == u1))
                          | ((x2 == 0.) & (x1_pos_to_u == u1)))
    cond_l1u1_7 = l1u1 & (x2 > 0.) & (x2_neg_to_l <= 0.)
    cond_l1u1_8 = l1u1 & (x2 > 0.)
    pred_table[cond_l1u1_0,   0] = -g2[cond_l1u1_0] * d2_pos_to_0[cond_l1u1_0]
    pred_table[cond_l1u1_1,   1] = -g1[cond_l1u1_1] * d1_neg_to_l[cond_l1u1_1] - g2[cond_l1u1_1] * d2_pos_to_0[cond_l1u1_1]
    pred_table[cond_l1u1_2,   2] = -g1[cond_l1u1_2] * d1_neg_to_l[cond_l1u1_2] - g2[cond_l1u1_2] * d2_pos_to_u[cond_l1u1_2]
    pred_table[cond_l1u1_345, 3] = -g1[cond_l1u1_345] * d1_neg_to_l[cond_l1u1_345]
    pred_table[cond_l1u1_345, 4] = 0.
    pred_table[cond_l1u1_345, 5] = -g1[cond_l1u1_345] * d1_pos_to_u[cond_l1u1_345]
    pred_table[cond_l1u1_6,   6] = -g1[cond_l1u1_6] * d1_pos_to_u[cond_l1u1_6] - g2[cond_l1u1_6] * d2_neg_to_l[cond_l1u1_6]
    pred_table[cond_l1u1_7,   7] = -g1[cond_l1u1_7] * d1_pos_to_u[cond_l1u1_7] - g2[cond_l1u1_7] * d2_neg_to_0[cond_l1u1_7]
    pred_table[cond_l1u1_8,   8] = -g2[cond_l1u1_8] * d2_neg_to_0[cond_l1u1_8]

    # Case -Inf < l2 < x2 < u2 < Inf
    cond_l2u2_0 = l2u2 & (x1 < 0.)
    cond_l2u2_1 = l2u2 & (x1 < 0.) & (x1_pos_to_u >= 0.)
    cond_l2u2_2 = l2u2 & (((x1 < 0.) & (x1_pos_to_u >= 0.) & (x2_neg_to_l == l2))
                          | ((x1 == 0.) & (x2_neg_to_l == l2)))
    cond_l2u2_345 = l2u2 & (x1 == 0.)
    cond_l2u2_6 = l2u2 & (((x1 > 0.) & (x1_neg_to_l <= 0.) & (x2_pos_to_u == u2))
                          | ((x1 == 0.) & (x2_pos_to_u == u2)))
    cond_l2u2_7 = l2u2 & (x1 > 0.) & (x1_neg_to_l <= 0.)
    cond_l2u2_8 = l2u2 & (x1 > 0.)
    pred_table[cond_l2u2_0,   0] = -g1[cond_l2u2_0] * d1_pos_to_0[cond_l2u2_0]
    pred_table[cond_l2u2_1,   1] = -g1[cond_l2u2_1] * d1_pos_to_0[cond_l2u2_1] - g2[cond_l2u2_1] * d2_neg_to_l[cond_l2u2_1]
    pred_table[cond_l2u2_2,   2] = -g1[cond_l2u2_2] * d1_pos_to_u[cond_l2u2_2] - g2[cond_l2u2_2] * d2_neg_to_l[cond_l2u2_2]
    pred_table[cond_l2u2_345, 3] = -g2[cond_l2u2_345] * d2_neg_to_l[cond_l2u2_345]
    pred_table[cond_l2u2_345, 4] = 0.
    pred_table[cond_l2u2_345, 5] = -g2[cond_l2u2_345] * d2_pos_to_u[cond_l2u2_345]
    pred_table[cond_l2u2_6,   6] = -g1[cond_l2u2_6] * d1_neg_to_l[cond_l2u2_6] - g2[cond_l2u2_6] * d2_pos_to_u[cond_l2u2_6]
    pred_table[cond_l2u2_7,   7] = -g1[cond_l2u2_7] * d1_neg_to_0[cond_l2u2_7] - g2[cond_l2u2_7] * d2_pos_to_u[cond_l2u2_7]
    pred_table[cond_l2u2_8,   8] = -g1[cond_l2u2_8] * d1_neg_to_0[cond_l2u2_8]

    amax_pred = np.argmax(pred_table, axis=1)
    assert (pred_table[l1u1 | l2u2,amax_pred[l1u1 | l2u2]] >= 0.).all()

    mask_x1_pos_to_u = (l1u1 & ((amax_pred == 5) | (amax_pred == 6) | (amax_pred == 7))) \
                       | (l2u2 & (amax_pred == 2))
    mask_x1_pos_to_0 = l2u2 & ((amax_pred == 1) | (amax_pred == 0))
    mask_x1_neg_to_0 = l2u2 & ((amax_pred == 7) | (amax_pred == 8))
    mask_x1_neg_to_l = (l1u1 & ((amax_pred == 3) | (amax_pred == 2) | (amax_pred == 1))) \
                       | (l2u2 & (amax_pred == 6))
    
    mask_x2_pos_to_u = (l1u1 & (amax_pred == 2)) \
                       | (l2u2 & ((amax_pred == 5) | (amax_pred == 6) | (amax_pred == 7)))
    mask_x2_pos_to_0 = l1u1 & ((amax_pred == 1) | (amax_pred == 0))                       
    mask_x2_neg_to_0 = l1u1 & ((amax_pred == 7) | (amax_pred == 8))
    mask_x2_neg_to_l = (l1u1 & (amax_pred == 6)) \
                       | (l2u2 & ((amax_pred == 3) | (amax_pred == 2) | (amax_pred == 1)))

    x1_cand[mask_x1_pos_to_u] = x1_pos_to_u[mask_x1_pos_to_u]
    x1_cand[mask_x1_pos_to_0] = x1_pos_to_0[mask_x1_pos_to_0]
    x1_cand[mask_x1_neg_to_0] = x1_neg_to_0[mask_x1_neg_to_0]
    x1_cand[mask_x1_neg_to_l] = x1_neg_to_l[mask_x1_neg_to_l]
    x2_cand[mask_x2_pos_to_u] = x2_pos_to_u[mask_x2_pos_to_u]
    x2_cand[mask_x2_pos_to_0] = x2_pos_to_0[mask_x2_pos_to_0]
    x2_cand[mask_x2_neg_to_0] = x2_neg_to_0[mask_x2_neg_to_0]
    x2_cand[mask_x2_neg_to_l] = x2_neg_to_l[mask_x2_neg_to_l]
    
    return x1_cand[l1u1 | l2u2], x2_cand[l1u1 | l2u2]


def project_to_complementarity(x_init, x1_init, x2_init, l1, u1, l2, u2, mask_n1, mask_n2, cc_code):
    # Case $\ell_1 \le x_1 \perp x_2 \ge \ell_2$
    l1l2 = cc_code[CCCode.L1_FINITE] & cc_code[CCCode.L2_FINITE]
    il1l2_tol1 = ((x1_init - l1) <= (x2_init - l2)) & l1l2
    il1l2_tol2 = ((x1_init - l1) > (x2_init - l2)) & l1l2
    x1_init[il1l2_tol1] = l1[il1l2_tol1]
    x2_init[il1l2_tol2] = l2[il1l2_tol2]

    # Case $\ell_1 \le x_1 \perp u_2 \ge x_2$
    l1u2 = cc_code[CCCode.L1_FINITE] & cc_code[CCCode.U2_FINITE]
    il1u2_tol1 = ((x1_init - l1) <= (u2 - x2_init)) & l1u2
    il1u2_tou2 = ((x1_init - l1) > (u2 - x2_init)) & l1u2
    x1_init[il1u2_tol1] = l1[il1u2_tol1]
    x2_init[il1u2_tou2] = u2[il1u2_tou2]

    # Case $x_1 \le u_1 \perp u_2 \ge x_2$
    u1u2 = cc_code[CCCode.U1_FINITE] & cc_code[CCCode.U2_FINITE]
    iu1u2_tol1 = ((u1 - x1_init) <= (u2 - x2_init)) & u1u2
    iu1u2_tou2 = ((u1 - x1_init) > (u2 - x2_init)) & u1u2
    x1_init[iu1u2_tol1] = u1[iu1u2_tol1]
    x2_init[iu1u2_tou2] = u2[iu1u2_tou2]

    # Case $x_1 \le u_1 \perp x_2 \ge \ell_2$
    u1l2 = cc_code[CCCode.U1_FINITE] & cc_code[CCCode.L2_FINITE]
    iu1l2_tou1 = ((u1 - x1_init) <= (x2_init - l2)) & u1l2
    iu1l2_tol2 = ((u1 - x1_init) > (x2_init - l2)) & u1l2
    x1_init[iu1l2_tou1] = u1[iu1l2_tou1]
    x2_init[iu1l2_tol2] = l2[iu1l2_tol2]

    # Case $\ell_1 \le x_1 \le u_1 \perp x_2$
    l1u1 = cc_code[CCCode.L1_FINITE] & cc_code[CCCode.U1_FINITE]
    il1u1_to02_a = l1u1 & (x2_init < 0.) & (-x2_init < (u1 - x_init[mask_n1]))
    il1u1_to02_b = l1u1 & (x2_init > 0.) & (x2_init < (x_init[mask_n1] - l1))
    il1u1_to02 = il1u1_to02_a | il1u1_to02_b
    il1u1_tol1 = l1u1 & (x2_init > 0.) & (~il1u1_to02)
    il1u1_tou1 = l1u1 & (x2_init < 0.) & (~il1u1_to02)
    x1_init[il1u1_tol1] = l1[il1u1_tol1]
    x1_init[il1u1_tou1] = l1[il1u1_tou1]
    x2_init[il1u1_to02] = np.zeros(np.sum(il1u1_to02))

    # Case $x_1 \perp \ell_2 \le x_2 \le u_2$
    l2u2 = cc_code[CCCode.L2_FINITE] & cc_code[CCCode.U2_FINITE]
    il2u2_to01_a = l2u2 & (x1_init < 0.) & (-x1_init < (u2 - x_init[mask_n2]))
    il2u2_to01_b = l2u2 & (x1_init > 0.) & (x1_init < (x_init[mask_n2] - l2))
    il2u2_to01 = il2u2_to01_a | il2u2_to01_b
    il2u2_tol2 = l2u2 & (x1_init > 0.) & (~il2u2_to01)
    il2u2_tou2 = l2u2 & (x1_init < 0.) & (~il2u2_to01)
    x1_init[il2u2_to01] = np.zeros(np.sum(il2u2_to01))
    x2_init[il2u2_tol2] = l2[il2u2_tol2]
    x2_init[il2u2_tou2] = u2[il2u2_tou2]


def project_to_bounds(x_init, l0, u0, l1, u1, l2, u2, mask_n0, mask_n1, mask_n2):
    x0_init = np.maximum(np.minimum(x_init[mask_n0], u0), l0)
    x1_init = np.maximum(np.minimum(x_init[mask_n1], u1), l1)
    x2_init = np.maximum(np.minimum(x_init[mask_n2], u2), l2)
    return x0_init, x1_init, x2_init


def x_feasible(x, mask_n0, mask_n1, mask_n2, l0, u0, l1, u1, l2, u2, cc_code):
    x0, x1, x2 = x[mask_n0], x[mask_n1], x[mask_n2]
    x0_feasible = (l0 <= x0).all() and (x0 <= u0).all()
    x1_feasible = (l1 <= x1).all() and (x1 <= u1).all()
    x2_feasible = (l2 <= x2).all() and (x2 <= u2).all()

    cc_feasible_arr = np.zeros(l1.size, dtype=bool)
    l1l2 = cc_code[CCCode.L1_FINITE] & cc_code[CCCode.L2_FINITE]
    l1u2 = cc_code[CCCode.L1_FINITE] & cc_code[CCCode.U2_FINITE]
    u1u2 = cc_code[CCCode.U1_FINITE] & cc_code[CCCode.U2_FINITE]
    u1l2 = cc_code[CCCode.U1_FINITE] & cc_code[CCCode.L2_FINITE]

    cc_feasible_arr[l1l2] = ((x1[l1l2] - l1[l1l2]) * (x2[l1l2] - l2[l1l2])) == np.zeros(l1l2.sum())
    cc_feasible_arr[l1u2] = ((x1[l1u2] - l1[l1u2]) * (u2[l1u2] - x2[l1u2])) == np.zeros(l1u2.sum())
    cc_feasible_arr[u1u2] = ((u1[u1u2] - x1[u1u2]) * (u2[u1u2] - x2[u1u2])) == np.zeros(u1u2.sum())
    cc_feasible_arr[u1l2] = ((u1[u1l2] - x1[u1l2]) * (x2[u1l2] - l2[u1l2])) == np.zeros(u1l2.sum())

    l1u1 = cc_code[CCCode.L1_FINITE] & cc_code[CCCode.U1_FINITE]
    l2u2 = cc_code[CCCode.L2_FINITE] & cc_code[CCCode.U2_FINITE]
    cc_feasible_arr[l1u1] = \
        np.maximum(x2[l1u1], np.zeros(l1u1.sum())) * (x1[l1u1] - l1[l1u1]) \
        + np.minimum(x2[l1u1], np.zeros(l1u1.sum())) * (u1[l1u1] - x1[l1u1]) \
        == np.zeros(l1u1.sum())
    cc_feasible_arr[l2u2] = \
        np.maximum(x1[l2u2], np.zeros(l2u2.sum())) * (x2[l2u2] - l2[l2u2]) \
        + np.minimum(x1[l2u2], np.zeros(l2u2.sum())) * (u2[l2u2] - x2[l2u2]) \
        == np.zeros(l2u2.sum())

    cc_feasible = cc_feasible_arr.all()
    return x0_feasible and x1_feasible and x2_feasible and cc_feasible


def check_bound_consistency(l0, u0, l1, u1, l2, u2, cc_code):
    assert not np.isin(np.array([np.NINF, np.Inf]), l0).any()
    assert not np.isin(np.array([np.NINF, np.Inf]), u0).any()
    assert (l0 <= u0).all()
    assert (l1 < np.Inf).all()
    assert (np.NINF < u1).all()
    assert (l2 < np.Inf).all()
    assert (np.NINF < u2).all()
    assert (l1 < u1).all()
    assert (l2 < u2).all()
    # For each complementarity variable pair $(x_{1,i}, x_{2,i})$
    # exactly two of the four bounds $(l_{1,i}, u_{1,i}, l_{2,i}, u_{2,i})$ are finite.
    assert np.array_equal(np.sum(cc_code, axis=0), np.full(cc_code.shape[1], 2.))


def check_shapes(n0, n12, nx, x_init, l0, u0, l1, u1, l2, u2, l12, u12):
    assert n0 + 2 * n12 == nx
    assert x_init.size == nx
    assert x_init.shape == (nx,)
    assert l0.size == n0
    assert l0.shape == (n0,)
    assert u0.size == n0
    assert u0.shape == (n0,)
    assert l1.size == n12
    assert l1.shape == (n12,)
    assert u1.size == n12
    assert u1.shape == (n12,)
    assert l2.size == n12
    assert l2.shape == (n12,)
    assert u2.size == n12
    assert u2.shape == (n12,)
    assert l12.size == 2 * n12
    assert l12.shape == (2 * n12,)
    assert u12.size == 2 * n12
    assert u12.shape == (2 * n12,)


def get_bounds(l, u, mask_n0, mask_n1, mask_n2, mask_n12):
    l0, u0 = l[mask_n0], u[mask_n0]
    l1, u1 = l[mask_n1], u[mask_n1]
    l2, u2 = l[mask_n2], u[mask_n2]
    l12, u12 = l[mask_n12], u[mask_n12]
    return l0, u0, l1, u1, l2, u2, l12, u12

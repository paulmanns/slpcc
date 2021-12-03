# distutils: language = c++
import cython

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libcpp.algorithm cimport copy as std_copy


'''!@brief Solves a quadratic along a piecewise linear path. The path is determined by following the coordinates,
        where greedy choices are made for complementary coordinates to determine a branch problem.
        The first local minimizer of the
        quadratic along the piecewise linear path determined by this branch problem is determined.

@param G 2d numpy.array, the square problem matrix 
@param d 1d numpy.array, the problem vector
@param x 1d numpy.array, starting vector 
@param g 1d numpy.array, negative of the direction vector along which the piecewise path is pursued (gradient).
@param mask_n0 1d numpy.array of booleans, True in the coordinates of $x$ that corresponds to a coordinate in $x_0$
    else False
@param mask_n12 1d numpy.array of booleans, True in the coordinates of $x$ that corresponds to a coordinate in $x_1$
    or $x_2$ else False
@param x_l, 1d numpy.array, lower bounds for the entries in $x$
@param x_u, 1d numpy.array, upper bounds for the entries in $x$
@return tuple of 1d numpy.array ($x$ at local minimizer), vector[int] (the set of modifiable coordinates the path's start
    (the path is uniquely determined from mc))
'''
def solve_qpwl(
        np.ndarray G,
        np.ndarray d,
        np.ndarray x,
        np.ndarray g,
        np.ndarray mask_n0,
        np.ndarray mask_n12,
        np.ndarray x_l,
        np.ndarray x_u,
        np.ndarray cc_code,
        gtol
        ):
    cdef np.ndarray times_to_bounds = compute_times_to_bounds(x, g, x_l, x_u)
    cdef vector[int] mc = compute_greedy_modifiable_coordinates(x, g, mask_n0, mask_n12, x_l, x_u, gtol)
    cdef np.ndarray x_qpwl = np.copy(x)

    if mc.size() > 0:
        x_segs, mc_segs, t_segs = compute_pwlpath(x, g, mc, times_to_bounds, x_l, x_u, mask_n0, mask_n12)
        # print('----------------\nx_segs')
        # print(x_segs)
        # print('\n',np.sum(mask_n0),np.sum(mask_n12) // 2)
        # print('\nmc_segs')
        # for mcc in mc_segs:
        #     print(mcc)
        #     print(g[mcc])
        x_qpwl, q_qpwl, t_qpwl = \
            lpec_qpwl_linesearch(G, d, x_segs, mc_segs, t_segs, g, np.Inf, x_l, x_u)
        # print(t_qpwl)

    cdef long n12 = np.sum(mask_n12) // 2
    ias0 = (x_l[mask_n0] < x_qpwl[mask_n0]) & (x_qpwl[mask_n0] < x_u[mask_n0])
    ias1 = np.full(n12, False)
    ias2 = np.full(n12, False)

    cdef np.ndarray[long] cands_n12 = np.sort(np.where(mask_n12 == True)[0].astype(np.int64))
    for i in range(n12):
        if x_qpwl[cands_n12[i]] > 0. and x_qpwl[cands_n12[i + n12]] == 0.:
            ias1[i] = True
        elif x_qpwl[cands_n12[i + n12]] > 0. and x_qpwl[cands_n12[i]] == 0.:
            ias2[i] = True
        elif g[cands_n12[i]] < g[cands_n12[i + n12]]:
            ias1[i] = True
        else:
            ias2[i] = True
    assert (ias1 | ias2).all()
    assert not (ias1 & ias2).any()
    return x_qpwl, mc, ias0, ias1, ias2


'''!@param For each entry $x_i$ in $x$, the (non-negative) time to bound is defined as the ratio of the distance
        between $x_i$ and the bound in the direction of $-g_i$ and $|g_i|$. This time determines the length of
        the piecewise path from the point on where the path starts altering $x$ in this this coordinate until it
        stops altering $x$ is in this coordinate. Therefore, the time is set to zero if $g_i == 0$ because such
        coordinates can not be altered.

@param x 1d numpy.array, input vector $(x_{11},\ldots,x_{1n},x_{21},\ldots,x_{2n})^T$
@param g 1d numpy.array, negative of the direction vector along which the piecewise path is pursued
@param x_l, 1d numpy.array, lower bounds for the entries in $x$
@param x_u, 1d numpy.array, upper bounds for the entries in $x$
@return 1d numpy.array, containing the time as defined above for all entries in $x$ / $g$
'''
cdef np.ndarray compute_times_to_bounds(
        np.ndarray x,
        np.ndarray g,
        np.ndarray x_l,
        np.ndarray x_u
        ):
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    cdef np.ndarray time_to_l = (x - x_l) / g
    cdef np.ndarray time_to_u = (x - x_u) / g
    cdef np.ndarray[double, ndim=1] time = np.empty(x.shape[0])
    time[g > 0.] = time_to_l[g > 0.]
    time[g < 0.] = time_to_u[g < 0.]
    time[g == 0.] = 0.
    np.seterr(**old_settings)
    return time


cdef vector[int] compute_greedy_modifiable_coordinates(
        np.ndarray x,
        np.ndarray g,
        np.ndarray mask_n0,
        np.ndarray mask_n12,
        np.ndarray x_l,
        np.ndarray x_u,
        float gtol # no use so far
        ):
    cdef vector[int] mc = vector[int]()
    cdef np.ndarray[long] cands_n0 = np.sort(np.where(mask_n0 == True)[0].astype(np.int64))
    cdef np.ndarray[long] cands_n12 = np.sort(np.where(mask_n12 == True)[0].astype(np.int64))
    n12 = cands_n12.size // 2

    for i in cands_n0:
        if g[i] > 0. and (not np.isclose(g[i], 0.)) and (not np.isclose(x[i], x_l[i])):
            mc.push_back(i)
        elif g[i] < 0. and (not np.isclose(g[i], 0.)) and (not np.isclose(x[i], x_u[i])):
            mc.push_back(i)

    cdef int i1 = -1
    cdef int i2 = -1
    for j in range(n12):
        i1, i2 = cands_n12[j], cands_n12[j + n12]
        if x[i1] > 0. and (not np.isclose(g[i1], 0.)) and (not (np.isclose(x[i1], 0.) and g[i1] > 0.)):
            mc.push_back(i1)
        elif x[i2] > 0. and (not np.isclose(g[i2], 0.)) and (not (np.isclose(x[i2], 0.) and g[i2] > 0.)):
            mc.push_back(i2)
        elif x[i1] == 0. and x[i2] == 0.:
            if g[i1] < 0. and g[i1] <= g[i2] and (not np.isclose(g[i1], 0.)):
                mc.push_back(i1)
            elif g[i2] < 0. and g[i2] <= g[i1] and (not np.isclose(g[i2], 0.)):
                mc.push_back(i2)
    return mc


'''!@brief Computes a piecewise linear path from the set of modifiable coordinates mc, a starting point $x$ and the direction
        vector $g$. The path is pursued along the entries in $-g$ (negative gradient).
        $x = (x_0^T, x_1^T, x_2^T)^T$ has lower and upper bounds on each entry and the complementarity constraint $0 \le x_1 \perp x_2 \ge 0$.
        $-g$ is followed along the coordinates in mc until a bound is reached. Then, the coordinate reaching the bound is thrown out of mc.
        If the reached bound was a lower bound being equal to $0$ which is also part of at a kink of the complementarity
        condition where the complementary coordinate has a direction of descent (the respective entry in $g$ is negative),
        the complementary coordinate is added to mc. This procedure is repeated if no alterable
        coordinates are left. Consequently, the last segment may have a length of $\infty$ (if no upper bounds are given by the caller
        on the complementary coordinates.)

@param x 1d numpy.array, starting vector of the piecewise linear path
@param g 1d numpy.array, negative of the direction vector along which the piecewise path is pursued
@param mc 1d vector[int], set of modifiable coordinates 
@param times_to_bounds 1d numpy.array, the entry i contains the virtual length (time) of the path from x[i] to the bound in
    the direction -g[i], i.e. the length divided by -g[i], may take the value Inf if the bound is at +-Inf. It is expected
    that times_to_bounds[i] == 0 if g[i] == 0.
@param x_l, 1d numpy.array, lower bounds for the entries in $x$
@param x_u, 1d numpy.array, upper bounds for the entries in $x$
@param mask_n0 1d numpy.array of booleans, True in the coordinates of $x$ that corresponds to a coordinate in $x_0$ else False
@param mask_n12 1d numpy.array of booleans, True in the coordinates of $x$ that corresponds to a coordinate in $x_1$ or $x_2$ else False
@return tuple of 2d nump.array (starting points of segments and endpoint), list of numpy.array (alterable coordinates),
    list of time points (starting points of segments and endpoints)
'''
cdef compute_pwlpath(
        np.ndarray x,
        np.ndarray g,
        vector[int] mc,
        np.ndarray[double] times_to_bounds, 
        np.ndarray x_l,
        np.ndarray x_u,
        np.ndarray mask_n0,
        np.ndarray mask_n12
        ):
    cdef vector[int] mc_ = vector[int](mc.size())
    std_copy(mc.begin(), mc.end(), mc_.begin())

    ## TODO: What shall happen if several indices arrive at the kinks simultaneously? 
    ##       Right now, an segment of length $0$ is generated.
    cdef int n0 = np.sum(mask_n0)
    cdef int n12 = np.sum(mask_n12) // 2
    cdef int n = n0 + 2*n12

    cdef np.ndarray[long] cands_n12 = np.sort(np.where(mask_n12 == True)[0].astype(np.int64))
    
    cdef np.ndarray x_bps = np.tile(x, (2*mc.size() + 1, 1))
    cdef vector[double] t_bps = vector[double]()
    cdef vector[vector[int]] mc_bps = vector[vector[int]]()

    mc_bps.push_back(mc_)

    cdef np.ndarray[double] ti = times_to_bounds[mc_]
    cdef double t_start = 0.
    cdef int i = 0
    cdef int idx_out_in_mc = -1
    # fail = False
    while mc_.size() > 0 and np.min(ti) < np.Inf:
        mc_ = vector[int](mc_bps[i].size())
        std_copy(mc_bps[i].begin(), mc_bps[i].end(), mc_.begin())

        idx_out_in_mc = np.argmin(ti)
        t_end = ti[idx_out_in_mc]
        mc_out = mc_[idx_out_in_mc]
        idx_out_in_cands_n12 = np.searchsorted(cands_n12, mc_out)

        if idx_out_in_cands_n12 < n12:
            mc_in_cand = cands_n12[idx_out_in_cands_n12 + n12]
        elif idx_out_in_cands_n12 < 2 * n12:
            mc_in_cand = cands_n12[idx_out_in_cands_n12 - n12]
        else: # not found in cands_n12 => mask_n0[mc_out] must be true.
            assert mask_n0[mc_out]

        if np.isclose(t_start, t_end):
            if g[mc_out] > 0.:
                x_bps[i, mc_out] = x_l[mc_out]
            else: # g[mc_out] < 0. ; g[mc_out] cannot be close to zero if the coordinates in mc were inserted correctly
                x_bps[i, mc_out] = x_u[mc_out]
            if mask_n0[mc_out] or g[mc_out] < 0. or x_l[mc_out] > 0. or g[mc_in_cand] > 0. or np.isclose(g[mc_in_cand], 0.) or x_u[mc_in_cand] == 0.:
                mc_.erase(mc_.begin() + idx_out_in_mc)
            else:
                mc_[idx_out_in_mc] = mc_in_cand
                times_to_bounds[mc_in_cand] += t_start
            mc_bps[i] = mc_
            ti = times_to_bounds[mc_]
            # fail = True
        else:
            x_bps[i + 1] = x_bps[i]
            x_bps[i + 1, mc_] -= (t_end - t_start) * g[mc_]
            if g[mc_out] > 0.:
                x_bps[i + 1, mc_out] = x_l[mc_out]
            else: # g[mc_out] < 0. ; g[mc_out] cannot be close to zero if the coordinates in mc were inserted correctly
                x_bps[i + 1, mc_out] = x_u[mc_out]
            
            if mask_n0[mc_out] or g[mc_out] < 0. or x_l[mc_out] > 0. or g[mc_in_cand] > 0. or np.isclose(g[mc_in_cand], 0.) or x_u[mc_in_cand] == 0.:
                mc_.erase(mc_.begin() + idx_out_in_mc)
            else:
                mc_[idx_out_in_mc] = mc_in_cand
                times_to_bounds[mc_in_cand] += t_start

            ti = times_to_bounds[mc_]
            mc_bps.push_back(mc_)
            t_bps.push_back(t_end)
            t_start = t_end
            i += 1

    t_bps.push_back(np.Inf)
    x_bps = x_bps[:i+1,:]

    # print()
    # print(x_bps)
    # print(mc)
    # print(mc_bps)
    # print(t_bps)
    # print()
    # if fail:
    #     assert False

    return x_bps, mc_bps, t_bps


'''!@brief Search for first local minimizers on projections $\min_x \frac{1}{2} x^T G x + x^T d$ subject to
        $0 \le x_1 \perp x_2 \ge 0$ onto the piecewise path given by starting vectors, alterable coordinates
        and segment lengths.

@param G 2d numpy.array, the square problem matrix 
@param d 1d numpy.array, the problem vector
@param num_pieces scalar, number of segments
@param x_bps 2d numpy.array, rows are segments, columns are starting vectors of the segment
@param ac_bps list of 1d numpy.array, numpy.array alterable indices per segment
@param t_bps 1d numpy.array, lengths of the path along the segments (length until $i$-th segment)
@param g 1d numpy.array, direction vector along which the path is pursued
@param t_max scalar, maximum length of the path
@return 2d numpy.array containing the minimizers of the segments, 1d numpy.array containing the 
    corresponding minimal objective values and 1d numpy.array containing the corresponding
    values of $t$
'''
cdef lpec_qpwl_linesearch(
        np.ndarray G,
        np.ndarray d,
        np.ndarray x_bps,
        vector[vector[int]] mc_bps,
        vector[double] t_bps,
        np.ndarray g,
        double t_max,
        np.ndarray x_l,
        np.ndarray x_u
        ):
    cdef unsigned long num_pieces = x_bps.shape[0]
    assert num_pieces == mc_bps.size()
    assert num_pieces == t_bps.size()
    
    cdef unsigned long i_max = np.searchsorted(t_bps, t_max, side='left')
    if mc_bps.back().size() == 0:
        i_max -= 1
    assert i_max + 1 <= num_pieces

    cdef np.ndarray x_start = np.empty(g.shape[0])
    cdef vector[int] mc = mc_bps[0]
    cdef double dt = 0.

    cdef np.ndarray x_prev = np.empty(g.shape[0])
    cdef double t_prev = np.Inf
    cdef double f_prev = np.Inf

    cdef np.ndarray x_cand = np.empty(g.shape[0])
    cdef double f_cand = np.Inf
    cdef double dt_cand = 0.
    old_settings = np.seterr(invalid='ignore')
    for i in range(i_max + 1):
        x_start = x_bps[i]
        mc = mc_bps[i]
        t_start = t_bps[i - 1] if i > 0 else 0.
        t_end = t_max if i == i_max and i_max == t_bps.size() else t_bps[i]
        dt = t_end - t_start

        x_cand, f_cand, dt_cand = minimize_1d_qseg(G, d, x_start, g, dt, mc, x_l, x_u)

        #print('It. %3u ----------------------' % i)
        if f_cand < f_prev:
            f_prev = f_cand
            x_prev = np.copy(x_cand)
            t_prev = t_start + dt_cand
        else:
            break
    np.seterr(**old_settings)
    return x_prev, f_prev, t_prev


'''!@brief Optimization of $\min_t \frac{1}{2} z^T G z + z^T d$ subject to
        $t \in [0,\Delta t]$, $z = x + t \tilde{g}$ with $\tilde{g}_i = g_i$ for $i \in \text{mc}$ and
        $\tilde{g}_i = 0$ else.
    
@param G 2d numpy.array, the square problem matrix 
@param d 1d numpy.array, the problem vector
@param x 1d numpy.array, starting vector 
@param g 1d numpy.array, direction vector 
@param dt scalar, $\Delta t$ that determines the upper bound on $t$
@return minimizing vector $z$ as numpy.array, objective value $f(x)$ as scalar
    and $t$ corresponding to $z$ as scalar
'''
@cython.cdivision(True)
cdef minimize_1d_qseg(
        np.ndarray G,
        np.ndarray d,
        np.ndarray x,
        np.ndarray g,
        double dt,
        vector[int] mc,
        np.ndarray x_l,
        np.ndarray x_u
        ):
    # setup quadratic minimization $q(t) = f^{(0)} + f^{(1)} t + \frac{1}{2} f^{(2)} t^2$
    cdef double f0 = d.dot(x) + .5 * x.dot(G.dot(x))
    cdef double f1 = d[mc].dot(-g[mc]) + .5 * x.dot(G[:,mc].dot(-g[mc])) + .5 * (-g[mc]).dot(G[mc,:].dot(x))
    cdef double f2 = (-g[mc]).dot(  (G[:, mc].dot(-g[mc]))[mc] )
    cdef np.ndarray x_res = np.copy(x)
    cdef double f_res = 0.
    cdef double t_res = 0.
    cdef double t_star = -f1 / f2
    t_res = t_star
    if np.isnan(t_star):
        if f1 != 0. or f2 != 0.:
            print('CASE f1 != 0 or f2 != 0 in NAN case of computation of t_star.')
            print(g[mc], f1, f2)
        assert f1 == 0.
        assert f2 == 0.
        t_res = 0.
    elif t_star < 0. or t_star >= dt or f2 <= 0.:
        t_res = 0. if f1 > 0. else dt
    
    x_res[mc] = x[mc] - t_res * g[mc]

    for j in mc:
        if np.isclose(x_res[j] - x_l[j], 0., atol=1e-13):
            x_res[j] = x_l[j]
        if np.isclose(x_res[j] - x_u[j], 0., atol=1e-13):
            x_res[j] = x_u[j]
    
    if t_res == np.Inf and f1 != 0. and f2 != 0. and np.sign(f1) != np.sign(f2):
        f_res = np.sign(f2) * np.Inf        
    else:
        f_res = f0 + (0. if f1 == 0. else f1 * t_res) + (0. if f2 == 0. else .5 * f2 * t_res**2)
    return x_res, f_res, t_res

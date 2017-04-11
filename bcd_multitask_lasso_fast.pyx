# Author: Eugene Ndiaye
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# GAP Safe Screening Rules for Sparse-Group Lasso.
# http://arxiv.org/abs/1602.06225
# firstname.lastname@telecom-paristech.fr


from libc.math cimport fabs, sqrt
from libc.stdlib cimport qsort
from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dscal
import numpy as np
cimport numpy as np
cimport cython

cdef int NO_SCREEN = 0
cdef int DGST3 = 1
cdef int GAPSAFE_SEQ = 2
cdef int GAPSAFE = 3
# cdef int GAPSAFE_SEQ_pp = 4
# cdef int GAPSAFE_pp = 5
cdef int STRONG_RULE = 10
cdef int STRONG_GAP_SAFE = 666


cdef inline double fmax(double x, double y) nogil:
    if x > y:
        return x
    return y


cdef inline double fsign(double f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


cdef double abs_max(int n, double * a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef double m = fabs(a[0])
    cdef double d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef double max(int n, double * a) nogil:
    """np.max(a)"""
    cdef int i
    cdef double m = a[0]
    cdef double d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double primal_value(int n_samples, int n_features, int n_tasks,
                         double[:, ::1] beta, double norm_res2,
                         double lambda_, int * disabled_features) nogil:

    cdef int j = 0
    cdef int inc = 1
    cdef double fval = 0.
    cdef double l1l2_norm = 0.

    # l21_norm = np.sqrt(np.sum(W ** 2, axis=0)).sum()
    for j in range(n_features):

        if disabled_features[j] == 1:
            continue
        # np.sqrt(np.sum(W ** 2, axis=0))
        l1l2_norm += dnrm2(& n_tasks, & beta[j, 0], & inc)

    fval = 0.5 * norm_res2 + lambda_ * l1l2_norm

    return fval


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dual(int n_samples, int n_tasks,
                 double[::1, :] residual, double[::1, :] y, double norm_res2,
                 double dual_scale, double lambda_) nogil:

    cdef int k = 0
    cdef int inc = 1
    cdef double Ry = 0.
    cdef double alpha = lambda_ / dual_scale

    # Ry = np.dot(residual, y)
    for k in range(n_tasks):
        Ry += ddot(& n_samples, & residual[0, k], & inc, & y[0, k], & inc)

    dval = Ry * alpha - 0.5 * norm_res2 * alpha ** 2

    return dval


cdef double dual_gap(int n_samples, int n_features, int n_tasks,
                     double[:, ::1] beta, double[::1, :] residual,
                     double[::1, :] y, double norm_res2, double dual_scale,
                     double lambda_, int * disabled_features) nogil:

    cdef double pobj = primal_value(n_samples, n_features, n_tasks,
                                    beta, norm_res2, lambda_, & disabled_features[0])

    cdef double dobj = dual(n_samples, n_tasks, residual, y,
                            norm_res2, dual_scale, lambda_)

    cdef double gap_ = pobj - dobj
    return gap_


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void screen_variable(int n_samples, int n_features, int n_tasks,
                          int * n_active_features, double[::1, :] X,
                          double[:, ::1] beta, double[::1, :] residual,
                          double[:, ::1] XTR,
                          double * norm2_X, double * norm_row_XTcenter,
                          double * norm_row_XTR, double dual_scale,
                          int * disabled_features, double r_screen,
                          int screening) nogil:

    cdef int j = 0
    cdef int k = 0
    cdef int inc = 1
    cdef double r_normX_j = 0.
    cdef int screen_test = 0

    for j in range(n_features):

        if disabled_features[j] == 1:
            continue

        r_normX_j = r_screen * sqrt(norm2_X[j])

        if screening == DGST3:
            screen_test = norm_row_XTcenter[j] + r_normX_j < 1

        else:
            screen_test = norm_row_XTR[j] / dual_scale + r_normX_j < 1

        if screen_test == 1:

            # Update residual
            # residual += np.dot(X[:, j], beta_old - beta[j, :])
            for k in range(n_tasks):

                if beta[j, k] != 0:
                    daxpy(& n_samples, & beta[j, k], & X[0, j], & inc,
                          & residual[0, k], & inc)

                    beta[j, k] = 0.
                # we "set" x_j to zero since it is inactive
                XTR[j, k] = 0.

            disabled_features[j] = 1
            n_active_features[0] -= 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bcd_fast(double[::1, :] X, double[::1, :] y, double[:, ::1] beta,
             double[::1, :] residual, double[:, ::1] XTR,
             double[:] norm_row_XTR, double[::1, :] n_DGST3,
             double norm2_n_DGST3, double[:] nTy_DGST3,
             int n_samples, int n_features, int n_tasks, double[:] norm2_X,
             double lambda_, double lambda_max, double lambda_prec,
             double dual_scale, int max_iter, int f, double tol, int screening,
             int[:] disabled_features, int wstr_plus):
    """
        Solve the sparse-group-lasso regression with elastic-net
        We minimize
        f(beta) + lambda_1 Omega(beta) + 0.5 * lambda_2 norm(beta, 2)^2
        where f(beta) = 0.5 * norm(y - X beta, 2)^2 and
        Omega(beta) = tau norm(beta, 1) +
                      (1 - tau) * sum_g omega_g * norm(beta_g, 2)
    """

    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    cdef int inc = 1
    cdef int n_iter = 666
    cdef int n_samples_n_tasks = n_samples * n_tasks
    cdef int n_active_features = n_features

    cdef double gap_t = 1.
    cdef double double_tmp = 0.
    cdef double mu_g = 0.
    cdef double L_g = 0.
    cdef double norm_grad = 0.
    cdef double norm_res2 = dnrm2(& n_samples_n_tasks, & residual[0, 0], & inc) ** 2
    cdef double r_screen = 666.
    cdef double delta_DGST3 = (lambda_max / lambda_ - 1) / norm2_n_DGST3

    cdef double norm_beta_j = 0.

    cdef double[:] beta_old_g = np.zeros(n_tasks, order='F')
    cdef double[:] gradient_step = np.zeros(n_tasks, order='F')

    cdef double[::1, :] center_DGST3 = np.zeros((n_samples, n_tasks), order='F')
    cdef double[:] norm_row_XTcenter = np.zeros(n_features, order='F')

    cdef int violation = n_features

    with nogil:
        if wstr_plus == 0:
            # disabled_features warm_start++
            for j in range(n_features):
                disabled_features[j] = 0

        if screening == STRONG_GAP_SAFE:
            screening = GAPSAFE

        if screening == STRONG_RULE:

            for j in range(n_features):

                if norm_row_XTR[j] < 2 * lambda_ - lambda_prec:
                    disabled_features[j] = 1
                    n_active_features -= 1

                    for k in range(n_tasks):
                        beta[j, k] = 0.

        if screening == DGST3:

            for k in range(n_tasks):
                for i in range(n_samples):
                    center_DGST3[i, k] = y[i, k] / lambda_ - \
                                         delta_DGST3 * n_DGST3[i, k]

            for j in range(n_features):
                # compute norm of XTcenter Bonnefoy
                norm_row_XTcenter[j] = 0
                for k in range(n_tasks):

                    norm_row_XTcenter[j] += ddot(& n_samples, & X[0, j], & inc,
                                                 & center_DGST3[0, k], & inc) ** 2
                norm_row_XTcenter[j] = sqrt(norm_row_XTcenter[j])

        while violation > 0:
            for n_iter in range(max_iter):

                if f != 0 and n_iter % f == 0:

                    # Compute dual point by dual scaling :
                    # theta_k = residual / dual_scale
                    dual_scale = 0.
                    for j in range(n_features):

                        if disabled_features[j] == 1:
                            continue

                        norm_row_XTR[j] = 0
                        # XTR[g_j] = np.dot(X[:, g_j], residual)
                        for k in range(n_tasks):
                            XTR[j, k] = ddot(& n_samples, & X[0, j], & inc,
                                             & residual[0, k], & inc)

                            norm_row_XTR[j] += XTR[j, k] ** 2

                        norm_row_XTR[j] = sqrt(norm_row_XTR[j])
                        dual_scale = fmax(dual_scale, norm_row_XTR[j])

                    dual_scale = fmax(lambda_, dual_scale)

                    norm_res2 = dnrm2(& n_samples_n_tasks, & residual[0, 0], & inc) ** 2

                    gap_t = dual_gap(n_samples, n_features, n_tasks, beta, residual, y,
                                     norm_res2, dual_scale, lambda_, & disabled_features[0])

                    if gap_t <= tol:
                        break

                    if screening == DGST3:

                        r_screen = 0.
                        for i in range(n_samples):
                            for k in range(n_tasks):
                                r_screen += (y[i, k] / lambda_ - \
                                             residual[i, k] / dual_scale) ** 2

                        r_screen = sqrt(r_screen - delta_DGST3 * (lambda_max / lambda_ - 1))
                        screen_variable(n_samples, n_features, n_tasks,
                                        & n_active_features, X, beta, residual,
                                        XTR, & norm2_X[0],
                                        & norm_row_XTcenter[0],
                                        & norm_row_XTR[0], dual_scale,
                                        & disabled_features[0],
                                        r_screen, screening)

                    if screening in [GAPSAFE, GAPSAFE_SEQ]:

                        if screening == GAPSAFE_SEQ and n_iter >= 1:
                            pass

                        else:

                            for j in range(n_features):

                                if disabled_features[j] == 1:
                                    continue

                            r_screen = sqrt(2 * gap_t) / lambda_

                            screen_variable(n_samples, n_features, n_tasks,
                                            & n_active_features, X, beta, residual,
                                            XTR, & norm2_X[0],
                                            & norm_row_XTcenter[0],
                                            & norm_row_XTR[0], dual_scale,
                                            & disabled_features[0],
                                            r_screen, screening)

                # Bloc-coordinate descent loop
                for j in range(n_features):

                    if disabled_features[j] == 1:
                        continue

                    if norm2_X[j] == 0:
                        continue

                    L_g = norm2_X[j]

                    # group soft tresholding
                    mu_g = lambda_ / L_g

                    norm_grad = 0
                    for k in range(n_tasks):

                        beta_old_g[k] = beta[j, k]

                        # XTR[g_j] = np.dot(X[:, g_j], residual)
                        XTR[j, k] = ddot(& n_samples, & X[0, j], & inc,
                                         & residual[0, k], & inc)

                        gradient_step[k] = beta[j, k] + XTR[j, k] / L_g
                        norm_grad += gradient_step[k] ** 2

                    norm_grad = sqrt(norm_grad)
                    scaling = fmax(1. - mu_g / norm_grad, 0.)

                    for k in range(n_tasks):

                        beta[j, k] = scaling * gradient_step[k]
                        # Update residual
                        # residual += np.dot(X[:, j], beta_old - beta[j, :])
                        double_tmp = beta_old_g[k] - beta[j, k]
                        daxpy(& n_samples, & double_tmp, & X[0, j], & inc,
                              & residual[0, k], & inc)

            if screening == STRONG_RULE:
                # check violation of KKT condition
                violation = 0
                for j in range(n_features):

                    if disabled_features[j] == 0:
                        continue

                    norm_beta_j = dnrm2(& n_tasks, & beta[j, 0], & inc)

                    if norm_beta_j != 0:
                        for k in range(n_tasks):

                            if fabs(XTR[j, k] - lambda_ * beta[j, k] / norm_beta_j ) > 1e-12:
                                disabled_features[j] = 0
                                violation += 1
                                break

                    else:

                        if fabs(norm_row_XTR[j] - lambda_) <= 1e-12 or\
                           norm_row_XTR[j] <= lambda_:
                            pass

                        else:
                            disabled_features[j] = 0
                            violation += 1
            else:
                violation = 0

    return (gap_t, dual_scale, n_iter, n_active_features)

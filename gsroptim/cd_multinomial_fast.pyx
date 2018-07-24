# TODO: clean more and add strong gap safe


from libc.math cimport fabs, sqrt, log, log1p, exp
from libc.stdlib cimport qsort
from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dscal, dasum
import numpy as np
cimport numpy as np
cimport cython

cdef int NO_SCREEN = 0
cdef int GAPSAFE = 1

cdef double np_inf = np.inf

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


cdef double min(int n, double * a) nogil:
    """np.min(a)"""
    cdef int i
    cdef double m = a[0]
    cdef double d
    for i in range(1, n):
        d = a[i]
        if d < m:
            m = d
    return m


# mimic rounding to zero
cdef double near_zero(double a) nogil:
    if fabs(a) <= 1e-14:
        return 0
    return a


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double primal(int n_samples, int n_features, int n_tasks,
                   double[::1, :] y, double[:, ::1] beta, double[::1, :] Xbeta,
                   double[::1, :] exp_Xbeta, double lambda_) nogil:

    cdef double group_norm = 0.
    cdef double fval = 0.
    cdef double l1l2_norm = 0.
    cdef double tmp1 = 0.
    cdef double tmp2 = 0.
    cdef double data_fit = 0.
    cdef int k = 0
    cdef int j = 0
    cdef int i = 0
    cdef int inc = 1

    for j in range(n_features):

        l1l2_norm += dnrm2(& n_tasks, & beta[j, 0], & inc)

    for i in range(n_samples):

        tmp1 = 0.
        tmp2 = 0.
        for k in range(n_tasks):
            tmp1 += y[i, k] * Xbeta[i, k]
            tmp2 += exp_Xbeta[k, i]

        data_fit += log(tmp2) - tmp1

    fval = data_fit + lambda_ * l1l2_norm

    return fval


cdef double NH(int size_x, double * x) nogil:

    cdef int i = 0
    cdef double sum_x = 0.
    cdef double xlogx = 0.

    if min(size_x, & x[0]) < 0:
        return np_inf

    for i in range(size_x):
        sum_x += x[i]

    # if sum_x != 1.:
    if fabs(sum_x - 1.) > 1e-12:
        return np_inf

    for i in range(size_x):
        xlogx += x[i] * log(x[i])

    return xlogx


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dual(int n_samples, int n_tasks, double * tab, double[::1, :] y,
                 double[::1, :] residual, double dual_scale, double lambda_) nogil:

    # tab is a temporary array of size n_tasks
    cdef double alpha = lambda_ / dual_scale
    cdef double dval = 0.
    cdef int k = 0
    cdef int i = 0

    for i in range(n_samples):

        for k in range(n_tasks):
            tab[k] = y[i, k] - alpha * residual[i, k]

        dval -= NH(n_tasks, & tab[0])

    return dval


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dual_gap(int n_samples, int n_features, int n_tasks,
                     double[::1, :] y, double[:, ::1] beta,
                     double[::1, :] Xbeta, double[::1, :] exp_Xbeta,
                     double * tab, double[::1, :] residual, double dual_scale,
                     double lambda_) nogil:

    cdef double pobj = primal(n_samples, n_features, n_tasks, y, beta, Xbeta,
                              exp_Xbeta, lambda_)

    cdef double dobj = dual(n_samples, n_tasks, & tab[0], y, residual,
                            dual_scale, lambda_)

    cdef double gap_ = pobj - dobj

    return gap_


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def bcd_fast(double[::1, :] X, double[::1, :] y, double[:, ::1] beta,
             double[::1, :] residual, double[::1, :] Xbeta,
             double[::1, :] exp_Xbeta, int n_samples, int n_features,
             int n_tasks, double[:] norm2_X, double lambda_, int max_iter,
             int f, double tol, int screen, int[:] disabled_features,
             int wstr_plus=1):
    """
        Solve the sparse-group-lasso regression with elastic-net
        We minimize ...
    """

    cdef int i = 0
    cdef int k = 0
    cdef int j = 0
    cdef int inc = 1
    cdef int n_iters = 0
    cdef int n_samples_n_tasks = n_samples * n_tasks
    cdef int screen_test = 0
    cdef int n_active_features = n_features

    cdef double dual_scale = 0.
    cdef double gap_t = 1.
    cdef double double_tmp = 0.
    cdef double mu_g = 0.
    cdef double L_g = 0.
    cdef double norm_grad = 0.
    cdef double r_normX_j = 0.
    cdef double r_screen = np_inf

    cdef double[:] beta_old_g = np.zeros(n_tasks, order='F')
    cdef double[:] gradient_step = np.zeros(n_tasks, order='F')
    cdef double[::1, :] XTR = np.zeros((n_features, n_tasks), order='F')
    cdef double[:] tab = np.zeros(n_tasks, order='F')
    cdef double[:] norm_scale = np.zeros(n_samples, order='F')
    cdef double[:] norm_row_XTR = np.zeros(n_features, order='F')

    with nogil:
        if wstr_plus == 0:
            # disabled_features warm_start++
            for j in range(n_features):
                disabled_features[j] = 0

        for n_iters in range(max_iter):

            if f != 0 and n_iters % f == 0:

                # Compute dual point by dual scaling :
                # theta_k = residual / dual_scale
                dual_scale = 0.
                for j in range(n_features):

                    # if disabled[j] == 1:
                    #     continue

                    norm_row_XTR[j] = 0.
                    # XTR[j] = np.dot(X[:, j], residual)
                    for k in range(n_tasks):

                        XTR[j, k] = ddot(& n_samples, & X[0, j], & inc,
                                         & residual[0, k], & inc)
                        norm_row_XTR[j] += XTR[j, k] ** 2

                    norm_row_XTR[j] = sqrt(norm_row_XTR[j])
                    dual_scale = fmax(dual_scale, norm_row_XTR[j])

                dual_scale = fmax(lambda_, dual_scale)

                # norm_res2 = dnrm2(& n_samples_n_tasks, & residual[0, 0], & inc) ** 2

                gap_t = dual_gap(n_samples, n_features, n_tasks, y, beta, Xbeta,
                                 exp_Xbeta, & tab[0], residual, dual_scale, lambda_)

                if screen == GAPSAFE:

                    r_screen = sqrt(2 * gap_t) / lambda_

                    for j in range(n_features):

                        if disabled_features[j] == 1:
                            continue

                        r_normX_j = r_screen * sqrt(norm2_X[j])

                        # Gap Safe Screening !!!
                        screen_test = norm_row_XTR[j] / dual_scale + r_normX_j < 1

                        if screen_test == 1:

                            # Update residual
                            # residual += np.dot(X[:, j], beta_old - beta[j, :])
                            for k in range(n_tasks):

                                if beta[j, k] != 0.:
                                    daxpy(& n_samples, & beta[j, k], & X[0, j],
                                          & inc, & residual[0, k], & inc)

                                    beta[j, k] = 0.
                                # we "set" x_j to zero since it is inactive
                                XTR[j, k] = 0.

                            disabled_features[j] = 1
                            n_active_features -= 1

                if gap_t <= tol:
                    break

            # Bloc-coordinate descent loop
            for j in range(n_features):

                L_g = norm2_X[j]
                mu_g = lambda_ / L_g

                norm_grad = 0.
                for k in range(n_tasks):

                    beta_old_g[k] = beta[j, k]
                    # XTR[j] = np.dot(X[:, j], residual)
                    XTR[j, k] = ddot(& n_samples, & X[0, j], & inc,
                                     & residual[0, k], & inc)

                    gradient_step[k] = beta[j, k] + XTR[j, k] / L_g

                    norm_grad += gradient_step[k] ** 2

                norm_grad = sqrt(norm_grad)
                scaling = fmax(1. - mu_g / norm_grad, 0.)
                # beta[g] = scaling * gradient_step
                for k in range(n_tasks):
                    beta[j, k] = scaling * gradient_step[k]
                    double_tmp = beta[j, k] - beta_old_g[k]
                    # Xbeta[:, k] = np.dot(X, beta[:, k])
                    daxpy(& n_samples, & double_tmp, & X[0, j],
                          & inc, & Xbeta[0, k], & inc)

                    for i in range(n_samples):
                        exp_Xbeta[k, i] = exp(Xbeta[i, k])

                for i in range(n_samples):
                    norm_scale[i] = dasum(& n_tasks, & exp_Xbeta[0, i], & inc)
                    for k in range(n_tasks):
                        residual[i, k] = y[i, k] - exp_Xbeta[k, i] / norm_scale[i]

    return (dual_scale, gap_t, n_active_features, n_iters)

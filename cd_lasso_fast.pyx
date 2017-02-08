from libc.math cimport fabs, sqrt
from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dscal
import numpy as np
cimport numpy as np
cimport cython


cdef int NO_SCREENING = 0
cdef int STATIC = 1
cdef int DST3 = 2
cdef int GAPSAFE_SEQ = 3
cdef int GAPSAFE = 4


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
cdef double primal_value(int n_features,
                         double * beta_data,
                         double norm_residual,
                         double lambda_,
                         int * disabled_features) nogil:

    cdef double l1_norm = 0
    cdef int j = 0

    for j in range(n_features):
        if disabled_features[j] == 1:
            continue
        l1_norm += fabs(beta_data[j])

    return 0.5 * norm_residual ** 2 + lambda_ * l1_norm


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dual(int n_samples,
                 int n_features,
                 double * residual_data,
                 double * y_data,
                 double dual_scale,
                 double norm_residual,
                 double lambda_) nogil:

    cdef int inc = 1
    cdef double dval = 0.
    cdef double alpha = lambda_ / dual_scale
    cdef double Ry = ddot(& n_samples, residual_data, & inc, y_data, & inc)

    if dual_scale != 0:
        dval = alpha * Ry - 0.5 * (alpha * norm_residual) ** 2

    return dval


cdef double ST(double u, double x) nogil:
    return fsign(x) * fmax(fabs(x) - u, 0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void screen_variables(int n_samples, int n_features,
                           int * n_active_features, double[::1, :] X,
                           double * residual, double * XTR, double * XTcenter,
                           double * beta, int * disabled_features,
                           double * norm_X2, double r_screen) nogil:

    cdef int j = 0
    cdef int inc = 1
    cdef double r_normX_j = 0.

    for j in range(n_features):

        if disabled_features[j] == 1:
            continue

        r_normX_j = r_screen * sqrt(norm_X2[j])
        if r_normX_j >= 1.:
            # screening test obviously will fail
            continue

        if fabs(XTcenter[j]) + r_normX_j < 1.:

            if beta[j] != 0.:
                # Update residual
                daxpy(& n_samples, & beta[j], & X[0, j],
                      & inc, & residual[0], & inc)
                # we "set" x_j to zero since the j_th feature is inactive
                XTR[j] = 0.
                beta[j] = 0.
            disabled_features[j] = 1
            n_active_features[0] -= 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cd_lasso(double[::1, :] X, double[:] y, double[:] beta, double[:] XTR,
             double[:] XTy, double[:] residual, int[:] disabled_features,
             double nrm2_y, double[:] norm_X2, double lambda_,
             double lambda_max, double dual_scale, double eps, int max_iter,
             int f, int screening, int j_star=0, int wstr_plus=0):
    """
        Solve 1/2 ||y - X beta||^2 + lambda_ ||beta||_1
    """

    cdef int i = 0
    cdef int k = 0
    cdef int j = 0
    cdef int inc = 1

    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_active_features = n_features

    cdef double tol = eps * nrm2_y
    cdef double gap_t = 1
    cdef double double_tmp = 0
    cdef double mu = 0
    cdef double beta_old_j = 0
    cdef double p_obj = 0.
    cdef double d_obj = 0.
    cdef double r_normX_j = 0.
    cdef double r_screen = np.inf
    cdef double norm_residual = dnrm2(& n_samples, & residual[0], & inc)
    cdef double[:] XTcenter = np.zeros(n_features, order='F')

    # Computation for DST3
    cdef double delta = 0.
    cdef double[:] center_DST3 = np.zeros(n_samples, order='F')

    with nogil:

        if wstr_plus == 0:
            for j in range(n_features):
                disabled_features[j] = 0

        if screening == DST3:
            delta = (lambda_max / lambda_ - 1) / sqrt(norm_X2[j_star])
            for i in range(n_samples):
                center_DST3[i] = y[i] / lambda_ - delta * X[i, j_star]

        if screening == STATIC:

            for j in range(n_features):
                XTcenter[j] = XTy[j] / lambda_

            r_screen = sqrt(nrm2_y) * fabs(1 / lambda_max - 1 / lambda_)
            screen_variables(n_samples, n_features, & n_active_features, X,
                             & residual[0], & XTR[0], & XTcenter[0], & beta[0],
                             & disabled_features[0], & norm_X2[0], r_screen)

        # Sequential GAPSAFE rule (with preceding optimal points)
        if screening == GAPSAFE_SEQ:

            for j in range(n_features):
                XTcenter[j] = XTR[j] / dual_scale

            p_obj = primal_value(n_features, & beta[0], norm_residual,
                                 lambda_, & disabled_features[0])
            d_obj = dual(n_samples, n_features, & residual[0], & y[0],
                         dual_scale, norm_residual, lambda_)

            r_screen = sqrt(2. * (p_obj - d_obj)) / lambda_
            screen_variables(n_samples, n_features, & n_active_features, X,
                             & residual[0], & XTR[0], & XTcenter[0], & beta[0],
                             & disabled_features[0], & norm_X2[0], r_screen)

        for k in range(max_iter):

            if f != 0 and k % f == 0:

                for j in range(n_features):

                    XTR[j] = ddot(& n_samples, & X[0, j], & inc,
                                  & residual[0], & inc)

                # Compute dual point by dual scaling :
                # theta_k = residual / dual_scale
                double_tmp = 0
                for j in range(n_features):

                    if disabled_features[j] == 1:
                        continue

                    double_tmp = fmax(double_tmp, fabs(XTR[j]))

                dual_scale = fmax(lambda_, double_tmp)
                norm_residual = dnrm2(& n_samples, & residual[0], & inc)

                p_obj = primal_value(n_features, & beta[0], norm_residual,
                                     lambda_, & disabled_features[0])

                d_obj = dual(n_samples, n_features, & residual[0], & y[0],
                             dual_scale, norm_residual, lambda_)
                gap_t = p_obj - d_obj

                if gap_t <= tol:
                    break

                if screening == DST3:

                    r_screen = 0.
                    for i in range(n_samples):
                        r_screen += (residual[i] / dual_scale - y[i] / lambda_) ** 2
                    r_screen = sqrt(r_screen - delta ** 2)

                    for j in range(n_features):
                        XTcenter[j] = ddot(& n_samples, & X[0, j], & inc,
                                           & center_DST3[0], & inc)

                    screen_variables(n_samples, n_features,
                                     & n_active_features, X, & residual[0],
                                     & XTR[0], & XTcenter[0], & beta[0],
                                     & disabled_features[0], & norm_X2[0],
                                     r_screen)

                # Dynamic Gap Safe rule
                if screening == GAPSAFE:

                    for j in range(n_features):
                        XTcenter[j] = XTR[j] / dual_scale
                    r_screen = sqrt(2 * gap_t) / lambda_
                    screen_variables(n_samples, n_features,
                                     & n_active_features, X, & residual[0],
                                     & XTR[0], & XTcenter[0], & beta[0],
                                     & disabled_features[0], & norm_X2[0],
                                     r_screen)

            # Coordinate descent
            for j in range(n_features):

                if disabled_features[j] == 1:
                    continue

                mu = lambda_ / norm_X2[j]
                beta_old_j = beta[j]
                XTR[j] = ddot(& n_samples, & X[0, j], & inc,
                              & residual[0], & inc)
                beta[j] = ST(mu, beta[j] + XTR[j] / norm_X2[j])

                if beta[j] != beta_old_j:

                    double_tmp = beta_old_j - beta[j]
                    # Update residual
                    daxpy(& n_samples, & double_tmp, & X[0, j],
                          & inc, & residual[0], & inc)

    return gap_t, dual_scale, k, n_active_features
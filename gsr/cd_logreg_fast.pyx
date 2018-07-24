from libc.math cimport fabs, sqrt, log, exp  # log1p
from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dscal, dcopy
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DOUBLE

cdef double np_inf = np.inf
cdef int NO_SCREENING = 0
cdef int GAPSAFE_SEQ = 1
cdef int GAPSAFE = 2
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


cdef inline double log_1pexp(double z) nogil:
    if z < -18:
        return exp(z)
    elif z > 18:
        return z
    else:
        return log(1. + exp(z))


cdef int isclose(double a, double b) nogil:

    if fabs(a) > 1e12 and fabs(b) > 1e12:
        return 1

    if fabs(a - b) < 1e-12:
        return 1

    return 0


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


cdef double Nh(double x) nogil:

    if 0 <= x and x <= 1:
        return x * log(x) + (1 - x) * log(1 - x)

    return np_inf


cdef double primal_log(double * Xbeta, double * exp_Xbeta, double * y,
                       double * beta, double lambda_,
                       int * disabled_features, double norm1_beta,
                       int n_samples, int n_features) nogil:

    cdef int inc = 1
    cdef int i = 0
    cdef int j = 0
    cdef double loss = 0
    cdef double tmp = 0
    cdef double primal = 0
    cdef double yTXbeta = ddot(& n_samples, & y[0], & inc, & Xbeta[0], & inc)

    for i in range(n_samples):
        # tmp += log1p(exp_Xbeta[i])
        tmp += log_1pexp(Xbeta[i])
    loss = -yTXbeta + tmp

    primal = loss + lambda_ * norm1_beta
    return primal


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dual_log(double * y, double * residual, double dual_scale,
                     double  lambda_, int n_samples) nogil:

    cdef int i = 0
    cdef double alpha = lambda_ / dual_scale
    cdef double dual = 0

    for i in range(n_samples):
        dual -= Nh(y[i] - alpha * residual[i])

    return dual


cdef double ST(double u, double x) nogil:
    return fsign(x) * fmax(fabs(x) - u, 0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double step_size(double[:] beta, double[:] beta_next, double[:] y,
                      double[::1, :] X, double[:] Xbeta,
                      double[:] Xbeta_next, double[:] exp_Xbeta_next,
                      double beta_prox, double delta, double p_obj,
                      double norm1_beta_next,
                      double XTR_j, double lambda_, int * disabled_features,
                      int n_samples, int n_features, int j) nogil:

    cdef int numerical_issue = 0
    cdef int decrease = 0
    cdef int t = 0
    cdef int i = 0
    cdef int inc = 1
    cdef double sigma = 0.01
    cdef double w = 0.5
    cdef double beta_next_old_j = 0.
    cdef double norm_beta_j_term = \
        lambda_ * (fabs(beta[j] + delta) - fabs(beta[j]))
    cdef double tmp = sigma * (norm_beta_j_term - XTR_j * delta)

    dcopy(& n_features, & beta[0], & inc, & beta_next[0], & inc)
    dcopy(& n_samples, & Xbeta[0], & inc, & Xbeta_next[0], & inc)

    while decrease == 0:

        alpha = w ** t

        beta_next_old_j = beta_next[j]
        beta_next[j] = (1. - alpha) * beta[j] + alpha * beta_prox

        if beta_next[j] == beta_next_old_j:
            break

        norm1_beta_next += fabs(beta_next[j]) - fabs(beta_next_old_j)

        for i in range(n_samples):
            Xbeta_next[i] += X[i, j] * (beta_next[j] - beta_next_old_j)
            exp_Xbeta_next[i] = exp(Xbeta_next[i])

        p_obj_next = primal_log(& Xbeta_next[0], & exp_Xbeta_next[0], & y[0],
                                & beta_next[0], lambda_, & disabled_features[0],
                                norm1_beta_next, n_samples, n_features)

        numerical_issue = isclose(p_obj_next - p_obj, alpha * tmp)
        if p_obj_next - p_obj <= alpha * tmp or numerical_issue:
            decrease = 1
        t += 1

    return alpha


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void screen_variables(int n_samples, int n_features,
                           int * n_active_features, double * beta,
                           double * Xbeta, double * exp_Xbeta,
                           double[::1, :] X, double * y, double * residual,
                           double * XTR, int * disabled_features,
                           double * norm_X2,
                           double dual_scale, double r_screen,
                           int screening=GAPSAFE) nogil:

    cdef int j = 0
    cdef int i = 0
    cdef double r_normX_j = 0.

    for j in range(n_features):

        if disabled_features[j] == 1:
            continue

        # if screening == GAPSAFE:
        r_normX_j = r_screen * sqrt(norm_X2[j])

        if r_normX_j >= 1.:
            # screening test obviously will fail
            continue

        if fabs(XTR[j] / dual_scale) + r_normX_j < 1.:
            # TODO: Update residual (removed ?)
            if beta[j] != 0:

                for i in range(n_samples):
                    Xbeta[i] -= X[i, j] * beta[j]
                    if Xbeta[i] > 18:
                        exp_Xbeta[i] = Xbeta[i]
                    else:
                        exp_Xbeta[i] = exp(Xbeta[i])
                    residual[i] = y[i] - exp_Xbeta[i] / (1. + exp_Xbeta[i])

                beta[j] = 0.
            # we "set" x_j to zero since the j_th feature is inactive
            XTR[j] = 0.

            disabled_features[j] = 1
            n_active_features[0] -= 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cd_logreg(double[::1, :] X, double[:] y, double[:] beta,
              double[:] XTR, double[:] Xbeta, double[:] exp_Xbeta,
              double[:] residual, int[:] disabled_features, double nrm2_y,
              double[:] norm_X2, double lambda_, double lambda_prec,
              double tol, double dual_scale, int max_iter, int f,
              int screening, int wstr_plus=0):
    """
        Solve ? + lambda_ ||beta||_1
    """

    cdef int i = 0
    cdef int j = 0
    cdef int n_iters = 0
    cdef int inc = 1

    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_active_features = n_features

    cdef double gap_t = 1.
    cdef double r_screen = np_inf
    cdef double double_tmp = 0  # the most important generic double variable
    cdef double mu = 0
    cdef double beta_old_j = 0
    cdef double p_obj = 0.
    cdef double d_obj = 0.
    cdef double L_j = 0
    cdef double delta_j = 0.
    cdef double beta_prox_j = 0.
    cdef double alpha_j = 0.
    cdef double norm1_beta = 0.
    cdef double[:] Hessian = np.zeros(n_features, order='F')
    cdef double[:] beta_next = np.zeros(n_features, order='F')
    cdef double[:] Xbeta_next = np.zeros(n_samples, order='F')
    cdef double[:] exp_Xbeta_next = np.ones(n_samples, order='F')

    # for those who violate the rules !
    cdef int violation = n_features

    with nogil:

        if wstr_plus == 0:
            for j in range(n_features):
                disabled_features[j] = 0

        for j in range(n_features):

            if disabled_features[j] == 1:
                continue

            norm1_beta += fabs(beta[j])

        if screening == STRONG_RULE:

            for j in range(n_features):

                if fabs(XTR[j]) < 2 * lambda_ - lambda_prec:
                    disabled_features[j] = 1
                    n_active_features -= 1
                    beta[j] = 0.

        while violation > 0:

            for n_iters in range(max_iter):

                if f != 0 and n_iters % f == 0:

                    double_tmp = 0.
                    for j in range(n_features):

                        if disabled_features[j] == 1:
                            continue

                        XTR[j] = ddot(& n_samples, & X[0, j], & inc,
                                      & residual[0], & inc)

                        # Compute dual point by dual scaling :
                        # theta_k = residual / dual_scale
                        double_tmp = fmax(double_tmp, fabs(XTR[j]))

                    dual_scale = fmax(lambda_, double_tmp)

                    p_obj = primal_log(& Xbeta[0], & exp_Xbeta[0], & y[0],
                                       & beta[0], lambda_, & disabled_features[0],
                                       norm1_beta, n_samples, n_features)

                    d_obj = dual_log(& y[0], & residual[0], dual_scale, lambda_,
                                     n_samples)
                    gap_t = p_obj - d_obj

                    if gap_t <= tol:
                        break

                    # Dynamic Gap Safe rule
                    if screening in [GAPSAFE, GAPSAFE_SEQ]:

                        if screening == GAPSAFE_SEQ and n_iters >= 1:
                            pass

                        else:
                            r_screen = sqrt(2 * gap_t) / (2 * lambda_)

                            screen_variables(n_samples, n_features,
                                             & n_active_features, & beta[0],
                                             & Xbeta[0], & exp_Xbeta[0], X, & y[0],
                                             & residual[0], & XTR[0],
                                             & disabled_features[0], & norm_X2[0],
                                             dual_scale, r_screen)

                # Coordinate descent with line search
                for j in range(n_features):

                    if disabled_features[j] == 1:
                        continue

                    Hessian[j] = 0
                    for i in range(n_samples):
                        Hessian[j] += X[i, j] ** 2 * exp_Xbeta[i] / (1. + exp_Xbeta[i]) ** 2

                    L_j = fmax(Hessian[j], 1e-12)
                    mu = lambda_ / L_j

                    beta_old_j = beta[j]
                    XTR[j] = ddot(& n_samples, & X[0, j], & inc,
                                  & residual[0], & inc)

                    beta_prox_j = ST(mu, beta[j] + XTR[j] / L_j)

                    # line search
                    delta_j = beta_prox_j - beta[j]

                    p_obj = primal_log(& Xbeta[0], & exp_Xbeta[0], & y[0],
                                       & beta[0], lambda_, & disabled_features[0],
                                       norm1_beta, n_samples, n_features)

                    alpha_j = step_size(beta, beta_next, y, X, Xbeta, Xbeta_next,
                                        exp_Xbeta_next, beta_prox_j, delta_j, p_obj,
                                        norm1_beta, XTR[j], lambda_, & disabled_features[0],
                                        n_samples, n_features, j)

                    # Update beta[j]
                    beta[j] = beta[j] + alpha_j * delta_j

                    if beta[j] != beta_old_j:

                        norm1_beta += fabs(beta[j]) - fabs(beta_old_j)

                        double_tmp = beta[j] - beta_old_j
                        # Xbeta += X[:, j].T(beta[j] - beta_old_j)
                        daxpy(& n_samples, & double_tmp, & X[0, j],
                              & inc, & Xbeta[0], & inc)

                        for i in range(n_samples):
                            if Xbeta[i] > 18:
                                exp_Xbeta[i] = Xbeta[i]
                            else:
                                exp_Xbeta[i] = exp(Xbeta[i])
                            residual[i] = y[i] - exp_Xbeta[i] / (1. + exp_Xbeta[i])

            if screening == STRONG_RULE:
                # check violation of KKT condition
                violation = 0
                for j in range(n_features):

                    if disabled_features[j] == 0:
                        continue

                    elif beta[j] != 0:

                        if fabs(XTR[j] - lambda_ * fsign(beta[j])) > 1e-12:
                            disabled_features[j] = 0
                            violation += 1

                    else:

                        if fabs(fabs(XTR[j]) - lambda_) <= 1e-12 or\
                           fabs(XTR[j]) <= lambda_:
                            pass

                        else:
                            disabled_features[j] = 0
                            violation += 1

            else:
                violation = 0

    return gap_t, dual_scale, n_iters, n_active_features

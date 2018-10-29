# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: Eugene Ndiaye
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# Gap Safe screening rules for sparsity enforcing penalties.
# https://arxiv.org/abs/1611.05780
# firstname.lastname@telecom-paristech.fr

from libc.math cimport fabs, sqrt, log, exp, log1p
from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dscal, dcopy
# from cython.parallel cimport prange, parallel
import numpy as np
cimport numpy as np
cimport cython

cdef:
    double inf = 1e18
    int inc = 1  # Default array increment for cython_blas operation
    int NO_SCREENING = 0
    int GAPSAFE_SEQ = 1
    int GAPSAFE = 2


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


cdef int isclose(double a, double b) nogil:

    if fabs(a) > 1e12 and fabs(b) > 1e12:
        return 1

    if fabs(a - b) < 1e-12:
        return 1

    return 0


cdef double Nh(double x) nogil:

    if 0 <= x and x <= 1:
        return x * log(x) + (1 - x) * log(1 - x)

    return inf


cdef double dual_log(double * y, double * residual, double dual_scale,
                     double  lambda_, int n_samples) nogil:

    cdef:
        int i = 0
        double alpha = lambda_ / dual_scale
        double dual = 0

    for i in range(n_samples):
        dual -= Nh(y[i] - alpha * residual[i])

    return dual


cdef double ST(double u, double x) nogil:
    return fsign(x) * fmax(fabs(x) - u, 0)


cdef void compute_XTR_j(int n_samples, double * XTR_j, double * X_j_ptr,
                        double[::1] X_data, int[::1] X_indices, int startptr,
                        int endptr, double[::1] residual, int sparse) nogil:

    cdef:
        int i = 0
        int i_ptr = 0
    XTR_j[0] = 0.

    if sparse:

        for i_ptr in range(startptr, endptr):
            i = X_indices[i_ptr]
            XTR_j[0] += X_data[i_ptr] * residual[i]

    else:
        XTR_j[0] = ddot(& n_samples, X_j_ptr, & inc, & residual[0], & inc)


cdef void update_residual(int n_samples, double[::1] y, double * X_j,
                          double[::1] X_data, double[::1] Xbeta,
                          int[::1] X_indices, double[::1] residual,
                          int startptr, int endptr, double beta_diff,
                          double * yTXbeta, double * p_obj, double lambda_,
                          double norm1_beta, int sparse) nogil:

    cdef:
        int i = 0
        int i_ptr = 0
        double log_term = 0.
        double exp_Xbeta_i = 0.

    yTXbeta[0] = 0.

    # TODO: improve this part
    if sparse:
        for i_ptr in range(startptr, endptr):
            i = X_indices[i_ptr]
            Xbeta[i] += X_data[i_ptr] * beta_diff       

        for i in range(n_samples):
            yTXbeta[0] += y[i] * Xbeta[i]
            exp_Xbeta_i = exp(Xbeta[i])
            residual[i] = y[i] - exp_Xbeta_i / (1. + exp_Xbeta_i)
            log_term += log1p(exp_Xbeta_i)

    else:
        for i in range(n_samples):
            Xbeta[i] += X_j[i] * beta_diff
            exp_Xbeta_i = exp(Xbeta[i])
            residual[i] = y[i] - exp_Xbeta_i / (1. + exp_Xbeta_i)
            # update p_obj
            yTXbeta[0] += y[i] * Xbeta[i]
            log_term += log1p(exp_Xbeta_i)

    p_obj[0] = -yTXbeta[0] + log_term + lambda_ * norm1_beta


# cdef double step_size(double[::1] y, double * X_j, double[::1] Xbeta,
#                       double[::1] Xbeta_next, double[::1] X_data,
#                       int[::1] X_indices, double beta_j, double beta_prox,
#                       double delta, double p_obj, double norm1_beta_next,
#                       double XTR_j, double lambda_, int n_samples,
#                       int startptr, int endptr, int sparse) nogil:
#     cdef:
#         int numerical_issue = 0
#         int decrease = 0
#         int t = 0
#         int i = 0
#         int i_ptr = 0
#         double sigma = 0.01
#         double w = 0.5
#         double beta_next_old_j = 0.
#         double beta_next_j = beta_j
#         double yTXbeta_next = 0.
#         double log_term = 0.
#         double beta_diff = 0.
#         double norm_beta_j_term = \
#             lambda_ * (fabs(beta_j + delta) - fabs(beta_j))
#         double tmp = sigma * (norm_beta_j_term - XTR_j * delta)

#     dcopy(& n_samples, & Xbeta[0], & inc, & Xbeta_next[0], & inc)

#     while decrease == 0:

#         alpha = w ** t

#         beta_next_old_j = beta_next_j
#         beta_next_j = (1. - alpha) * beta_j + alpha * beta_prox
#         beta_diff = beta_next_j - beta_next_old_j

#         if beta_diff == 0.:
#             break

#         norm1_beta_next += fabs(beta_next_j) - fabs(beta_next_old_j)
#         yTXbeta_next = 0.
#         log_term = 0.

#         if sparse:
#             for i_ptr in range(startptr, endptr):
#                 i = X_indices[i_ptr]
#                 Xbeta_next[i] += X_data[i_ptr] * beta_diff
#                 yTXbeta_next += y[i] * Xbeta_next[i]

#             for i in range(n_samples):
#                 log_term += log1p(exp(Xbeta_next[i]))
#         else:
#             for i in range(n_samples):
#                 Xbeta_next[i] += X_j[i] * beta_diff
#                 yTXbeta_next += y[i] * Xbeta_next[i]
#                 log_term += log1p(exp(Xbeta_next[i]))

#         p_obj_next = -yTXbeta_next + log_term + lambda_ * norm1_beta_next

#         numerical_issue = isclose(p_obj_next - p_obj, alpha * tmp)
#         if p_obj_next - p_obj <= alpha * tmp or numerical_issue:
#             decrease = 1
#         t += 1

#     return alpha


def cd_logreg(double[::1, :] X, double[::1] X_data, int[::1] X_indices,
              int[::1] X_indptr, double[::1] y, double[::1] beta,
              double[::1] XTR, double[::1] Xbeta,
              double[::1] Xbeta_next, double[::1] residual,
              int[::1] disabled_features, double[::1] norm_X2,
              double p_obj, double norm1_beta, double lambda_,
              double tol, double dual_scale, int max_iter, int f,
              int screening, int wstr_plus=0, int sparse=0):
    """
    We solve:
    argmin_{beta} sum_{i=1}^{n} f_i(dot(x_{i}, beta)) + lambda * norm(beta, 1)
    where f_i(z) = -y_i * z + log(1 + exp(z)).
    """
    cdef:
        int i = 0
        int j = 0
        int i_ptr = 0
        int n_iters = 0
        int n_samples = y.shape[0]
        int n_features = beta.shape[0]
        int n_active_features = n_features
        int startptr = 0
        int endptr = n_samples
        double beta_old_j = 0
        double delta_j = 0.
        double beta_prox_j = 0.
        double alpha_j = 0.
        double yTXbeta = 0.
        double log_term = 0.
        double beta_diff = 0.
        double X_ij = 0.
        double double_tmp = 0
        double mu = 0
        double d_obj = 0.
        double L_j = 0.
        double exp_Xbeta_i = 0.
        double r_normX_j = 0.
        double r_screen = 1.
        double gap_t = inf
        double hessian_j = 0.
        double * X_j_ptr = & X[0, 0]

    with nogil:

        if wstr_plus == 0:
            for j in range(n_features):
                disabled_features[j] = 0

        for n_iters in range(max_iter):

            if f != 0 and n_iters % f == 0:

                double_tmp = 0.
                for j in range(n_features):

                    if disabled_features[j] == 1:
                        continue

                    if sparse:
                        startptr = X_indptr[j]
                        endptr = X_indptr[j + 1]
                    else:
                        X_j_ptr = & X[0, j]

                    compute_XTR_j(n_samples, & XTR[j], X_j_ptr, X_data,
                                  X_indices, startptr, endptr, residual,
                                  sparse)

                    # Compute dual point by dual scaling :
                    # theta_k = residual / dual_scale
                    double_tmp = fmax(double_tmp, fabs(XTR[j]))

                dual_scale = fmax(lambda_, double_tmp)
                d_obj = dual_log(& y[0], & residual[0], dual_scale, lambda_, n_samples)
                gap_t = p_obj - d_obj

                if gap_t <= tol:
                    break

                # Dynamic Gap Safe rule
                if screening in [GAPSAFE, GAPSAFE_SEQ]:

                    if screening == GAPSAFE_SEQ and n_iters >= 1:
                        pass

                    else:
                        r_screen = sqrt(2 * gap_t) / (2 * lambda_)
                        # safely screen variables
                        for j in range(n_features):

                            if disabled_features[j] == 1:
                                continue

                            if sparse:
                                startptr = X_indptr[j]
                                endptr = X_indptr[j + 1]
                            else:
                                X_j_ptr = & X[0, j]

                            # if screening == GAPSAFE:
                            r_normX_j = r_screen * sqrt(norm_X2[j])

                            if r_normX_j >= 1.:
                                # screening test obviously will fail
                                continue

                            if fabs(XTR[j] / dual_scale) + r_normX_j < 1.:

                                if beta[j] != 0:
                                    norm1_beta -= fabs(beta[j])
                                    update_residual(n_samples, y, X_j_ptr,
                                                    X_data, Xbeta, X_indices,
                                                    residual, startptr, endptr,
                                                    beta_diff, & yTXbeta,
                                                    & p_obj, lambda_,
                                                    norm1_beta, sparse)
                                    beta[j] = 0.
                                # we "set" x_j to zero since the j_th feature is inactive
                                XTR[j] = 0.
                                disabled_features[j] = 1
                                n_active_features -= 1

            # Coordinate descent with line search
            for j in range(n_features):

                if disabled_features[j] == 1:
                    continue

                hessian_j = 0.

                if sparse:
                    startptr = X_indptr[j]
                    endptr = X_indptr[j + 1]
                else:
                    X_j_ptr = & X[0, j]

                for i_ptr in range(startptr, endptr):
                    if sparse:
                        i = X_indices[i_ptr]
                        X_ij = X_data[i_ptr]
                    else:
                        i = i_ptr
                        X_ij = X[i, j]

                    exp_Xbeta_i = exp(Xbeta[i])
                    hessian_j += X_ij ** 2 * exp_Xbeta_i / (1. + exp_Xbeta_i) ** 2

                L_j = fmax(hessian_j, 1e-12)

                mu = lambda_ / L_j
                beta_old_j = beta[j]

                compute_XTR_j(n_samples, & XTR[j], X_j_ptr, X_data, X_indices,
                              startptr, endptr, residual, sparse)

                beta[j] = ST(mu, beta[j] + XTR[j] / L_j)
                # beta_prox_j = ST(mu, beta[j] + XTR[j] / L_j)
                # # line search
                # delta_j = beta_prox_j - beta[j]
                # alpha_j = step_size(y, & X[0, j], Xbeta, Xbeta_next, X_data,
                #                     X_indices, beta[j], beta_prox_j, delta_j,
                #                     p_obj, norm1_beta, XTR[j], lambda_,
                #                     n_samples, startptr, endptr, sparse)

                # # Update beta[j]
                # with gil:
                #     if alpha_j != 1.:
                #         print "alpha_j = ", alpha_j
                # beta[j] = beta[j] + alpha_j * delta_j
                beta_diff = beta[j] - beta_old_j

                if beta_diff != 0.:

                    norm1_beta += fabs(beta[j]) - fabs(beta_old_j)
                    update_residual(n_samples, y, X_j_ptr, X_data, Xbeta,
                                    X_indices, residual, startptr, endptr,
                                    beta_diff, & yTXbeta, & p_obj,
                                    lambda_, norm1_beta, sparse)

    return gap_t, p_obj, norm1_beta, dual_scale, n_iters, n_active_features

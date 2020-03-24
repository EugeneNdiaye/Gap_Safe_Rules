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

from libc.math cimport fabs, sqrt
from scipy.linalg.cython_blas cimport daxpy, ddot, dnrm2, dscal
import numpy as np
cimport numpy as np
cimport cython

cdef:
    int inc = 1  # Default array increment for cython_blas operation
    int NO_SCREENING = 0
    int GAPSAFE_SEQ = 1
    int GAPSAFE = 2
    int DEEPS = 414


cdef inline double fmax(double x, double y) nogil:
    if x > y:
        return x
    return y


cdef inline double fmin(double x, double y) nogil:
    if x < y:
        return x
    return y


cdef inline double fsign(double f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


def matrix_column_norm(int n_samples, int n_features, double[::1] X_data,
                       int[::1] X_indices, int[::1] X_indptr, double[::1] norm_Xcent,
                       double[::1] X_mean, int center=0):
    cdef:
        int i = 0
        int j = 0
        int i_ptr = 0
        int start = 0
        int end = 0
        double X_mean_j = 0.

    with nogil:
        for j in range(n_features):

            if center:
                X_mean_j = X_mean[j]

            start = X_indptr[j]
            end = X_indptr[j + 1]
            norm_Xcent[j] = 0.

            for i_ptr in range(start, end):
                norm_Xcent[j] += (X_data[i_ptr] - X_mean_j) ** 2

            if center:
                norm_Xcent[j] += (n_samples - end + start) * X_mean_j ** 2


cdef double primal_value(int n_features, double * beta_data, double norm_residual,
                         double lambda_, int * disabled_features) nogil:

    cdef:
        double l1_norm = 0
        int j = 0

    for j in range(n_features):
        if disabled_features[j] == 1:
            continue
        l1_norm += fabs(beta_data[j])

    return 0.5 * norm_residual ** 2 + lambda_ * l1_norm


cdef double dual(int n_samples, int n_features, double * residual_data,
                 double * y_data, double dual_scale, double norm_residual,
                 double lambda_) nogil:

    cdef:
        double dval = 0.
        double alpha = lambda_ / dual_scale
        double Ry = ddot(& n_samples, residual_data, & inc, y_data, & inc)

    if dual_scale != 0:
        dval = alpha * Ry - 0.5 * (alpha * norm_residual) ** 2

    return dval


cdef double ST(double u, double x) nogil:
    return fsign(x) * fmax(fabs(x) - u, 0)


cdef void update_residual(int n_samples, double * X_j, double[::1] X_data,
                          int[::1] X_indices, double * residual,
                          double * beta, double X_mean_j, int startptr,
                          int endptr, double beta_diff, double * sum_residual,
                          int sparse, int center) nogil:

    cdef:
        int i = 0
        int i_ptr = 0

    if sparse:
        for i_ptr in range(startptr, endptr):
            i = X_indices[i_ptr]
            residual[i] += X_data[i_ptr] * beta_diff

        if center:
            sum_residual[0] = 0.
            for i in range(n_samples):
                residual[i] -= X_mean_j * beta_diff
                sum_residual[0] += residual[i]
    else:
        daxpy(& n_samples, & beta_diff, X_j, & inc, & residual[0], & inc)


cdef void compute_XTR_j(int n_samples, double * XTR_j, double * X_j_ptr,
                        double[::1] X_data, int[::1] X_indices, int startptr,
                        int endptr, double * residual, double X_mean_j,
                        double sum_residual, int sparse, int center) nogil:

    cdef:
        int i = 0
        int i_ptr = 0
    XTR_j[0] = 0.

    if sparse:

        for i_ptr in range(startptr, endptr):
            i = X_indices[i_ptr]
            XTR_j[0] += X_data[i_ptr] * residual[i]

        if center:
            XTR_j[0] -= X_mean_j * sum_residual

    else:
        XTR_j[0] = ddot(& n_samples, X_j_ptr, & inc, & residual[0], & inc)


def cd_lasso(double[::1, :] X, double[::1] X_data, int[::1] X_indices,
             int[::1] X_indptr, double[::1] y, double[::1] X_mean, double[::1] beta,
             double[::1] norm_Xcent, double[::1] XTR, double[::1] residual,
             int[::1] disabled_features, double nrm2_y, double lambda_,
             double sum_residual, double tol, int max_iter, int f,
             int screening, int wstr_plus=0, int sparse=0, int center=0):
    """
        Solve 1/2 ||y - X beta||^2 + lambda_ ||beta||_1
    """

    cdef:
        int i = 0
        int j = 0
        int i_ptr = 0
        int startptr = 0
        int endptr = 0

        int n_iter = 0
        int n_samples = y.shape[0]
        int n_features = beta.shape[0]
        int n_active_features = n_features

        double gap_t = 1
        double double_tmp = 0
        double gamma = 1e-4
        double mu = 0
        double beta_old_j = 0
        double p_obj = 0.
        double d_obj = 0.
        double r_normX_j = 0.
        double X_mean_j = 0.
        double beta_diff = 0.
        double r_screen = 1.
        double norm_residual = 1.
        double dual_scale = 0.
        double p_obj_old = 0.
        double * X_j_ptr = & X[0, 0]

    with nogil:

        if wstr_plus == 0:
            for j in range(n_features):
                disabled_features[j] = 0

        for n_iter in range(max_iter):

            if f != 0 and n_iter % f == 0:

                # Computation of XTR
                double_tmp = 0.
                # Compute dual point by dual scaling :
                # theta_k = residual / dual_scale
                for j in range(n_features):

                    if disabled_features[j] == 1:
                        continue

                    if sparse:
                        startptr = X_indptr[j]
                        endptr = X_indptr[j + 1]
                        if center:
                            X_mean_j = X_mean[j]
                    else:
                        X_j_ptr = & X[0, j]

                    compute_XTR_j(n_samples, & XTR[j], X_j_ptr, X_data,
                                  X_indices, startptr, endptr, & residual[0],
                                  X_mean_j, sum_residual, sparse, center)

                    double_tmp = fmax(double_tmp, fabs(XTR[j]))

                dual_scale = fmax(lambda_, double_tmp)
                norm_residual = dnrm2(& n_samples, & residual[0], & inc)

                p_obj_old = p_obj
                p_obj = primal_value(n_features, & beta[0], norm_residual,
                                     lambda_, & disabled_features[0])

                d_obj = dual(n_samples, n_features, & residual[0], & y[0],
                             dual_scale, norm_residual, lambda_)
                gap_t = p_obj - d_obj

                if gap_t <= tol:
                    break

                # Dynamic Gap Safe rule
                if screening in [GAPSAFE, GAPSAFE_SEQ, DEEPS]:

                    if screening == GAPSAFE_SEQ and n_iter >= 1:
                        pass

                    else:
                        # Yes with a quadratic loss we can gain a factor of sqrt{2}
                        if screening == GAPSAFE:
                            r_screen = sqrt(gap_t) / lambda_

                        if screening == DEEPS:

                            double_tmp = (1 - gamma) * fabs(p_obj_old - p_obj) + gamma * gap_t
                            r_screen = sqrt(fmin(double_tmp, gap_t)) / lambda_

                        for j in range(n_features):

                            if disabled_features[j] == 1:
                                continue

                            r_normX_j = r_screen * sqrt(norm_Xcent[j])
                            if r_normX_j >= 1.:
                                # screening test obviously will fail
                                continue

                            if sparse:
                                startptr = X_indptr[j]
                                endptr = X_indptr[j + 1]
                                if center:
                                    X_mean_j = X_mean[j]

                            else:
                                X_j_ptr = & X[0, j]

                            if fabs(XTR[j]) + r_normX_j * dual_scale < dual_scale:

                                beta_old_j = beta[j]
                                beta[j] = 0.

                                if beta[j] != beta_old_j:
                                    beta_diff = beta_old_j - beta[j]
                                    update_residual(n_samples, X_j_ptr, X_data, X_indices,
                                                    & residual[0], & beta[0], X_mean_j,
                                                    startptr, endptr, beta_diff,
                                                    & sum_residual, sparse, center)

                                # we "set" x_j to zero since the j_th feature is inactive
                                XTR[j] = 0.
                                disabled_features[j] = 1
                                n_active_features -= 1

            # Coordinate descent
            for j in range(n_features):

                if disabled_features[j] == 1:
                    continue

                mu = lambda_ / norm_Xcent[j]
                beta_old_j = beta[j]

                if sparse:
                    startptr = X_indptr[j]
                    endptr = X_indptr[j + 1]
                    if center:
                        X_mean_j = X_mean[j]
                else:
                    X_j_ptr = & X[0, j]

                compute_XTR_j(n_samples, & XTR[j], X_j_ptr, X_data,
                              X_indices, startptr, endptr, & residual[0],
                              X_mean_j, sum_residual, sparse, center)

                beta[j] = ST(mu, beta[j] + XTR[j] / norm_Xcent[j])

                # TODO: add update of norm1_beta ??
                if beta[j] != beta_old_j:

                    beta_diff = beta_old_j - beta[j]
                    update_residual(n_samples, X_j_ptr, X_data, X_indices,
                                    & residual[0], & beta[0], X_mean_j,
                                    startptr, endptr, beta_diff,
                                    & sum_residual, sparse, center)

    return gap_t, sum_residual, n_iter, n_active_features

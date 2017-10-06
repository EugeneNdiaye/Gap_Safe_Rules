# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
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

cdef:
    int inc = 1  # Default array increment for cython_blas operation
    int NO_SCREEN = 0
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


cdef double primal_value(int n_samples, int n_features, int n_tasks,
                         double[:, ::1] beta, double norm_res2,
                         double lambda_, int * disabled_features) nogil:

    cdef:
        int j = 0
        double fval = 0.
        double l1l2_norm = 0.

    # l21_norm = np.sqrt(np.sum(W ** 2, axis=0)).sum()
    for j in range(n_features):

        if disabled_features[j] == 1:
            continue
        # np.sqrt(np.sum(W ** 2, axis=0))
        l1l2_norm += dnrm2(& n_tasks, & beta[j, 0], & inc)

    fval = 0.5 * norm_res2 + lambda_ * l1l2_norm

    return fval


cdef double dual(int n_samples, int n_tasks,
                 double[::1, :] residual, double[::1, :] y, double norm_res2,
                 double dual_scale, double lambda_) nogil:

    cdef:
        int k = 0
        double Ry = 0.
        double alpha = lambda_ / dual_scale

    # Ry = np.dot(residual, y)
    for k in range(n_tasks):
        Ry += ddot(& n_samples, & residual[0, k], & inc, & y[0, k], & inc)

    return Ry * alpha - 0.5 * norm_res2 * alpha ** 2


cdef double dual_gap(int n_samples, int n_features, int n_tasks,
                     double[:, ::1] beta, double[::1, :] residual,
                     double[::1, :] y, double norm_res2, double dual_scale,
                     double lambda_, int * disabled_features) nogil:

    cdef:
        double pobj = primal_value(n_samples, n_features, n_tasks, beta,
                                   norm_res2, lambda_, & disabled_features[0])
        double dobj = dual(n_samples, n_tasks, residual, y, norm_res2,
                           dual_scale, lambda_)

    return pobj - dobj


def bcd_fast(double[::1, :] X, double[::1, :] y, double[:, ::1] beta,
             double[::1, :] residual, double[:, ::1] XTR,
             double[::1] norm_row_XTR, int n_samples, int n_features,
             int n_tasks, double[::1] norm2_X, double lambda_,
             double dual_scale, double norm_res2, int max_iter, int f,
             double tol, int screening, int[::1] disabled_features,
             double[::1] beta_old_g, double[::1] gradient_step,
             int wstr_plus=0):
    """
        Solve the sparse-group-lasso regression with elastic-net
        We minimize
        f(beta) + lambda_1 Omega(beta) + 0.5 * lambda_2 norm(beta, 2)^2
        where f(beta) = 0.5 * norm(y - X beta, 2)^2 and
        Omega(beta) = tau norm(beta, 1) +
                      (1 - tau) * sum_g omega_g * norm(beta_g, 2)
    """

    cdef:
        int i = 0
        int j = 0
        int k = 0
        int inc = 1
        int n_iter = 666
        int n_samples_n_tasks = n_samples * n_tasks
        int n_active_features = n_features

        double gap_t = 1.
        double double_tmp = 0.
        double mu_g = 0.
        double L_g = 0.
        double norm_grad = 0.
        # double norm_res2 = dnrm2(& n_samples_n_tasks, & residual[0, 0], & inc) ** 2
        double r_screen = 666.
        double norm_beta_j = 0.

        double * X_j_ptr = & X[0, 0]

    with nogil:
        if wstr_plus == 0:
            # disabled_features warm_start++
            for j in range(n_features):
                disabled_features[j] = 0

        for n_iter in range(max_iter):

            if f != 0 and n_iter % f == 0:

                # Compute dual point by dual scaling :
                # theta_k = residual / dual_scale
                dual_scale = 0.
                for j in range(n_features):

                    if disabled_features[j] == 1:
                        continue

                    X_j_ptr = & X[0, j]

                    norm_row_XTR[j] = 0.
                    # XTR[g_j] = np.dot(X[:, g_j], residual)
                    for k in range(n_tasks):
                        XTR[j, k] = ddot(& n_samples, X_j_ptr, & inc,
                                         & residual[0, k], & inc)

                        norm_row_XTR[j] += XTR[j, k] ** 2

                    norm_row_XTR[j] = sqrt(norm_row_XTR[j])
                    dual_scale = fmax(dual_scale, norm_row_XTR[j])

                dual_scale = fmax(lambda_, dual_scale)

                norm_res2 = dnrm2(& n_samples_n_tasks, & residual[0, 0], & inc) ** 2

                gap_t = dual_gap(n_samples, n_features, n_tasks, beta,
                                 residual, y, norm_res2, dual_scale, lambda_,
                                 & disabled_features[0])

                if gap_t <= tol:
                    break

                if screening in [GAPSAFE, GAPSAFE_SEQ]:

                    if screening == GAPSAFE_SEQ and n_iter >= 1:
                        pass

                    else:
                        r_screen = sqrt(2 * gap_t) / lambda_
                        for j in range(n_features):

                            if disabled_features[j] == 1:
                                continue

                            X_j_ptr = & X[0, j]
                            r_normX_j = r_screen * sqrt(norm2_X[j])
                            if norm_row_XTR[j] + dual_scale * r_normX_j < dual_scale:
                                # Update residual
                                # residual += np.dot(X[:, j], beta_old - beta[j, :])
                                for k in range(n_tasks):

                                    if beta[j, k] != 0:
                                        daxpy(& n_samples, & beta[j, k], X_j_ptr, & inc,
                                              & residual[0, k], & inc)

                                        beta[j, k] = 0.
                                    # we "set" x_j to zero since it is inactive
                                    XTR[j, k] = 0.

                                disabled_features[j] = 1
                                n_active_features -= 1

            # Bloc-coordinate descent loop
            for j in range(n_features):

                if disabled_features[j] == 1:
                    continue

                if norm2_X[j] == 0:
                    continue

                X_j_ptr = & X[0, j]
                L_g = norm2_X[j]
                # group soft tresholding
                mu_g = lambda_ / L_g
                norm_grad = 0.
                for k in range(n_tasks):

                    beta_old_g[k] = beta[j, k]

                    # XTR[g_j] = np.dot(X[:, g_j], residual)
                    XTR[j, k] = ddot(& n_samples, X_j_ptr, & inc,
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
                    daxpy(& n_samples, & double_tmp, X_j_ptr, & inc,
                          & residual[0, k], & inc)

    return (gap_t, dual_scale, n_iter, n_active_features, norm_res2)

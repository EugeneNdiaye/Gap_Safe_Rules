from __future__ import print_function
# Author: Eugene Ndiaye
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# GAP Safe Screening Rules for Sparse-Group Lasso.
# http://arxiv.org/abs/1602.06225
# firstname.lastname@telecom-paristech.fr

import numpy as np
from sgl_fast import bcd_fast
from sgl_tools import build_lambdas, precompute_norm

NO_SCREENING = 0
GAPSAFE_SEQ = 1
GAPSAFE = 2


def sgl_path(X, y, size_groups, omega, screen, beta_init=None, lambdas=None,
             tau=0.5, lambda2=0, max_iter=30000, f=10, eps=1e-4,
             gap_active_warm_start=False, strong_active_warm_start=False):

    """Compute Sparse-Group-Lasso path with block coordinate descent

    The Sparse-Group-Lasso optimization solves:

    f(beta) + lambda_1 Omega(beta) + 0.5 * lambda_2 norm(beta,2)^2
    where f(beta) = 0.5 * norm(y - X beta,2)^2 and
    Omega(beta) = tau norm(beta,1) + (1 - tau) * sum_g omega_g * norm{beta_g,2}

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    y : ndarray, shape = (n_samples,)
        Target values

    size_groups : ndarray, shape = (n_groups,)
        List of sizes of the different groups
        (n_groups are the number of groups considered).

    omega : ndarray, shape = (n_groups,)
        List of the weight of the different groups: n_groups are the number
        of groups considered.

    screen : integer
        Screening rule to be used: it must be choosen in the following list

        NO_SCREENING = 0 : Standard method

        GAPSAFE_SEQ = 1 : Proposed safe screening rule using duality gap
                                 in a sequential way.

        GAPSAFE = 2 : Proposed safe screening rule using duality gap in both a
                      sequential and dynamic way.

    beta_init : array, shape (n_features, ), optional
        The initial values of the coefficients.

    lambdas : ndarray
        List of lambdas where to compute the models.

    tau : float, optional
        Parameter that make a tradeoff between l1 and l1_2 penalties

    f : float, optional
        The screening rule will be execute at each f pass on the data

    eps : float, optional
        Prescribed accuracy on the duality gap.

    Returns
    -------
    coefs : array, shape (n_features, n_alphas)
        Coefficients along the path.

    dual_gaps : array, shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.

    lambdas : ndarray
        List of lambdas where to compute the models.

    screening_sizes_groups : array, shape (n_alphas,)
        Number of active groups.

    screening_sizes_features : array, shape (n_alphas,)
        Number of active variables.

    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.

    """

    n_groups = len(size_groups)
    g_start = np.cumsum(size_groups, dtype=np.intc) - size_groups[0]

    if lambdas is None:
        lambdas, imax = build_lambdas(X, y, omega, size_groups, g_start)

    # Useful precomputation
    norm_X, norm_X_g, nrm2_y = precompute_norm(X, y, size_groups, g_start)
    tol = eps * nrm2_y  # duality gap tolerance

    n_lambdas = len(lambdas)
    n_samples, n_features = X.shape
    lambda_max = lambdas[0]

    # Fortran-contiguous array are used to avoid useless copy of the data.
    X = np.asfortranarray(X)
    y = np.asfortranarray(y)
    size_groups = np.asfortranarray(size_groups, dtype=np.intc)
    norm2_X = np.asfortranarray(norm_X ** 2)
    norm2_X_g = np.asfortranarray(norm_X_g ** 2)
    omega = np.asfortranarray(omega)
    g_start = np.asfortranarray(g_start, dtype=np.intc)

    if beta_init is None:
        beta_init = np.zeros(n_features, order='F')
    else:
        beta_init = np.asfortranarray(beta_init)

    coefs = np.zeros((n_features, n_lambdas), order='F')
    residual = np.asfortranarray(y - np.dot(X, beta_init))
    XTR = np.asfortranarray(np.dot(X.T, residual))
    dual_scale = lambda_max  # good iif beta_init = 0

    dual_gaps = np.ones(n_lambdas)
    screening_sizes_features = np.zeros(n_lambdas)
    screening_sizes_groups = np.zeros(n_lambdas)
    n_iters = np.zeros(n_lambdas)

    disabled_features = np.zeros(n_features, dtype=np.intc, order='F')
    disabled_groups = np.zeros(n_groups, dtype=np.intc, order='F')

    active_ws = False
    strong_ws = False

    for t in range(n_lambdas):

        if t == 0:
            lambda_prec = lambda_max
        else:
            lambda_prec = lambdas[t - 1]

        if strong_active_warm_start:
            strong_ws = True

        if gap_active_warm_start:
            active_ws = (screening_sizes_features[t] < n_features or
                         screening_sizes_groups[t] < n_groups)

        if strong_ws or active_ws:

            bcd_fast(X, y, beta_init, XTR, residual, dual_scale, omega,
                     n_samples, n_features, n_groups, size_groups, g_start,
                     norm2_X, norm2_X_g, nrm2_y, tau, lambdas[t],
                     lambda_prec, lambda2, max_iter, f, tol,
                     screen, disabled_features, disabled_groups,
                     wstr_plus=active_ws, strong_warm_start=strong_ws)

        model = bcd_fast(X, y, beta_init, XTR, residual, dual_scale, omega,
                         n_samples, n_features, n_groups, size_groups, g_start,
                         norm2_X, norm2_X_g, nrm2_y, tau, lambdas[t],
                         lambda_prec, lambda2, max_iter, f, tol,
                         screen, disabled_features, disabled_groups,
                         wstr_plus=0, strong_warm_start=0)

        dual_scale, dual_gaps[t], n_active_groups, n_active_features, \
            n_iters[t] = model

        coefs[:, t] = beta_init.copy()

        if abs(dual_gaps[t]) > tol:
            print("Warning did not converge ... t = %s gap = %s eps = %s n_iter = %s" %
                  (t, dual_gaps[t], eps, n_iters[t]))

    return (coefs, dual_gaps, lambdas, screening_sizes_groups,
            screening_sizes_features, n_iters)


if __name__ == '__main__':

    from sgl_tools import generate_data
    import time

    n_samples = 50
    n_features = 800
    size_group = 40  # all groups have size = size_group
    delta = 3
    tau = .34

    n_groups = n_features / size_group
    size_groups = size_group * np.ones(n_groups, order='F', dtype=np.intc)
    omega = np.ones(n_groups)
    groups = np.arange(n_features) // size_group
    group_labels = [np.where(groups == i)[0] for i in np.unique(groups)]
    X, y = generate_data(n_samples, n_features, size_groups, rho=0.4)

    tic = time.time()
    gaps = sgl_path(X, y, size_groups, omega, screen=2, tau=tau, max_iter=1e5,
                    eps=1e-14, strong_active_warm_start=True)[1]
    print("time = ", time.time() - tic)

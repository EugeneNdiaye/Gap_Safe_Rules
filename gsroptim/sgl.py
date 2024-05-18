# Author: Eugene Ndiaye
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# GAP Safe Screening Rules for Sparse-Group Lasso.
# http://arxiv.org/abs/1602.06225
# firstname.lastname@telecom-paristech.fr

import numpy as np
import warnings

from sklearn.exceptions import ConvergenceWarning

from .sgl_fast import bcd_fast
from gsroptim.sgl_tools import build_lambdas, precompute_norm

NO_SCREENING = 0
GAPSAFE_SEQ = 1
GAPSAFE = 2


def sgl_path(X, y, size_groups, omega, lambdas=None, tau=0.5, lambda2=0,
             beta_init=None, screen=GAPSAFE, max_iter=30000, f=10, eps=1e-4,
             gap_active_warm_start=False, strong_active_warm_start=True,
             verbose=False):
    """Compute Sparse-Group-Lasso path with block coordinate descent

    The Sparse-Group Lasso formulation reads:

    f(beta) + lambda_1 Omega(beta) + 0.5 * lambda_2 norm(beta,2)^2
    where f(beta) = 0.5 * norm(y - X beta,2)^2 and
    Omega(beta) = tau norm(beta,1) + (1 - tau) * sum_g omega_g * norm{beta_g,2}
    where g belongs to a group structure.

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
    betas : array, shape (n_features, n_lambdas)
        Coefficients along the path.

    gaps : array, shape (n_lambdas,)
        The dual gaps at the end of the optimization for each lambda.

    n_iters : array-like, shape (n_lambdas,)
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.

    screening_sizes_groups : array, shape (n_lambdas,)
        Number of active groups.

    screening_sizes_features : array, shape (n_lambdas,)
        Number of active variables.

    """

    n_groups = len(size_groups)
    g_start = np.cumsum(size_groups, dtype=np.intc) - size_groups[0]

    if lambdas is None:
        lambdas, _ = build_lambdas(X, y, omega, size_groups, g_start)

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

    betas = np.zeros((n_features, n_lambdas), order='F')
    residual = np.asfortranarray(y - np.dot(X, beta_init))
    XTR = np.asfortranarray(np.dot(X.T, residual))
    dual_scale = lambda_max  # good iif beta_init = 0

    gaps = np.ones(n_lambdas)
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

        dual_scale, gaps[t], _, _, n_iters[t] = model

        betas[:, t] = beta_init.copy()

        if verbose and abs(gaps[t]) > tol[t]:
            warnings.warn('Solver did not converge after '
                          '%i iterations: dual gap: %.3e'
                          % (max_iter, gaps[t]), ConvergenceWarning)

    # return (betas, gaps, n_iters, screening_sizes_groups,
    #         screening_sizes_features)
    return (betas, gaps, n_iters, disabled_groups, disabled_features)

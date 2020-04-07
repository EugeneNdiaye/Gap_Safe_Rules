# Author: Eugene Ndiaye
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# GAP Safe Screening Rules for Sparse-Group Lasso.
# http://arxiv.org/abs/1602.06225
# firstname.lastname@telecom-paristech.fr

import warnings
import numpy as np

from sklearn.exceptions import ConvergenceWarning

from .bcd_multitask_lasso_fast import bcd_fast

NO_SCREENING = 0
GAPSAFE_SEQ = 1
GAPSAFE_DYN = 2


def multitask_lasso_path(X, y, lambdas, beta_init=None, screen=GAPSAFE_DYN,
                         eps=1e-4, max_iter=int(1e7), f=10,
                         gap_active_warm_start=False,
                         strong_active_warm_start=True, verbose=False):
    """Compute multitask Lasso path with block coordinate descent

    The multitask Lasso optimization solves:

    f(beta) + lambda_1 Omega(beta) + 0.5 * lambda_2 norm(beta,2)^2
    where f(beta) = 0.5 * norm(y - X beta, fro)^2 and
    Omega(beta) = sum_j norm{beta[j,:], 2}

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    y : ndarray, shape = (n_samples, n_tasks)
        Target values

    size_groups : ndarray, shape = (n_groups,)
        List of sizes of the different groups
        (n_groups are the number of groups considered).

    screen : integer
        Screening rule to be used: it must be choosen in the following list

        NO_SCREENING = 0 : Standard method

        GAPSAFE_SEQ = 1 : Proposed safe screening rule using duality gap
                                 in a sequential way.

        GAPSAFE_DYN = 2 : Proposed safe screening rule using duality gap
                                 in both a sequential and dynamic way.

    beta_init : array, shape (n_features, n_tasks), optional
        The initial values of the coefficients.

    lambdas : ndarray
        List of lambdas where to compute the models.

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

    lambdas : ndarray
        List of lambdas where to compute the models.

    screening_sizes_groups : array, shape (n_lambdas,)
        Number of active groups.

    screening_sizes_features : array, shape (n_lambdas,)
        Number of active variables.

    n_iters : array-like, shape (n_lambdas,)
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.

    """
    n_lambdas = len(lambdas)
    n_samples, n_features = X.shape
    n_tasks = y.shape[1]

    # Fortran-contiguous array are used to avoid useless copy of the data.
    X = np.asfortranarray(X)
    y = np.asfortranarray(y)

    active_warm_start = strong_active_warm_start or gap_active_warm_start
    run_active_warm_start = True

    norm2_X = np.sum(X ** 2, axis=0)
    nrm2_y = np.linalg.norm(y) ** 2
    tol = eps * nrm2_y
    norm_res2 = nrm2_y

    if beta_init is None:
        beta_init = np.zeros((n_features, n_tasks), order='C')
    else:
        beta_init = np.asarray(beta_init, order='C')

    betas = np.zeros((n_features, n_tasks, n_lambdas), order='C')
    residual = np.asfortranarray(y - np.dot(X, beta_init))
    disabled_features = np.zeros(n_features, dtype=np.intc, order='F')
    dual_scale = lambdas[0]

    gaps = np.ones(n_lambdas)
    n_iters = np.zeros(n_lambdas)
    n_active_features = np.zeros(n_lambdas)

    XTR = np.asarray(np.dot(X.T, residual), order='C')
    norm_row_XTR = np.asfortranarray(np.linalg.norm(XTR, axis=1))
    beta_old_g = np.zeros(n_tasks, order='F')
    gradient_step = np.zeros(n_tasks, order='F')

    for t in range(n_lambdas):

        if active_warm_start and t != 0:

            if strong_active_warm_start:
                disabled_features = (norm_row_XTR < 2. * lambdas[t] -
                                     lambdas[t - 1]).astype(np.intc)

            if gap_active_warm_start:
                run_active_warm_start = n_active_features[t] < n_features

            if run_active_warm_start:

                model = bcd_fast(X, y, beta_init, residual, XTR, norm_row_XTR,
                                 n_samples, n_features, n_tasks, norm2_X,
                                 lambdas[t], dual_scale, norm_res2, max_iter,
                                 f, tol, screen, disabled_features, beta_old_g,
                                 gradient_step, wstr_plus=1)
                norm_res2 = model[-1]

        model = bcd_fast(X, y, beta_init, residual, XTR, norm_row_XTR,
                         n_samples, n_features, n_tasks, norm2_X, lambdas[t],
                         dual_scale, norm_res2, max_iter, f, tol, screen,
                         disabled_features, beta_old_g, gradient_step,
                         wstr_plus=0)

        gaps[t], dual_scale, n_iters[t], n_active_features[t],\
            norm_res2 = model
        betas[:, :, t] = beta_init.copy()

        if verbose and abs(gaps[t]) > tol:
            warnings.warn('Solver did not converge after '
                          '%i iterations: dual gap: %.3e'
                          % (max_iter, gaps[t]), ConvergenceWarning)

    return betas, gaps, n_iters, n_active_features

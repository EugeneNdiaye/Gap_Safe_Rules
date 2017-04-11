# Author: Eugene Ndiaye
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# GAP Safe Screening Rules for Sparse-Group Lasso.
# http://arxiv.org/abs/1602.06225
# firstname.lastname@telecom-paristech.fr

import numpy as np
from bcd_multitask_lasso_fast import bcd_fast

NO_SCREENING = 0

DGST3 = 1

GAPSAFE_SEQ = 2
GAPSAFE = 3

GAPSAFE_SEQ_pp = 4
GAPSAFE_pp = 5

STRONG_RULE = 10

STRONG_GAP_SAFE = 666


def multitask_lasso_path(X, y, screen=NO_SCREENING, beta_init=None,
                         lambdas=None, max_iter=100, f=10, eps=1e-4,
                         j_star=0, wstr_plus=False):
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

        STATIC_SAFE = 1 : Use static safe screening rule
            cf. El Ghaoui, L., Viallon, V., and Rabbani, T.
            "Safe feature elimination in sparse supervised learning".
            J. Pacific Optim., 2012.

        DST3 = 3 : Adaptation of the DST3 safe screening rules
            cf.  Xiang, Z. J., Xu, H., and Ramadge, P. J.,
            "Learning sparse representations of high dimensional data on large
            scale dictionaries". NIPS 2011

            cf. Bonnefoy, A., Emiya, V., Ralaivola, L., and Gribonval, R.
            "Dynamic Screening: Accelerating First-Order Al-
            gorithms for the Lasso and Group-Lasso".
            IEEE Trans. Signal Process., 2015.

        GAPSAFE_SEQ = 4 : Proposed safe screening rule using duality gap
                                 in a sequential way.

        GAPSAFE = 5 : Proposed safe screening rule using duality gap in both a
                      sequential and dynamic way.

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
    n_lambdas = len(lambdas)
    n_samples, n_features = X.shape
    n_tasks = y.shape[1]

    # Fortran-contiguous array are used to avoid useless copy of the data.
    X = np.asfortranarray(X)
    y = np.asfortranarray(y)

    norm2_X = np.sum(X ** 2, axis=0)
    tol = eps * np.linalg.norm(y) ** 2

    if beta_init is None:
        beta_init = np.zeros((n_features, n_tasks), order='C')
    else:
        beta_init = np.asarray(beta_init, order='C')

    coefs = np.zeros((n_features, n_tasks, n_lambdas), order='C')
    residual = np.asfortranarray(y - np.dot(X, beta_init))
    disabled_features = np.zeros(n_features, dtype=np.intc, order='F')
    dual_scale = lambdas[0]

    dual_gaps = np.ones(n_lambdas)
    n_iters = np.zeros(n_lambdas)
    n_active_features = np.zeros(n_lambdas)

    XTR = np.asarray(np.dot(X.T, residual), order='C')
    norm_row_XTR = np.asfortranarray(np.linalg.norm(XTR, axis=1))

    if screen == DGST3:

        # print "j_star = ", j_star
        n_DGST3 = [X[:, j_star] * np.dot(X[:, j_star].T, y[:, k]) / lambdas[0]
                   for k in range(n_tasks)]
        n_DGST3 = np.array(n_DGST3).T
        norm2_n_DGST3 = np.linalg.norm(n_DGST3, ord='fro') ** 2
        nTy_DGST3 = [np.dot(n_DGST3[:, k], y[:, k]) for k in range(n_tasks)]
        n_DGST3 = np.asfortranarray(n_DGST3)
        nTy_DGST3 = np.asfortranarray(nTy_DGST3)

    else:
        n_DGST3 = np.empty((1, 1), order='F')
        norm2_n_DGST3 = 1.
        nTy_DGST3 = np.empty(1, order='F')

    for t in range(n_lambdas):

        if t == 0:
            lambda_prec = lambdas[0]
        else:
            lambda_prec = lambdas[t - 1]

        if screen == STRONG_GAP_SAFE:

            # TODO: cythonize this part
            # reset the active set
            disabled_features = np.zeros(n_features, dtype=np.intc, order='F')

            # Compute the strong active set
            mask = np.where(norm_row_XTR < 2 * lambdas[t] - lambda_prec)[0]
            beta_init[mask, :] = 0.
            disabled_features[mask] = 1

            model = bcd_fast(X, y, beta_init, residual, XTR, norm_row_XTR,
                             n_DGST3, norm2_n_DGST3, nTy_DGST3,
                             n_samples, n_features, n_tasks,
                             norm2_X, lambdas[t], lambdas[0], lambda_prec,
                             dual_scale, max_iter, f, tol, screen,
                             disabled_features, wstr_plus=1)

        model = bcd_fast(X, y, beta_init, residual, XTR, norm_row_XTR,
                         n_DGST3, norm2_n_DGST3, nTy_DGST3,
                         n_samples, n_features, n_tasks,
                         norm2_X, lambdas[t], lambdas[0], lambda_prec,
                         dual_scale, max_iter,
                         f, tol, screen, disabled_features, wstr_plus=0)

        dual_gaps[t], dual_scale, n_iters[t], n_active_features[t] = model
        coefs[:, :, t] = beta_init.copy()

        if t == 0 and screen != NO_SCREENING:
            n_active_features[0] = 0

        if wstr_plus and t < n_lambdas - 1 and t != 0 and \
           n_active_features[t] < n_features:

            bcd_fast(X, y, beta_init, residual, XTR, norm_row_XTR,
                     n_DGST3, norm2_n_DGST3, nTy_DGST3,
                     n_samples, n_features, n_tasks,
                     norm2_X, lambdas[t + 1], lambdas[0], lambda_prec,
                     dual_scale, max_iter,
                     f, tol, screen, disabled_features, wstr_plus=1)

        if abs(dual_gaps[t]) > tol:
            print("Warning did not converge ... t = %s gap = %s tol = %s n_iter = %s" %
                  (t, dual_gaps[t], tol, n_iters[t]))

    return (coefs, dual_gaps, n_iters, n_active_features)

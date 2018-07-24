# Author: Eugene Ndiaye
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# GAP Safe Screening Rules for Sparse-Group Lasso.
# http://arxiv.org/abs/1602.06225
# firstname.lastname@telecom-paristech.fr

import numpy as np

from gsr.bcd_multitask_lasso_fast import bcd_fast

NO_SCREENING = 0
GAPSAFE_SEQ = 1
GAPSAFE_DYN = 2


def multitask_lasso_path(X, y, screen=GAPSAFE_DYN, beta_init=None,
                         lambdas=None, max_iter=100, f=10, eps=1e-4,
                         gap_active_warm_start=False,
                         strong_active_warm_start=False):
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

    coefs = np.zeros((n_features, n_tasks, n_lambdas), order='C')
    residual = np.asfortranarray(y - np.dot(X, beta_init))
    disabled_features = np.zeros(n_features, dtype=np.intc, order='F')
    dual_scale = lambdas[0]

    dual_gaps = np.ones(n_lambdas)
    n_iters = np.zeros(n_lambdas)
    n_active_features = np.zeros(n_lambdas)

    XTR = np.asarray(np.dot(X.T, residual), order='C')
    norm_row_XTR = np.asfortranarray(np.linalg.norm(XTR, axis=1))
    beta_old_g = np.zeros(n_tasks, order='F')
    gradient_step = np.zeros(n_tasks, order='F')

    for t in range(n_lambdas):

        if active_warm_start and t != 0:

            if strong_active_warm_start:
                disabled_features = (norm_row_XTR < 2. * lambdas[t] - lambdas[t - 1]).astype(np.intc)

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

        dual_gaps[t], dual_scale, n_iters[t], n_active_features[t],\
            norm_res2 = model
        coefs[:, :, t] = beta_init.copy()

        if abs(dual_gaps[t]) > tol:
            print("Warning did not converge ... t = %s gap = %s tol = %s n_iter = %s" %
                  (t, dual_gaps[t], tol, n_iters[t]))

    return (coefs, dual_gaps, n_iters, n_active_features)


if __name__ == '__main__':

    from sklearn.datasets import make_regression
    from sklearn.linear_model import lasso_path
    import time

    n_samples, n_features, n_tasks = (200, 500, 75)
    # generate dataset
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           n_targets=n_tasks, random_state=42)

    # tmp_y = y.copy()
    # y = y[:, None]
    # parameters
    lambda_max = np.max(np.sqrt(np.sum(np.dot(X.T, y) ** 2, axis=1)))
    n_lambdas = 100
    lambda_ratio = 1e-3 ** (1. / (n_lambdas - 1))
    lambdas = np.array([lambda_max * (lambda_ratio ** i)
                        for i in range(0, n_lambdas)])

    # lambdas = np.array([lambda_max / 20.])
    tol = 1e-12

    tic = time.time()
    beta, gap, n_iters, n_active_features = \
        multitask_lasso_path(X, y, lambdas=lambdas, eps=tol, f=10,
                             max_iter=5000, screen=GAPSAFE_DYN,
                             strong_active_warm_start=True)
    print "our time = ", time.time() - tic, np.max(gap)

    # tic = time.time()
    # alphas, coefs, gaps = lasso_path(X, y, alphas=lambdas / n_samples,
    #                                  fit_intercept=False, normalize=False,
    #                                  tol=tol, max_iter=5000)
    # print "our sk = ", time.time() - tic, np.max(gaps)

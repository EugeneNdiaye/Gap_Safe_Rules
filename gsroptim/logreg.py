# Author: Eugene Ndiaye
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# Gap Safe screening rules for sparsity enforcing penalties.
# https://arxiv.org/abs/1611.05780
# firstname.lastname@telecom-paristech.fr

import numpy as np
import scipy as sp
from .cd_logreg_fast import cd_logreg

NO_SCREENING = 0
GAPSAFE_SEQ = 1
GAPSAFE = 2


def logreg_path(X, y, lambdas, beta_init=None, eps=1e-4, max_iter=3000, f=10,
                screening=GAPSAFE, gap_active_warm_start=False,
                strong_active_warm_start=True):
    """Compute l1-regularized logistic regression path with coordinate descent

    We solve:

    argmin_{beta} sum_{i=1}^{n} f_i(dot(x_{i}, beta)) + lambda * norm(beta, 1)
    where f_i(z) = -y_i * z + log(1 + exp(z)).

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    y : ndarray, shape = (n_samples,)
        Target values : the label must be 0 and 1.

    lambdas : ndarray
        List of lambdas where to compute the models.

    beta_init : array, shape (n_features, ), optional
        The initial values of the coefficients.

    screening : integer
        Screening rule to be used: it must be choosen in the following list

        NO_SCREENING = 0: Standard method

        GAPSAFE_SEQ = 1: Proposed safe screening rule using duality gap
                          in a sequential way: Gap Safe (Seq.)

        GAPSAFE = 2: Proposed safe screening rule using duality gap in both a
                      sequential and dynamic way.: Gap Safe (Seq. + Dyn)

    strong_active_warm_start : Proposed safe screening rule using duality gap
                             in a sequential way along with strong warm start
                             strategies: Gap Safe (Seq. + strong warm start)

    gap_active_warm_start : Proposed safe screening rule using duality gap
                         in both a sequential and dynamic way along with
                         active warm start strategies:
                         Gap Safe (Seq. + Dyn + active warm start).

    f : float, optional
        The screening rule will be execute at each f pass on the data

    eps : float, optional
        Prescribed accuracy on the duality gap.

    Returns
    -------
    betas : array, shape (n_features, n_lambdas)
        Coefficients along the path.

    dual_gaps : array, shape (n_lambdas,)
        The dual gaps at the end of the optimization for each lambda.

    lambdas : ndarray
        List of lambdas where to compute the models.

    n_iters : array-like, shape (n_lambdas,)
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.

    n_actives_features : array, shape (n_lambdas,)
        Number of active variables.

    """

    if type(lambdas) != np.ndarray:
        lambdas = np.array([lambdas])

    n_lambdas = len(lambdas)

    n_samples, n_features = X.shape
    n_1 = np.sum(y == 1)
    n_0 = n_samples - n_1
    tol = eps * max(1, min(n_1, n_0)) / float(n_samples)

    active_warm_start = strong_active_warm_start or gap_active_warm_start
    run_active_warm_start = True

    betas = np.zeros((n_features, n_lambdas))
    disabled_features = np.zeros(n_features, dtype=np.intc, order='F')
    gaps = np.ones(n_lambdas)
    n_iters = np.zeros(n_lambdas)
    n_active_features = np.zeros(n_lambdas)

    y = np.asfortranarray(y, dtype=float)
    sparse = sp.sparse.issparse(X)

    if sparse:
        X_ = None
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr
        norm_X2 = sp.sparse.linalg.norm(X, axis=0) ** 2
    else:
        X_ = np.asfortranarray(X, dtype=float)
        X_data = None
        X_indices = None
        X_indptr = None
        norm_X2 = np.sum(X_ ** 2, axis=0)

    if beta_init is None:
        beta_init = np.zeros(n_features, dtype=float, order='F')
        norm1_beta = 0.
        p_obj = n_samples * np.log(2)
        Xbeta = np.zeros(n_samples, dtype=float, order='F')
        residual = np.asfortranarray(y - 0.5)
    else:
        norm1_beta = np.linalg.norm(beta_init, ord=1)
        Xbeta = np.asfortranarray(X.dot(beta_init))
        exp_Xbeta = np.asfortranarray(np.exp(Xbeta))
        residual = np.asfortranarray(y - exp_Xbeta / (1. + exp_Xbeta))
        yTXbeta = np.dot(y, Xbeta)
        log_term = np.sum(np.log1p(exp_Xbeta))
        p_obj = -yTXbeta + log_term + lambdas[0] * norm1_beta

    XTR = np.asfortranarray(X.T.dot(residual))
    dual_scale = lambdas[0]  # True only if beta lambdas[0] ==  lambda_max

    Hessian = np.zeros(n_features, dtype=float, order='F')
    Xbeta_next = np.zeros(n_samples, dtype=float, order='F')

    norm_X2 = np.asfortranarray(norm_X2)

    for t in range(n_lambdas):

        if active_warm_start and t != 0:

            if strong_active_warm_start:
                disabled_features = (np.abs(XTR) < 2. * lambdas[t] -
                                     lambdas[t - 1]).astype(np.intc)

            if gap_active_warm_start:
                run_active_warm_start = n_active_features[t] < n_features

            if run_active_warm_start:

                # solve the problem restricted to the strong active set
                _, p_obj, norm1_beta, _, _, _ =\
                    cd_logreg(X_, X_data, X_indices, X_indptr, y, beta_init,
                              XTR, Xbeta, Hessian, Xbeta_next, residual,
                              disabled_features, norm_X2, p_obj, norm1_beta,
                              lambdas[t], tol, dual_scale, max_iter, f,
                              screening, wstr_plus=1, sparse=sparse)

        gaps[t], p_obj, norm1_beta, dual_scale, n_iters[t],\
            n_active_features[t] = \
            cd_logreg(X_, X_data, X_indices, X_indptr, y, beta_init, XTR,
                      Xbeta, Hessian, Xbeta_next, residual, disabled_features,
                      norm_X2, p_obj, norm1_beta, lambdas[t], tol, dual_scale,
                      max_iter, f, screening, wstr_plus=0, sparse=sparse)

        betas[:, t] = beta_init.copy()

        if abs(gaps[t]) > tol:

            print("warning: did not converge, t = ", t)
            print("gap = ", gaps[t], "eps = ", eps)

    return betas, gaps, n_iters, n_active_features

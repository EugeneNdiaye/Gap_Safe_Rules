import numpy as np
from numpy.linalg import norm
from cd_lasso_fast import cd_lasso

NO_SCREENING = 0

STATIC = 1
DST3 = 2

GAPSAFE_SEQ = 3
GAPSAFE = 4

GAPSAFE_SEQ_pp = 5
GAPSAFE_pp = 6


def lasso_path(X, y, lambdas, eps=1e-4, max_iter=3000, f=10, screening=1,
               warm_start_plus=False, j_star=0):

    """Compute Lasso path with coordinate descent

    The Lasso optimization solves:

    argmin_{beta} 0.5 * norm(y - X beta, 2)^2 + lambda * norm(beta, 1)

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication.

    y : ndarray, shape = (n_samples,)
        Target values

    screen : integer
        Screening rule to be used: it must be choosen in the following list

        NO_SCREENING = 0: Standard method

        STATIC_SAFE = 1: Use static safe screening rule
            cf. El Ghaoui, L., Viallon, V., and Rabbani, T.
            "Safe feature elimination in sparse supervised learning".
            J. Pacific Optim., 2012.

        DST3 = 2: Adaptation of the DST3 safe screening rules
            cf.  Xiang, Z. J., Xu, H., and Ramadge, P. J.,
            "Learning sparse representations of high dimensional data on large
            scale dictionaries". NIPS 2011

        GAPSAFE_SEQ = 3: Proposed safe screening rule using duality gap
                          in a sequential way: Gap Safe (Seq.)

        GAPSAFE = 4: Proposed safe screening rule using duality gap in both a
                      sequential and dynamic way.: Gap Safe (Seq. + Dyn)

        GAPSAFE_SEQ_pp = 5: Proposed safe screening rule using duality gap
                             in a sequential way along with active warm start
                             strategies: Gap Safe (Seq. + active warm start)

        GAPSAFE_pp = 6: Proposed safe screening rule using duality gap
                         in both a sequential and dynamic way along with
                         active warm start strategies:
                         Gap Safe (Seq. + Dyn + active warm start).

    beta_init : array, shape (n_features, ), optional
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

    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.

    n_active_features : array, shape (n_alphas,)
        Number of active variables.

    """

    if type(lambdas) != np.ndarray:
        lambdas = np.array([lambdas])

    n_lambdas = len(lambdas)

    n_samples, n_features = X.shape
    betas = np.zeros((n_lambdas, n_features))
    beta_init = np.zeros(n_features, dtype=float, order='F')
    disabled_features = np.zeros(n_features, dtype=np.intc, order='F')
    gaps = np.ones(n_lambdas)
    n_iters = np.zeros(n_lambdas)
    n_active_features = np.zeros(n_lambdas)
    norm_X2 = np.sum(X ** 2, axis=0)

    residual = np.asfortranarray(y - np.dot(X, beta_init))
    nrm2_y = norm(y) ** 2
    XTR = np.asfortranarray(np.dot(X.T, residual))

    # True only if beta lambdas[0] ==  lambda_max
    lambda_max = dual_scale = lambdas[0]

    # Fortran-contiguous array are used to avoid useless copy of the data.
    X = np.asfortranarray(X)
    y = np.asfortranarray(y)
    norm_X2 = np.asfortranarray(norm_X2)

    if screening == STATIC:
        XTy = np.asfortranarray(np.dot(X.T, y))
    else:
        XTy = None

    for t in range(n_lambdas):

        gaps[t], dual_scale, n_iters[t], n_active_features[t] = \
            cd_lasso(X, y, beta_init, XTR, XTy, residual, disabled_features,
                     nrm2_y, norm_X2, lambdas[t], lambda_max, dual_scale,
                     eps, max_iter, f, screening, j_star, wstr_plus=0)

        betas[t, :] = beta_init.copy()
        if t == 0 and screening != NO_SCREENING:
            n_active_features[0] = 0

        if warm_start_plus and t < n_lambdas - 1 and t != 0 and \
           n_active_features[t] < n_features:

            cd_lasso(X, y, beta_init, XTR, XTy, residual, disabled_features,
                     nrm2_y, norm_X2, lambdas[t + 1], lambda_max, dual_scale,
                     eps, max_iter, f, screening=screening, j_star=j_star,
                     wstr_plus=1)

        if gaps[t] > eps * nrm2_y:

            print "warning: did not converge, t = ", t,
            print "gap = ", gaps[t], "eps = ", eps

    return betas, gaps, n_iters, n_active_features

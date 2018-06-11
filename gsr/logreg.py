import numpy as np

from numpy.linalg import norm

from gsr.cd_logreg_fast import cd_logreg

NO_SCREENING = 0

GAPSAFE_SEQ = 1
GAPSAFE = 2

GAPSAFE_SEQ_pp = 3
GAPSAFE_pp = 4

STRONG_RULE = 10

SAFE_STRONG_RULE = 666


def logreg_path(X, y, lambdas, eps=1e-4, max_iter=3000, f=10, screening=0,
                warm_start_plus=False):

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
        Target values

    screen : integer
        Screening rule to be used: it must be choosen in the following list

        NO_SCREENING = 0: Standard method

        GAPSAFE_SEQ = 1: Proposed safe screening rule using duality gap
                          in a sequential way: Gap Safe (Seq.)

        GAPSAFE = 2: Proposed safe screening rule using duality gap in both a
                      sequential and dynamic way.: Gap Safe (Seq. + Dyn)

        GAPSAFE_SEQ_pp = 3: Proposed safe screening rule using duality gap
                             in a sequential way along with active warm start
                             strategies: Gap Safe (Seq. + active warm start)

        GAPSAFE_pp = 4: Proposed safe screening rule using duality gap
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

    n_actives_features : array, shape (n_alphas,)
        Number of active variables.

    """

    if type(lambdas) != np.ndarray:
        lambdas = np.array([lambdas])

    n_lambdas = len(lambdas)

    n_samples, n_features = X.shape
    n_1 = np.sum(y == 1)
    n_0 = n_samples - n_1
    tol = eps * max(1, min(n_1, n_0)) / float(n_samples)

    betas = np.zeros((n_lambdas, n_features))
    beta_init = np.zeros(n_features, dtype=float, order='F')
    disabled_features = np.zeros(n_features, dtype=np.intc, order='F')
    gaps = np.ones(n_lambdas)
    n_iters = np.zeros(n_lambdas)
    n_active_features = np.zeros(n_lambdas)
    norm_X2 = np.sum(X ** 2, axis=0)

    Xbeta = np.asfortranarray(np.dot(X, beta_init))
    exp_Xbeta = np.asfortranarray(np.exp(Xbeta))
    residual = np.asfortranarray(y - exp_Xbeta / (1. + exp_Xbeta))

    nrm2_y = norm(y) ** 2
    XTR = np.asfortranarray(np.dot(X.T, residual))
    dual_scale = lambdas[0]  # True only if beta lambdas[0] ==  lambda_max

    # Fortran-contiguous array are used to avoid useless copy of the data.
    X = np.asfortranarray(X, dtype=float)
    y = np.asfortranarray(y, dtype=float)
    norm_X2 = np.asfortranarray(norm_X2)

    for t in range(n_lambdas):

        if t == 0:
            lambda_prec = lambdas[0]
        else:
            lambda_prec = lambdas[t - 1]

        if screening == SAFE_STRONG_RULE:

            #TODO: cythonize this part
            # reset the active set
            disabled_features = np.zeros(n_features, dtype=np.intc, order='F')

            # Compute the strong active set
            # check the validity of this rule
            mask = np.where(np.abs(XTR) < 2 * lambdas[t] - lambda_prec)[0]
            beta_init[mask] = 0.
            disabled_features[mask] = 1

            # solve the problem restricted to the strong active set
            cd_logreg(X, y, beta_init, XTR, Xbeta, exp_Xbeta, residual,
                      disabled_features, nrm2_y, norm_X2,
                      lambdas[t], lambda_prec, tol, dual_scale, max_iter,
                      f, screening, wstr_plus=1)

        gaps[t], dual_scale, n_iters[t], n_active_features[t] = \
            cd_logreg(X, y, beta_init, XTR, Xbeta, exp_Xbeta, residual,
                      disabled_features, nrm2_y, norm_X2,
                      lambdas[t], lambda_prec, tol, dual_scale, max_iter, f,
                      screening, wstr_plus=0)

        betas[t, :] = beta_init.copy()
        if t == 0 and screening != NO_SCREENING:
            n_active_features[0] = 0

        if warm_start_plus and t < n_lambdas - 1 and t != 0 and \
           n_active_features[t] < n_features:

            cd_logreg(X, y, beta_init, XTR, Xbeta, exp_Xbeta, residual,
                      disabled_features, nrm2_y, norm_X2,
                      lambdas[t + 1], lambda_prec, tol, dual_scale,
                      max_iter, f, screening=screening, wstr_plus=1)

        if abs(gaps[t]) > tol:

            print("warning: did not converge, t = %d" % t)
            print("gap = ", gaps[t], "eps = ", eps)

    return betas, gaps, n_iters, n_active_features

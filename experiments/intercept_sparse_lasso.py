import numpy as np
from numpy.linalg import norm
from intercept_sparse_cd_lasso_fast import cd_lasso
from intercept_sparse_cd_lasso_fast import matrix_column_norm
# from scdl_fast import cd_lasso
from scipy.sparse import csc_matrix
from sklearn.linear_model import lasso_path

NO_SCREENING = 0

STATIC = 1
DST3 = 2

GAPSAFE_SEQ = 3
GAPSAFE = 4

GAPSAFE_SEQ_pp = 5
GAPSAFE_pp = 6

STRONG_RULE = 10
EDPP = 11

STRONG_GAP_SAFE = 666


def sp_lasso_path(X, y, lambdas, eps=1e-4, max_iter=3000, f=10, screening=1,
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
    intercept = np.zeros(n_lambdas)
    disabled_features = np.zeros(n_features, dtype=np.intc, order='F')
    gaps = np.ones(n_lambdas)
    n_iters = np.zeros(n_lambdas)
    n_active_features = np.zeros(n_lambdas)

    X_data = np.asfortranarray(X.data, dtype=float)
    X_indices = np.asfortranarray(X.indices, dtype=np.intc)
    X_indptr = np.asfortranarray(X.indptr, dtype=np.intc)

    # We center the data for the intercept
    X_mean = np.asfortranarray(X.mean(axis=0)).ravel()
    y_mean = y.mean()
    # TODO: avoid this modification of y
    y -= y_mean

    norm_Xcent = np.zeros(n_features, dtype=float, order='F')
    matrix_column_norm(n_samples, n_features, X_data, X_indices, X_indptr,
                       norm_Xcent, X_mean, center=1)

    residual = np.asfortranarray(y - X.dot(beta_init) + X_mean.dot(beta_init))
    nrm2_y = norm(y) ** 2
    XTR = np.asfortranarray(X.T.dot(residual))

    tol = eps * nrm2_y  # duality gap tolerance

    # True only if beta lambdas[0] ==  lambda_max
    lambda_max = dual_scale = lambdas[0]

    # Fortran-contiguous array are used to avoid useless copy of the data.
    # X = np.asfortranarray(X)
    # y = np.asfortranarray(y)

    if screening == STATIC:
        # XTy = np.asfortranarray(X.T.dot(y))
        XTy = XTR.copy()  # assume that beta_init=0 for the benchmark
    else:
        XTy = None

    if screening == EDPP:
        X_j_star = X[:, j_star].toarray().ravel()
        v1 = np.asfortranarray(X_j_star * np.sign(np.dot(X_j_star, y)))
        # import pdb; pdb.set_trace()
    else:
        v1 = None

    for t in range(n_lambdas):

        if t == 0:
            lambda_prec = lambda_max
        else:
            lambda_prec = lambdas[t - 1]

        if screening == STRONG_GAP_SAFE:

            disabled_features = np.zeros(n_features, dtype=np.intc, order='F')
            mask = np.where(np.abs(XTR) < 2 * lambdas[t] - dual_scale)[0]
            disabled_features[mask] = 1

            # Strong Rule without kkt
            cd_lasso(X_data, X_indices, X_indptr,
                     y, X_mean, beta_init, norm_Xcent, XTR, XTy, residual, v1,
                     disabled_features, nrm2_y, lambdas[t],
                     lambda_prec, lambda_max, dual_scale, tol, max_iter,
                     f, screening, j_star, wstr_plus=1, kkt=0)

        gaps[t], dual_scale, n_iters[t], n_active_features[t] = \
            cd_lasso(X_data, X_indices, X_indptr,
                     y, X_mean, beta_init, norm_Xcent, XTR, XTy, residual, v1,
                     disabled_features, nrm2_y, lambdas[t],
                     lambda_prec, lambda_max, dual_scale, tol, max_iter,
                     f, screening, j_star, wstr_plus=0)

        betas[t, :] = beta_init.copy()
        # import pdb; pdb.set_trace()
        intercept[t] = y_mean - X_mean.dot(beta_init)
        if t == 0 and screening != NO_SCREENING:
            n_active_features[0] = 0

        if warm_start_plus and t < n_lambdas - 1 and t != 0 and \
           n_active_features[t] < n_features:

                    cd_lasso(X_data, X_indices, X_indptr,
                             y, beta_init, XTR, XTy, residual, v1,
                             disabled_features, nrm2_y, lambdas[t + 1],
                             lambda_prec, lambda_max, dual_scale, tol,
                             max_iter, f, screening=screening, j_star=j_star,
                             wstr_plus=1)

        if abs(gaps[t]) > tol:

            print "warning: did not converge, t = ", t
            print "gap = ", gaps[t], "eps = ", eps

    return intercept, betas, gaps, n_iters, n_active_features


if __name__ == '__main__':

    import time
    from sklearn.datasets.mldata import fetch_mldata
    import scipy as sp

    n_samples = 100
    n_features = 500

    # X = np.random.randn(n_samples, n_features)
    # X[np.random.uniform(size=(n_samples, n_features)) < 0.9] = 0
    # y = np.random.uniform(size=n_samples)

    dataset = "leukemia"
    data = fetch_mldata(dataset)
    X = data.data
    y = data.target
    X = X.astype(float)
    y = y.astype(float)

    # dataset = "finance"
    # X = sp.sparse.load_npz('finance_filtered.npz')
    # y = np.load("finance_target.npy")

    n_samples, n_features = X.shape

    # parameters
    j_star = np.argmax(np.abs(X.T.dot(y)))
    alpha_max = np.linalg.norm(X.T.dot(y), ord=np.inf)
    n_alphas = 5
    eps = 1e-3
    alpha_ratio = eps ** (1. / (n_alphas - 1))
    # alphas = np.array([alpha_max * (alpha_ratio ** i) for i in range(0, n_alphas)])
    max_iter = 5000
    tol = 1e-8
    scg = NO_SCREENING

    X = csc_matrix(X)
    tic = time.time()
    intercept, sp_beta, sp_gap, sp_n_iters, _ =\
        sp_lasso_path(X, y.copy(), [alpha_max / 100.], eps=tol, max_iter=max_iter,
                      screening=scg, j_star=j_star)
    print "our time = ", time.time() - tic

    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=alpha_max / 100. / n_samples,
                             fit_intercept=True)
    clf.fit(X, y)

    import blitzl1
    blitzl1.set_use_intercept(1)
    prob = blitzl1.LassoProblem(X, y)
    sol = prob.solve(alpha_max / 100.)

    print "intercept = ", intercept, clf.intercept_, sol.intercept



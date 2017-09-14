# Author: Eugene Ndiaye
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# Gap Safe screening rules for sparsity enforcing penalties.
# https://arxiv.org/abs/1611.05780
# firstname.lastname@telecom-paristech.fr

import numpy as np
from numpy.linalg import norm
import scipy as sp
from save_cd_logreg_fast import cd_logreg

NO_SCREENING = 0
GAPSAFE_SEQ = 1
GAPSAFE = 2


def logreg_path(X, y, lambdas, eps=1e-4, max_iter=3000, f=10, beta_init=None,
                screening=GAPSAFE, gap_active_warm_start=False,
                strong_active_warm_start=False, warm_start_plus=False):

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

    active_warm_start = strong_active_warm_start or gap_active_warm_start
    run_active_warm_start = True

    betas = np.zeros((n_lambdas, n_features))
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
                disabled_features = (np.abs(XTR) < 2. * lambdas[t] - lambdas[t - 1]).astype(np.intc)

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

        betas[t, :] = beta_init.copy()

        if abs(gaps[t]) > tol:

            print "warning: did not converge, t = ", t,
            print "gap = ", gaps[t], "eps = ", eps

    return betas, gaps, n_iters, n_active_features


if __name__ == '__main__':

    import sys
    import time
    sys.path.append('/home/endiaye/Documents/phd/lightning_private')
    from lightning.classification import CDClassifier
    from sklearn.datasets.mldata import fetch_mldata

    dataset = "leukemia"
    data = fetch_mldata(dataset)
    X = data.data
    y = data.target
    X = X.astype(float)
    # y_blitz = y.copy()
    y[y == -1] = 0
    y = y.astype(float)
    # X = sp.sparse.csc_matrix(X)

    # n_samples = 100
    # n_features = 1000

    # X = np.random.randn(n_samples, n_features)
    # X[np.random.uniform(size=(n_samples, n_features)) < 0.9] = 0
    # y = np.array(np.random.uniform(size=n_samples) > 0.3, dtype=int)

    # parameters
    lambda_max = np.linalg.norm(X.T.dot(0.5 - y), ord=np.inf)
    lambdas = np.array([lambda_max / 5.])
    eps = 1e-3
    n_lambdas = 100
    lambda_ratio = eps ** (1. / (n_lambdas - 1))
    lambdas = np.array([lambda_max * (lambda_ratio ** i) for i in range(0, n_lambdas)])

    tic = time.time()
    beta, gap, n_iters, _ = logreg_path(X, y, lambdas, eps=1e-8,
                                        screening=2, max_iter=1000,
                                        strong_active_warm_start=True)
    print "our time = ", time.time() - tic

    # from blitz_path import blitz_path
    # y_blitz = 2 * y - 1
    # alpha_max_blitz = np.linalg.norm(np.dot(X.T, 0.5 - y_blitz), ord=np.inf)
    # n_alphas = 100
    # alphas_blitz  = np.array([alpha_max_blitz / 10.])
    # alpha_ratio = eps ** (1. / (n_alphas - 1))
    # alphas_blitz = np.logspace(np.log10(alpha_max_blitz / 1000.),
    #                            np.log10(alpha_max_blitz), n_alphas)[::-1]
    # tic = time.time()
    # betas, gaps = blitz_path(X, y, lambdas, eps=1e-8,
    #                          method="logreg", max_iter=int(1e5))
    # print "time blitz = ", time.time() - tic

    # tic = time.time()
    # clf = CDClassifier(loss="log", alpha=lambdas[0], C=1.0, max_iter=1000,
    #                    penalty='l1', tol=1e-8, termination='dual_gap',
    #                    selection='cyclic')
    # clf.fit(X, y)
    # print("time lightning = ", time.time() - tic)
    # print clf.coef_[clf.coef_ != 0], "\n"
    # print beta[beta != 0]


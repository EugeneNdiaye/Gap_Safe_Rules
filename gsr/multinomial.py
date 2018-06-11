# TODO: clean more + strong gap safe

import numpy as np

from gsr.cd_multinomial_fast import bcd_fast

from sklearn.preprocessing import normalize  # TODO remove it

NO_SCREEN = 0
GAPSAFE = 1


def multinomial_path(X, y, screen=NO_SCREEN, beta_init=None, lambdas=None,
                     max_iter=100, f=10, eps=1e-4, wstr_plus=False):

    tol = eps  # TODO normalize

    n_lambdas = len(lambdas)
    n_samples, n_features = X.shape
    n_tasks = y.shape[1]

    print("screening = %d, wstr_plus = %s" % (screen, wstr_plus))

    # Fortran-contiguous array are used to avoid useless copy of the data.
    X = np.asfortranarray(X)
    y = np.asfortranarray(y)
    # y = np.asarray(y, order='C')

    norm2_X = np.sum(X ** 2, axis=0)

    if beta_init is None:
        beta_init = np.zeros((n_features, n_tasks), order='C')
    else:
        beta_init = np.asarray(beta_init, order='C')

    coefs = np.zeros((n_features, n_tasks, n_lambdas), order='C')

    Xbeta = np.asfortranarray(np.dot(X, beta_init))
    exp_Xbeta = np.exp(Xbeta)
    # TODO:make it sklearn independent
    normalize_exp_Xbeta = normalize(exp_Xbeta, norm='l1', axis=1)
    residual = np.asfortranarray(y - normalize_exp_Xbeta)

    exp_Xbeta = np.asfortranarray(np.exp(Xbeta).T)
    # TODO remove the ugly transposition in exp_Xbeta
    # issue in the computation of norm_scale

    dual_gaps = np.ones(n_lambdas)
    n_iters = np.zeros(n_lambdas)
    n_active_features = np.zeros(n_lambdas)
    disabled_features = np.zeros(n_features, dtype=np.intc, order='F')

    for t in range(n_lambdas):

        model = bcd_fast(X, y, beta_init, residual, Xbeta, exp_Xbeta,
                         n_samples, n_features, n_tasks, norm2_X, lambdas[t],
                         max_iter, f, tol, screen, disabled_features,
                         wstr_plus=0)

        dual_scale_p, dual_gaps[t], n_active_features[t], n_iters[t] = model
        coefs[:, :, t] = beta_init.copy()

        if wstr_plus and t < n_lambdas - 1 and t != 0 and \
           n_active_features[t] < n_features:

            bcd_fast(X, y, beta_init, residual, Xbeta, exp_Xbeta,
                     n_samples, n_features, n_tasks, norm2_X, lambdas[t + 1],
                     max_iter, f, tol, GAPSAFE, disabled_features, wstr_plus=1)

        if abs(dual_gaps[t]) > eps:
            print("Warning did not converge ... t = %s gap = %s eps = %s" %
                  (t, dual_gaps[t], eps))

    return (coefs, dual_gaps, n_active_features, n_iters)

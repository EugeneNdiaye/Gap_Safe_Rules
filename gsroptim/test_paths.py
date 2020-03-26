import numpy as np

from sklearn.datasets import make_classification, make_regression

from gsroptim.sgl_tools import generate_data
from gsroptim.logreg import logreg_path
from gsroptim.lasso import lasso_path
from gsroptim.multi_task_lasso import multitask_lasso_path
from gsroptim.sgl import sgl_path, build_lambdas


def test_logreg_path():
    n_samples, n_features = 20, 100
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_classes=2, random_state=0)
    lambda_max = np.linalg.norm(np.dot(X.T, 0.5 - y), ord=np.inf)
    lambdas = lambda_max / np.arange(5, 30, 5)
    eps = 1e-8
    betas, gaps = logreg_path(X, y, lambdas, eps=eps)[:2]
    # beware that tol is scaled inside:
    n_1 = np.sum(y == 1)
    n_0 = n_samples - n_1
    tol = eps * max(1, min(n_1, n_0)) / float(n_samples)
    np.testing.assert_array_less(gaps, tol)


def test_lasso_path():
    n_samples, n_features = 20, 100
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features, random_state=2)
    lambda_max = np.linalg.norm(np.dot(X.T, y), ord=np.inf)
    lambdas = lambda_max / np.arange(5, 30, 5)

    eps = 1e-8
    betas, gaps = lasso_path(X, y, lambdas, eps=eps)[1:3]
    # beware that tol is scaled inside:
    tol = eps * np.linalg.norm(y) ** 2
    np.testing.assert_array_less(gaps, tol)


def test_mtl_path():
    X, y = make_regression(n_samples=20, n_features=100,
                           n_targets=4, random_state=3)
    lambda_max = np.max(np.sqrt(np.sum(np.dot(X.T, y) ** 2, axis=1)))
    lambdas = lambda_max / np.arange(5, 30, 5)
    eps = 1e-8
    betas, gaps = multitask_lasso_path(X, y, lambdas, eps=eps)[:2]
    tol = eps * np.linalg.norm(y) ** 2
    np.testing.assert_array_less(gaps, tol)


def test_sgl_path():
    n_samples, n_features = 20, 100
    size_group = 20  # all groups have size = size_group
    size_groups = size_group * np.ones(n_features // size_group,
                                       dtype=np.intc)
    X, y = generate_data(n_samples, n_features, size_groups, rho=0.4)
    omega = np.sqrt(size_groups)
    g_start = np.cumsum(size_groups, dtype=np.intc) - size_groups[0]
    lambda_max = build_lambdas(X, y, omega, size_groups, g_start,
                               n_lambdas=1)[0]
    lambdas = lambda_max / np.arange(5, 30, 5)
    eps = 1e-8
    betas, gaps = sgl_path(X, y, size_groups, omega, lambdas, eps=eps)[:2]
    tol = eps * np.linalg.norm(y) ** 2
    np.testing.assert_array_less(gaps, tol)

from sklearn.preprocessing import LabelBinarizer
import pytest
import itertools
import numpy as np

from scipy import sparse
from sklearn.datasets import make_classification, make_regression

from gsroptim.sgl_tools import generate_data
from gsroptim.logreg import logreg_path
from gsroptim.lasso import lasso_path
from gsroptim.multi_task_lasso import multitask_lasso_path
from gsr_optim.multinomial import multinomial_path
from gsroptim.sgl import sgl_path, build_lambdas


SCREEN_METHODS = [
    "Gap Safe (GS)", "aggr. GS", "strong GS", "aggr. strong GS",
    "active warm start", "active GS", "aggr. active GS"]


@pytest.mark.parametrize("sparse_X, init",
                         itertools.product([True, False], [True, False]))
def test_logreg_path(sparse_X, init):
    n_samples, n_features = 20, 100
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_classes=2, random_state=0)
    if sparse_X:
        X = sparse.random(n_samples, n_features, random_state=2, format='csc',
                          density=0.5)
    lambda_max = np.linalg.norm(X.T @ (0.5 - y), ord=np.inf)
    lambdas = lambda_max / np.arange(5, 30, 5)
    eps = 1e-8
    beta_init = np.zeros(n_features) if init else None
    betas, gaps = logreg_path(X, y, lambdas, beta_init=beta_init, eps=eps)[:2]
    # beware that tol is scaled inside:
    n_1 = np.sum(y == 1)
    n_0 = n_samples - n_1
    tol = eps * max(1, min(n_1, n_0)) / float(n_samples)
    np.testing.assert_array_less(gaps, tol)


@pytest.mark.parametrize("sparse_X, fit_intercept",
                         itertools.product([True, False], [True, False]))
def test_lasso_path(sparse_X, fit_intercept):
    n_samples, n_features = 20, 100
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features, random_state=2)
    if sparse_X:
        X = sparse.random(n_samples, n_features, random_state=2, format='csc',
                          density=0.5)
    lambda_max = np.linalg.norm(X.T @ y, ord=np.inf)
    lambdas = lambda_max / np.arange(5, 30, 5)

    eps = 1e-8
    betas, gaps = lasso_path(X, y, lambdas, eps=eps,
                             fit_intercept=fit_intercept)[1:3]
    # beware that tol is scaled inside:
    tol = eps * np.linalg.norm(y) ** 2
    np.testing.assert_array_less(gaps, tol)


@pytest.mark.parametrize("screen_method", SCREEN_METHODS)
def test_lasso_rules(screen_method):
    n_samples, n_features = 20, 100
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features, random_state=2)
    lambda_max = np.linalg.norm(X.T @ y, ord=np.inf)
    lambdas = lambda_max / np.arange(5, 30, 5)

    eps = 1e-8
    betas, gaps = lasso_path(
        X, y, lambdas, eps=eps, screen_method=screen_method)[1:3]
    # beware that tol is scaled inside:
    tol = eps * np.linalg.norm(y) ** 2
    np.testing.assert_array_less(gaps, tol)


def test_mtl_path():
    X, y = make_regression(n_samples=20, n_features=100,
                           n_targets=4, random_state=0)
    lambda_max = np.max(np.sqrt(np.sum((X.T @ y) ** 2, axis=1)))
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


def test_multinomial_path():
    n_samples, n_features = 20, 50
    X, _ = generate_data(n_samples, n_features, rho=0.4)
    y = np.random.choice(4, n_samples)
    y = LabelBinarizer().fit_transform(y)
    _, gaps = multinomial_path(X, y)

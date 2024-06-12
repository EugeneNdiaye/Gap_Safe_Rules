[![](https://travis-ci.com/EugeneNdiaye/Gap_Safe_Rules.svg?branch=master)](https://travis-ci.com/github/EugeneNdiaye/Gap_Safe_Rules)
[![](https://codecov.io/gh/EugeneNdiaye/Gap_Safe_Rules/branch/master/graphs/badge.svg?branch=master)](https://codecov.io/gh/EugeneNdiaye/Gap_Safe_Rules/)
# Gap Safe screening rules for sparsity enforcing penalties.

This package implements coordinate descent with Gap Safe screening rules. See our paper https://arxiv.org/abs/1611.05780 for more details.


# Examples in classification and regression
```python
import numpy as np

from sklearn.datasets import make_classification, make_regression

from gsroptim.sgl_tools import generate_data
from gsroptim.logreg import logreg_path
from gsroptim.lasso import lasso_path
from gsroptim.multi_task_lasso import multitask_lasso_path
from gsroptim.sgl import sgl_path, build_lambdas

n_samples = 20
n_features = 100

# Sparse Logistic Regression
X, y = make_classification(n_samples=n_samples, n_features=n_features,
                           n_classes=2)
lambda_max = np.linalg.norm(np.dot(X.T, 0.5 - y), ord=np.inf)
lambdas = lambda_max / np.arange(5, 30, 5)
betas, gaps = logreg_path(X, y, lambdas)[:2]


# Sparse Least Squares Regression
X, y = make_regression(n_samples=n_samples, n_features=n_features)
lambda_max = np.linalg.norm(np.dot(X.T, y), ord=np.inf)
lambdas = lambda_max / np.arange(5, 30, 5)
betas, gaps = lasso_path(X, y, lambdas)[1:3]

# Sparse Multi-task Regression
X, y = make_regression(n_samples=20, n_features=100, n_targets=4)
lambda_max = np.max(np.sqrt(np.sum(np.dot(X.T, y) ** 2, axis=1)))
lambdas = lambda_max / np.arange(5, 30, 5)
betas, gaps = multitask_lasso_path(X, y, lambdas)[:2]


# Sparse Group Lasso
size_group = 20  # all groups have size = size_group
size_groups = size_group * np.ones(int(n_features / size_group), order='F',
                                   dtype=np.intc)
X, y = generate_data(n_samples, n_features, size_groups, rho=0.4)
omega = np.sqrt(size_groups)
n_groups = len(size_groups)
g_start = np.cumsum(size_groups, dtype=np.intc) - size_groups[0]
lambda_max = build_lambdas(X, y, omega, size_groups, g_start, n_lambdas=1)[0]
lambdas = lambda_max / np.arange(5, 30, 5)
betas, gaps = sgl_path(X, y, size_groups, omega, lambdas)[:2]

```

## Installation & Requirements

The compilation proceed as follows:

```
$ pip install -e .
```

This package has the following requirements:

- [numpy](http://numpy.org)
- [scipy](http://scipy.org)
- [scikit-learn](http://scikit-learn.org)
- [cython](http://cython.org/)

We recommend to install or update anaconda (at least version 0.16.1).

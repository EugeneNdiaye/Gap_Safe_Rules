# Gap Safe screening rules for sparsity enforcing penalties.

This package implements coordinate descent with Gap Safe screening rules. See our paper https://arxiv.org/abs/1611.05780 for more details.


# Example in binary classification with sparse logistic regression
```python
import time
import numpy as np
from sklearn.datasets import make_classification
from logreg import logreg_path
import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')
plt.style.use('ggplot')

X, y = make_classification(n_samples=100, n_features=1000, n_classes=2,
                           random_state=414)
X = X.astype(float)
y = y.astype(float)

eps = 1e-3  # the smaller it is the longer is the path
lambda_max = np.linalg.norm(np.dot(X.T, 0.5 - y), ord=np.inf)
n_lambdas = 100
lambda_ratio = eps ** (1. / (n_lambdas - 1))
lambdas = lambda_max * (lambda_ratio ** np.arange(n_lambdas))

NO_SCREENING = 0
GAPSAFE_SEQ = 1
GAPSAFE = 2
GAPSAFE_SEQ_pp = 3
GAPSAFE_pp = 4
STRONG_RULE = 10
SAFE_STRONG_RULE = 666

seq = " (Seq.)"
seq_dyn = " (Seq. + Dyn.)"
seq_wrst = " (Seq. + Active warm start)"
seq_dyn_wrst = " (Seq. + Dyn. + Active warm start)"
seq_dyn_strong_wrst = " (Seq. + Dyn. + Strong warm start)"

screenings = [0, 1, 3, 2, 10, 4, 666]
screenings_names = [r"No Screening",
                    r"Gap Safe" + seq,
                    r"Gap Safe" + seq_wrst,
                    r"Gap Safe" + seq_dyn,
                    r"Strong Rule" + seq,
                    r"Gap Safe" + seq_dyn_wrst,
                    r"Gap Safe" + seq_dyn_strong_wrst]
screenings_colors = ['b', 'c', 'k', 'm', '#00FF00', 'y', '#800000']

tols = [2, 4, 6]
times = np.zeros((len(screenings), len(tols)))

# Bench computational time
for itol, tol_exp in enumerate(tols):

    tol = 10 ** (-tol_exp)
    for iscreening, screening_type in enumerate(screenings):

        if screening_type == GAPSAFE_SEQ_pp:
            screening_type = GAPSAFE_SEQ
            wstr_plus = True

        elif screening_type == GAPSAFE_pp:
            screening_type = GAPSAFE
            wstr_plus = True

        else:
            wstr_plus = False

        tic = time.time()

        betas, gaps, n_iters, _ = logreg_path(X, y, lambdas=lambdas, eps=tol,
                                              screening=screening_type,
                                              max_iter=int(1e5),
                                              warm_start_plus=wstr_plus)

        toc = time.time() - tic
        times[iscreening, itol] = toc
        print(screenings_names[iscreening], "tol = ", tol, "time = ", toc)


df = pd.DataFrame(times.T, columns=screenings_names)
df.plot(kind='bar', rot=0, color=screenings_colors)
plt.xticks(range(len(tols)), [r"$%s$" % (np.str(t)) for t in tols])
plt.xlabel(r"-log10(duality gap)")
plt.ylabel(r"Time (s)")
plt.grid(color='w')
leg = plt.legend(frameon=True, loc='upper left')
plt.tight_layout()
plt.savefig("toybench_logreg.png", format="png")
plt.show()
```

![Computational time](toybench_logreg.png)


## Installation & Requirements
This package has the following requirements:

- [numpy](http://numpy.org)
- [scipy](http://scipy.org)
- [scikit-learn](http://scikit-learn.org)
- [cython](http://cython.org/)

We recommend to install or update anaconda (at least version 0.16.1).

The compilation proceed as follows:

```
$ python setup.py build_ext --inplace
```

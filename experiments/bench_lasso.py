
from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
from lasso import lasso_path
from sklearn.datasets.mldata import fetch_mldata
from blitz_path import blitz_path
from check_dual_gap import check_lasso

plt.close('all')

dataset_id = 2
bench_time = 1
bench_active_set = 0

NO_SCREENING = 0

STATIC = 1
DST3 = 2

GAPSAFE_SEQ = 3
GAPSAFE = 4

GAPSAFE_SEQ_pp = 5
GAPSAFE_pp = 6

STRONG_RULE = 10
EDPP = 11

SAFE_STRONG_RULE = 666

BLITZ = -42


seq = " (Seq.)"
seq_dyn = " (Seq. + Dyn.)"
seq_wrst = " (Seq. + Active warm start)"
seq_dyn_wrst = " (Seq. + Dyn. + Active warm start)"
seq_dyn_strong_wrst = " (Seq. + Dyn. + Strong warm start)"

screenings = [0, 1, 2, 11, 3, 5, 4, 10, 6, -42, 666]
# screenings_names = [r"No Screening",
#                     r"Static",
#                     r"DST3" + seq_dyn,
#                     r"EDPP" + seq,
#                     r"Gap Safe" + seq,
#                     r"Gap Safe" + seq_wrst,
#                     r"Gap Safe" + seq_dyn,
#                     r"Strong Rule" + seq,
#                     r"Gap Safe" + seq_dyn_wrst,
#                     r"Blitz",
#                     r"Gap Safe" + seq_dyn_strong_wrst]
screenings_names = [r"No Screening",
                    r"Static",
                    r"DST3" + seq_dyn,
                    r"EDPP" + seq,
                    r"Gap Safe" + seq,
                    r"Gap Safe" + seq + "++",
                    r"Gap Safe" + seq_dyn,
                    r"Strong Rule" + seq,
                    r"Gap Safe" + seq_dyn + "++",
                    r"Blitz",
                    r"Strong Gap Safe"]

if dataset_id == 1:
    dataset = "synthetic"
    X, y = make_regression(n_samples=10, n_features=20,
                           random_state=42)
    X = X.astype(float)
    y = y.astype(float)
    eps = 1e-3

elif dataset_id == 2:
    dataset = "leukemia"
    data = fetch_mldata(dataset)
    X = data.data
    y = data.target
    X = X.astype(float)
    y = y.astype(float)
    eps = 1e-3  # the smaller it is the longer is the path


j_star = np.argmax(np.abs(np.dot(X.T, y)))
alpha_max = np.linalg.norm(np.dot(X.T, y), ord=np.inf)
n_alphas = 10
alpha_ratio = eps ** (1. / (n_alphas - 1))
alphas = np.array([alpha_max * (alpha_ratio ** i) for i in range(0, n_alphas)])
space = "".ljust(15)  # just for displaying results

if bench_time:
    tols = [2, 4, 6, 8]
    # tols = [8]

    times = np.zeros((len(screenings), len(tols)))

    for itol, tol_exp in enumerate(tols):

        print("\n")

        tol = 10 ** (-tol_exp)
        scaled_tol = tol * np.linalg.norm(y) ** 2
        print("tol = ", scaled_tol, "".ljust(15), "time", "".center(25),
              "gap check", "".center(25), "gap solver")
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

            if screening_type == BLITZ:

                beta, gap = blitz_path(X, y, alphas, tol, max_iter=int(1e5))

            else:

                beta, gap, n_iters, _ = lasso_path(X, y, lambdas=alphas,
                                                   eps=tol,
                                                   screening=screening_type,
                                                   max_iter=int(1e5),
                                                   j_star=j_star,
                                                   warm_start_plus=wstr_plus)

            toc = time.time() - tic
            times[iscreening, itol] = toc

            dual_gap = np.ones(n_alphas)
            for i_l, lambda_ in enumerate(alphas):
                dual_gap[i_l] = check_lasso(X, y, beta[i_l, :], lambda_,
                                            tol, i_l)

            print(screenings_names[iscreening].ljust(25), toc, space,
                  np.max(dual_gap), space, np.max(gap))

    np.save("bench_data/times_lasso_" + dataset + ".npy", times)


if bench_active_set:
    T = n_alphas
    max_iters = 2 ** np.arange(1, 10, 1)

    # remove active_set for No screening, Gap Safe Seq ++ & Gap Safe ++
    screenings = screenings[+1:-2]
    screenings_names = screenings_names[+1:-2]

    alphas_ratio = alphas[-1] / alphas[0]
    screening_sizes_features = np.zeros((len(screenings), len(max_iters), T))

    beta_init = np.zeros((X.shape[1], len(screenings)))

    for imax_iter, max_iter in enumerate(max_iters):
        print("max_iter = ", max_iter)
        for iscreening, screening_type in enumerate(screenings):

            beta, gap, n_iters, screening_size_feature = \
                lasso_path(X, y, lambdas=alphas, eps=0, j_star=j_star, f=1,
                           screening=screening_type, max_iter=max_iter)

            screening_sizes_features[iscreening, imax_iter] = \
                screening_size_feature

    np.save("bench_data/active_set_lasso_" + dataset + ".npy",
            screening_sizes_features)

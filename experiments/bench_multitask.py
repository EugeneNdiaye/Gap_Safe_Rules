
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from multi_task_lasso import multitask_lasso_path
from sklearn.datasets.mldata import fetch_mldata
from scipy import io
import pandas as pd

plt.close('all')

dataset_id = 3
bench_time = 1
bench_active_set = 0

if dataset_id == 1:
    dataset = "synthetic"
    n_samples, n_features, n_tasks = (47, 1177, 20)
    # generate dataset
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           n_targets=n_tasks)  # , random_state=2)
    X = X.astype(float)
    y = y.astype(float)
    eps = 1e-3

elif dataset_id == 2:
    dataset = "leukemia"
    data = fetch_mldata(dataset)
    X = data.data  # [:, ::10]
    y = data.target[:, None]
    X = X.astype(float)
    y = y.astype(float)
    eps = 1e-3

if dataset_id == 3:
    dataset = 'meg_full'
    data = io.loadmat('meg_Xy_full.mat')
    X = np.array(data['X'], dtype=np.float, order='F')
    Y = np.array(data['Y'], dtype=np.float)
    y = Y
    idx = np.argmax(np.sum(Y ** 2, axis=0))
    y = Y[:, idx - 10:idx + 10]
    eps = 1e-3

j_star = np.argmax(np.sqrt(np.sum(np.dot(X.T, y) ** 2, axis=1)))
alpha_max = np.max(np.sqrt(np.sum(np.dot(X.T, y) ** 2, axis=1)))
n_alphas = 100
alpha_ratio = eps ** (1. / (n_alphas - 1))
alphas = np.array([alpha_max * (alpha_ratio ** i) for i in range(n_alphas)])


NO_SCREENING = 0

DGST3 = 1

GAPSAFE_SEQ = 2
GAPSAFE = 3

GAPSAFE_SEQ_pp = 4
GAPSAFE_pp = 5

STRONG_RULE = 10

SAFE_STRONG_RULE = 666

seq = " (Seq.)"
seq_dyn = " (Seq. + Dyn.)"
seq_wrst = " (Seq. + Active warm start)"
seq_dyn_wrst = " (Seq. + Dyn. + Active warm start)"
seq_dyn_strong_wrst = " (Seq. + Dyn. + Strong warm start)"

screenings = [0, 1, 2, 4, 3, 10, 5, 666]
screenings_names = [r"No Screening",
                    r"DST3" + seq_dyn,
                    r"Gap Safe" + seq,
                    r"Gap Safe" + seq_wrst,
                    r"Gap Safe" + seq_dyn,
                    r"Strong Rule" + seq,
                    r"Gap Safe" + seq_dyn_wrst,
                    r"Gap Safe" + seq_dyn_strong_wrst]

tols = [4, 6, 8]

if bench_time:

    times = np.zeros((len(screenings), len(tols)))
    # times = pd.DataFrame(None, index=tols, columns=screenings_names)

    for itol, tol_exp in enumerate(tols):

        print
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

            beta, gap, n_iters, n_active_features = \
                multitask_lasso_path(X, y, lambdas=alphas, eps=tol,
                                     max_iter=50000, j_star=j_star,
                                     screen=screening_type,
                                     wstr_plus=wstr_plus)

            toc = time.time() - tic
            times[iscreening, itol] = toc
            # times[screenings_names[iscreening]][tol_exp] = toc
            print screenings_names[iscreening], "tol = ", tol, "time = ", toc

    np.save("bench_data/times_multi_tasks_" + dataset + ".npy", times)
    # times.to_csv("bench_data/times_multi_tasks_" + dataset + ".csv")


if bench_active_set:

    # remove active_set for No Screening & Gap Safe ++
    screenings = screenings[+1:-1]
    screenings_names = screenings_names[+1:-1]

    T = n_alphas
    max_iters = 2 ** np.arange(1, 10, 1)

    alpha_ratio = alphas / alphas[0]
    screening_sizes_features = np.zeros((len(screenings), len(max_iters), T))
    # screening_sizes_features = pd.DataFrame(None, index=max_iters,
    #                                         columns=screenings_names)
    beta_init = np.zeros((X.shape[1], len(screenings)))

    for imax_iter, max_iter in enumerate(max_iters):
        for iscreening, screening_type in enumerate(screenings):

            if screening_type == GAPSAFE_pp:
                screening_type = GAPSAFE
                wstr_plus = True
            else:
                wstr_plus = False

            beta, gap, n_iters, screening_size_feature = \
                multitask_lasso_path(X, y, lambdas=alphas, eps=0,
                                     max_iter=max_iter, f=1,
                                     j_star=j_star,
                                     screen=screening_type,
                                     wstr_plus=wstr_plus)

            screening_sizes_features[iscreening, imax_iter] = \
                screening_size_feature
            # screening_sizes_features[screenings_names[iscreening]][max_iter] =\
            #     screening_size_feature

    np.save("bench_data/active_set_multi_tasks_" + dataset + ".npy",
            screening_sizes_features)
    # screening_sizes_features.to_csv("bench_data/active_set_multi_tasks_" +
    #                                 dataset + ".csv")


import time
import numpy as np
from sklearn.datasets import make_classification
from logreg import logreg_path
from sklearn.datasets.mldata import fetch_mldata
from sklearn.datasets import load_svmlight_file
# import pandas as pd
from blitz_path import blitz_path
import matplotlib.pyplot as plt
from matplotlib import rc

plt.close('all')

plt.style.use('ggplot')

fontsize = 13
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 20,
          'font.size': 15,
          'legend.fontsize': 15,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'text.usetex': True,
          'text.latex.preamble': r'\usepackage{amsmath}'}
plt.rcParams.update(params)

dataset_id = 2
bench_time = 1
bench_active_set = 0
display_time = 0

if dataset_id == 1:
    dataset = "synthetic"
    X, y = make_classification(n_samples=50,
                               n_features=3000,
                               n_classes=2,
                               random_state=42)
    X = X.astype(float)
    X /= np.sqrt(np.sum(X ** 2, axis=0))
    mask = np.sum(np.isnan(X), axis=0) == 0
    if np.any(mask):
        X = X[:, mask]
    y = y.astype(float)
    y_blitz = 2 * y - 1  # blitz's label = +-1
    eps = 1e-3  # the smaller it is the longer is the path

elif dataset_id == 2:
    dataset = "leukemia"
    data = fetch_mldata(dataset)
    X = data.data  # [:, ::10]
    y = data.target
    X = X.astype(float)
    y = y.astype(float)
    y_blitz = y.copy()  # blitz's label = +-1
    y[y == -1] = 0
    eps = 1e-3  # the smaller it is the longer is the path

elif dataset_id == 3:

    # download the file here
    # http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#news20.binary
    X, y = load_svmlight_file("data/news20.binary")
    X = X.astype(float)
    y = y.astype(float)
    y[y == -1] = 0


alpha_max = np.linalg.norm(np.dot(X.T, 0.5 - y), ord=np.inf)
alpha_max_blitz = np.linalg.norm(np.dot(X.T, y_blitz), ord=np.inf) / 2.
n_alphas = 100
alpha_ratio = eps ** (1. / (n_alphas - 1))
alphas = np.array([alpha_max * (alpha_ratio ** i) for i in range(0, n_alphas)])
alphas_blitz = np.logspace(np.log10(alpha_max_blitz / 1000.),
                           np.log10(alpha_max_blitz), n_alphas)[::-1]


NO_SCREENING = 0

GAPSAFE_SEQ = 1
GAPSAFE = 2

GAPSAFE_SEQ_pp = 3
GAPSAFE_pp = 4

STRONG_RULE = 10
SLORES = 11

SAFE_STRONG_RULE = 666

BLITZ = -42

seq = " (Seq.)"
seq_dyn = " (Seq. + Dyn.)"
seq_wrst = " (Seq. + Active warm start)"
seq_dyn_wrst = " (Seq. + Dyn. + Active warm start)"
seq_dyn_strong_wrst = " (Seq. + Dyn. + Strong warm start)"

# screenings = [0, 1, 3, 2, 10, 4, 666]
# screenings_names = [r"No Screening",
#                     r"Gap Safe" + seq,
#                     r"Gap Safe" + seq_wrst,
#                     r"Gap Safe" + seq_dyn,
#                     r"Strong Rule" + seq,
#                     r"Gap Safe" + seq_dyn_wrst,
#                     r"Gap Safe" + seq_dyn_strong_wrst]

screenings = [2, 4, 666, -42]
screenings_names = [r"Gap Safe" + seq_dyn,
                    r"Gap Safe" + seq_dyn_wrst,
                    r"Gap Safe" + seq_dyn_strong_wrst,
                    r"Blitz"]

tols = [2, 4, 6, 8]

if bench_time:

    times = np.zeros((len(screenings), len(tols)))

    for itol, tol_exp in enumerate(tols):

        tol = 10 ** (-tol_exp)
        print("\n tol = ", tol)
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

                betas, gaps = blitz_path(X, y_blitz, alphas_blitz, tol,
                                         method="logreg", max_iter=int(1e5))

            else:
                betas, gaps, n_iters, _ = logreg_path(X, y, lambdas=alphas,
                                                      eps=tol,
                                                      screening=screening_type,
                                                      max_iter=int(1e5),
                                                      warm_start_plus=wstr_plus)

            toc = time.time() - tic
            times[iscreening, itol] = toc
            # print screenings_names[iscreening], "tol = ", tol, "time = ", toc
            print(screenings_names[iscreening], "time = ", toc)
            # print "\n gaps = ", gaps

    # np.save("bench_data/times_logreg_" + dataset + ".npy", times)


if bench_active_set:
    T = n_alphas
    max_iters = 2 ** np.arange(1, 10, 1)

    # remove active_set for No screening, Gap Safe Seq ++ & Gap Safe ++
    screenings = screenings[+1:-2]
    screenings_names = screenings_names[+1:-2]

    alpha_ratio = alphas[-1] / alphas[0]
    screening_sizes_features = np.zeros((len(screenings), len(max_iters), T))

    beta_init = np.zeros((X.shape[1], len(screenings)))

    for imax_iter, max_iter in enumerate(max_iters):
        print("max_iter = ", max_iter)
        for iscreening, screening_type in enumerate(screenings):

            beta, gap, n_iters, screening_size_feature = \
                logreg_path(X, y, lambdas=alphas, eps=0, f=1,
                            screening=screening_type, max_iter=max_iter)

            screening_sizes_features[iscreening, imax_iter] = \
                screening_size_feature

    np.save("bench_data/active_set_logreg_" + dataset + ".npy",
            screening_sizes_features)

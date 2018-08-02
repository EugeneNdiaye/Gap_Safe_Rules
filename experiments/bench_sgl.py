# -*- coding: utf-8 -*-
import numpy as np
from sgl import sgl_path
from sgl_tools import generate_data, build_lambdas
# from process_climate import target_region
import time
from sklearn.datasets.mldata import fetch_mldata
import pandas as pd

T = 100
dataset_id = 1

if dataset_id == 0:
    dataset = "synthetic"
    n_samples = 50
    n_features = 800
    size_group = 40  # all groups have size = size_group
    delta = 3
    tau = .34

    n_groups = n_features / size_group
    size_groups = size_group * np.ones(n_groups, order='F', dtype=np.intc)
    omega = np.ones(n_groups)
    groups = np.arange(n_features) // size_group
    group_labels = [np.where(groups == i)[0] for i in np.unique(groups)]
    X, y = generate_data(n_samples, n_features, size_groups, rho=0.4)

    T = 30


elif dataset_id == 1:
    target = "Dakar"
    dataset = "DK_clim"

    if target == "Dakar":
        latitude = 14
        longitude = -17

    # X, y = target_region(latitude, longitude)
    X = np.load("_Xclimate_design.npy")
    y = np.load("_yclimate_target.npy")
    X = np.asfortranarray(X)
    y = np.asfortranarray(y)

    n_samples, n_features = X.shape
    groups = np.arange(n_features) // 7
    n_groups = n_features / 7
    size_groups = 7 * np.ones(n_groups)
    omega = np.ones(n_groups)  # since all groups have the same size
    group_labels = [np.where(groups == i)[0] for i in np.unique(groups)]
    T = 100
    delta = 2.5
    tau = 0.4


elif dataset_id == 2:
    dataset = "leukemia"
    data = fetch_mldata(dataset)
    X = data.data  # [:, ::10][:, :712]
    y = data.target
    X = X.astype(float)
    y = y.astype(float)
    n_samples, n_features = X.shape
    delta = 3
    tau = .4

    # n_groups = n_features / 8
    # n_groups = 357
    # size_groups = 8 * np.ones(n_groups, order='F', dtype=np.intc)
    # omega = np.ones(n_groups)
    groups = np.arange(n_features) // 20
    group_labels = [np.where(groups == i)[0] for i in np.unique(groups)]
    n_groups = len(group_labels)
    omega = np.ones(n_groups)
    size_groups = np.asarray([len(np.where(groups == i)[0])
                              for i in np.unique(groups)], dtype=np.intc,
                             order='F')


n_samples, n_features = X.shape
print "n = ", n_samples, "p = ", n_features, dataset

NO_SCREENING = 0

STATIC_SAFE = 1
DYNAMIC_SAFE = 2
DST3 = 3

GAPSAFE_SEQ = 4
GAPSAFE = 5

GAPSAFE_SEQ_pp = 6
GAPSAFE_pp = 7

STRONG_RULE = 8
TLFre = 9

SAFE_STRONG_RULE = 666

seq = " (Seq.)"
seq_dyn = " (Seq. + Dyn.)"
seq_wrst = " (Seq. + Active warm start)"
seq_dyn_wrst = " (Seq. + Dyn. + Active warm start)"
seq_dyn_strong_wrst = " (Seq. + Dyn. + Strong warm start)"

screenings = [NO_SCREENING, DST3, TLFre, GAPSAFE_SEQ, GAPSAFE_SEQ_pp,
              GAPSAFE, STRONG_RULE, GAPSAFE_pp, SAFE_STRONG_RULE]

screenings_names = [r"No Screening",
                    r"DST3" + seq_dyn,
                    r"TLfre" + seq,
                    r"Gap Safe" + seq,
                    r"Gap Safe" + seq_wrst,
                    r"Gap Safe" + seq_dyn,
                    r"Strong Rule" + seq,
                    r"Gap Safe" + seq_dyn_wrst,
                    r"Gap Safe" + seq_dyn_strong_wrst]

eps_ = range(4, 10, 2)
times = np.zeros((len(screenings), len(eps_)))
# times = pd.DataFrame(None, index=eps_, columns=screenings_names)
# import pdb; pdb.set_trace()

# g_start = np.zeros(n_groups, order='F', dtype=np.intc)
# for i in range(1, n_groups):
#     g_start[i] = size_groups[i - 1] + g_start[i - 1]
g_start = np.cumsum(size_groups, dtype=np.intc) - size_groups[0]

lambdas, imax = build_lambdas(X, y, omega, size_groups, g_start,
                              n_lambdas=100, delta=delta, tau=tau)

size_groups = np.asfortranarray(size_groups, dtype=np.intc)

for ieps, eps in enumerate(eps_):

    print
    for iscreening, screening in enumerate(screenings):
        begin = time.time()
        # print("screening = %s" % screenings_names[iscreening])

        tic = time.time()
        beta_init = np.zeros(X.shape[1])

        if screening == GAPSAFE_SEQ_pp:
            screening = GAPSAFE_SEQ
            wstr_plus = True

        elif screening == GAPSAFE_pp:
            screening = GAPSAFE
            wstr_plus = True

        else:
            wstr_plus = False

        # sgl_path(X, y, size_groups, omega, screening, lambdas=lambdas,
        #          tau=tau, max_iter=1e5, eps=10**(-eps),
        #          warm_start_plus=wstr_plus)
        sgl_path(X, y, size_groups, omega, screening,
                 tau=tau, max_iter=1e5, eps=10**(-eps),
                 warm_start_plus=wstr_plus)

        toc = time.time() - tic
        times[iscreening, ieps] = toc
        # times[screenings_names[iscreening]][eps] = toc
        print screenings_names[iscreening], "tol = ", 10**(-eps), "time = ", toc

np.save("bench_data/times_sgl_" + dataset + ".npy", times)
# times.to_csv("bench_data/times_sgl_" + dataset + ".csv")

max_iters = 2 ** np.arange(1, 10, 1)

screening_sizes_groups = np.zeros((len(screenings), len(max_iters), T))
screening_sizes_features = np.zeros((len(screenings), len(max_iters), T))


# import pdb; pdb.set_trace()
# remove active_set for No screening, Gap Safe Seq ++ & Gap Safe ++
# screenings = screenings[+1:-2]
# screenings_names = screenings_names[+1:-2]

# screening_sizes_groups = pd.DataFrame(None, index=max_iters,
#                                       columns=screenings_names)
# screening_sizes_features = pd.DataFrame(None, index=max_iters,
#                                         columns=screenings_names)

# for imax_iter, max_iter in enumerate(max_iters):
#     for iscreening, screening in enumerate(screenings):

#         print screenings_names[iscreening], max_iter

#         coefs, dual_gaps, lambdas, screening_sizes_group,\
#             screening_sizes_feature, n_iters = \
#             sgl_path(X, y, size_groups, omega, screening, lambdas=lambdas,
#                      tau=tau, max_iter=max_iter, f=1, eps=0)

#         # screening_sizes_groups[screenings_names[iscreening]][max_iter] = \
#         #     screening_sizes_group

#         # screening_sizes_features[screenings_names[iscreening]][max_iter] =\
#         #     screening_sizes_feature

#         screening_sizes_groups[iscreening, imax_iter] = screening_sizes_group
#         screening_sizes_features[iscreening, imax_iter] =\
#             screening_sizes_feature

# # np.save("bench_data/active_feat_sgl_" + dataset + ".npy",
# #         screening_sizes_features)
# # np.save("bench_data/active_group_sgl_" + dataset + ".npy",
# #         screening_sizes_groups)

# np.save("bench_data/nips_poster_active_feat_sgl_" + dataset + ".npy",
#         screening_sizes_features)
# np.save("bench_data/nips_poster_active_group_sgl_" + dataset + ".npy",
#         screening_sizes_groups)

# screening_sizes_groups.to_csv("bench_data/active_group_sgl_" + dataset + ".csv")
# screening_sizes_features.to_csv("bench_data/active_feat_sgl_" + dataset + ".csv")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets.mldata import fetch_mldata
from sgl_tools import build_lambdas
from scipy import io
from matplotlib import rc
from sklearn.datasets import load_svmlight_file
import scipy as sp

# download the file here
# http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#E2006-log1p
# dataset = "log1p.E2006.train"
# X, y = load_svmlight_file("data/log1p.E2006.train")
# X = X.tocsc()

dataset = "finance"
X = sp.sparse.load_npz('finance_filtered.npz')
y = np.load("finance_target.npy")

method = "lasso"

seq = " (Seq.)"
seq_dyn = " (Seq. + Dyn.)"
seq_wrst = " (Seq. + Active warm start)"
seq_dyn_wrst = " (Seq. + Dyn. + Active warm start)"
seq_dyn_strong_wrst = " (Seq. + Dyn. + Strong warm start)"

lambda_max = np.linalg.norm(X.T.dot(y), ord=np.inf)
# screenings = [0, 1, 2, 11, 3, 5, 4, 10, 6, -42, 666]
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
# screenings_colors = ['b', 'g', 'r', '#808080', 'c', 'k', 'm', '#00FF00',
#                      'y', '#FF8C00', '#800000']


screenings = [3, 4, 10]
screenings_names = [r"Gap Safe" + seq,
                    r"Gap Safe" + seq_dyn,
                    r"Strong Rule" + seq]
screenings_colors = ['b', 'g', 'r']

fig_size_screenings = (7, 4.75)
fig_size_times = (8, 6)
scale_right = 0.65
scale_top = 0.6
tols = [4, 6]

eps = 1e-3
n_lambdas = 100
tmp = eps ** (1. / (n_lambdas - 1))
lambdas = np.array([lambda_max * (tmp ** i) for i in range(0, n_lambdas)])
lambda_ratio = lambdas / lambdas[0]
len_tols = len(tols)

plt.style.use('ggplot')

fontsize = 13
legend_fontsize = 15


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 20,
          'font.size': fontsize,
          'legend.fontsize': legend_fontsize,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'text.usetex': True,
          'text.latex.preamble': r'\usepackage{amsmath}'}
plt.rcParams.update(params)


# times = np.load("bench_data/times_" + method + "_" + dataset + ".npy")
# df = pd.DataFrame(times.T, columns=screenings_names)


# fig, ax = plt.subplots(1, 1, figsize=fig_size_times)
# df.plot(kind='bar', ax=ax, rot=0, color=screenings_colors)
# plt.xticks(range(len_tols), [r"$%s$" % (np.str(t)) for t in tols])
# plt.xlabel(r"$-\log_{10}\text{(duality gap)}$")
# plt.ylabel(r"$\text{Time (s)}$")
# plt.grid(color='w')
# leg = plt.legend(frameon=True, loc='upper left')
# plt.tight_layout()

# plt.savefig("img/times_" + "_" + method + dataset + ".pdf", format="pdf")
# plt.savefig("img/times_" + "_" + method + dataset + ".svg", format="svg")
# plt.savefig("img/times_" + "_" + method + dataset + ".png", format="png")


T = n_lambdas
max_iters = 2 ** np.arange(1, 10, 1)


screening_sizes_features = \
    np.load("bench_data/active_set_" + method + "_" + dataset + ".npy")


left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height
nb_method = 0


plt.figure()
fig, axes = plt.subplots(nrows=len(screenings) - nb_method, ncols=1,
                         sharey=True, figsize=fig_size_screenings)

for iscreening, ax in zip(range(nb_method, len(screenings)), axes):

    im = ax.imshow(screening_sizes_features[iscreening] / float(X.shape[1]),
                   interpolation='nearest',
                   clim=[0, 1], cmap=plt.cm.Spectral, aspect='auto',
                   extent=[0, -np.log10(np.min(lambda_ratio)),
                           np.min(np.log2(max_iters)),
                           np.max(np.log2(max_iters))])

    plt.xlabel(r"$-\log_{10}(\lambda / \lambda_{max})$")
    ax.set_ylabel(r"$\log_2(K)$", fontsize=fontsize + 2)
    ax.set_yticks([int(np.log2(k)) for k in max_iters][::-1][::2])
    ax.set_yticklabels([str(int(np.log2(k))) for k in max_iters][::2],
                       fontsize=fontsize - 2)

    ax.text(left + scale_right * right, bottom + scale_top * top,
            screenings_names[iscreening],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=fontsize + 4, color='w',
            transform=ax.transAxes)

    ax.grid('off')
    if iscreening < (len(screenings) - 1):
        ax.set_xticks([])

plt.tight_layout()
cb = fig.colorbar(im, ax=axes.ravel().tolist())
cb.solids.set_rasterized(True)

plt.savefig(
    "img/active_set_" + method + "_" + dataset + ".pdf", format="pdf")
plt.savefig(
    "img/active_set_" + method + "_" + dataset + ".svg", format="svg")
plt.savefig(
    "img/active_set_" + method + "_" + dataset + ".png", format="png")

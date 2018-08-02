import numpy as np
from sklearn.datasets import load_svmlight_file
import scipy as sp
from sklearn.preprocessing import normalize

scratch = False
preprocess = True

if scratch:
    # download the file here
    # http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#E2006-log1p
    dataset = "log1p.E2006.train"
    X, y = load_svmlight_file("data/log1p.E2006.train")
    X = X.tocsc()

    if preprocess:
        normalize(X, copy=False, axis=0)
        y = (y - y.mean()) / y.std()

    X = X.astype(float)
    y = y.astype(float)

    NNZ = np.diff(X.indptr)
    X_new = X[:, NNZ >= 3]
    X_new.sort_indices()

    sp.sparse.save_npz("finance_filtered", X_new)
    np.save("finance_target", y)

else:
    X = sp.sparse.load_npz('finance_filtered.npz')
    y = np.load("finance_target.npy")

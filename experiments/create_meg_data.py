"""
To run this script you need to install the MNE-Python software.

See:
http://martinos.org/mne/stable/install_mne_python.html
"""

# Author : Alexandre Gramfort
# BSD License

import numpy as np
from scipy.io import savemat
import mne
from mne.datasets import sample
from mne.inverse_sparse.mxne_inverse import _prepare_gain, _check_loose_forward


data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'
subjects_dir = data_path + '/subjects'
condition = 'Left Auditory'
loose, depth = 0.2, 0.9

# Read noise covariance matrix
noise_cov = mne.read_cov(cov_fname)
# Handling average file
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked.crop(tmin=0.04, tmax=0.340)

evoked = evoked.pick_types(eeg=True, meg=True)
# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)

all_ch_names = evoked.ch_names

loose, forward = _check_loose_forward(loose, forward)

# Handle depth weighting and whitening (here is no weights)
X, X_info, whitener, _, _ = _prepare_gain(
    forward, evoked.info, noise_cov, pca=False, depth=depth,
    loose=loose, weights=None, weights_min=None)

# Select channels of interest
sel = [all_ch_names.index(name) for name in X_info['ch_names']]
Y = evoked.data[sel]

# Whiten data
Y = np.dot(whitener, Y)

savemat('meg_Xy_new.mat', dict(X=X, Y=Y))

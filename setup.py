# Author: Eugene Ndiaye (The padawan :-D)
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# GAP Safe Screening Rules for Sparse-Group Lasso.
# firstname.lastname@telecom-paristech.fr

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("*.pyx"),
    include_dirs=[np.get_include()]
)

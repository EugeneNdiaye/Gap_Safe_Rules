# Author: Eugene Ndiaye (The padawan :-D)
#         Olivier Fercoq
#         Alexandre Gramfort
#         Joseph Salmon
# GAP Safe Screening Rules for Sparse-Group Lasso.
# firstname.lastname@telecom-paristech.fr

import os
import numpy as np

from distutils.core import setup, Extension
from Cython.Build import build_ext

descr = 'Gap Safe Screening rules package for sparse GLMs.'

version = None
with open(os.path.join('gsr', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DISTNAME = 'gsr'
DESCRIPTION = descr
MAINTAINER = 'Eugene Ndiaye'
MAINTAINER_EMAIL = 'eugene.ndiaye@telecom-paristech.fr'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'XXX'
VERSION = version
URL = 'XXX'

setup(name='gsr',
      version=VERSION,
      description=DESCRIPTION,
      long_description=open('README.md').read(),
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      packages=['gsr'],
      cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension('gsr.bcd_multitask_lasso_fast',
                    sources=['gsr/bcd_multitask_lasso_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()]),
          Extension('gsr.cd_lasso_fast',
                    sources=['gsr/cd_lasso_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()]),
          Extension('gsr.cd_logreg_fast',
                    sources=['gsr/cd_logreg_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()]),
          Extension('gsr.cd_multinomial_fast',
                    sources=['gsr/cd_multinomial_fast.pyx'],
                    language='c++',
                    include_dirs=[np.get_include()]),
                 ],
      )

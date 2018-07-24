import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

DISTNAME = 'gsroptim'
DESCRIPTION = 'Fast coordinate descent solver with Gap Safe screening Rules'
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Eugene Ndiaye'
MAINTAINER_EMAIL = 'ndiayeeugene@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/EugeneNdiaye/Gap_Safe_Rules.git'
URL = 'https://github.com/EugeneNdiaye/Gap_Safe_Rules.git'
VERSION = None

setup(name='gsroptim',
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      packages=['gsroptim'],
      ext_modules=cythonize("gsroptim/*.pyx"),
      include_dirs=[np.get_include()]
      )

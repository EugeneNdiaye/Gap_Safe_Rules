import glob
from setuptools import dist, setup, Extension
from setuptools.command.build_ext import build_ext
dist.Distribution().fetch_build_eggs(['numpy>=1.12'])
from Cython.Build import cythonize
import numpy as np


DISTNAME = 'gsroptim'
DESCRIPTION = 'Fast coordinate descent solver with Gap Safe screening Rules'
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Eugene Ndiaye'
MAINTAINER_EMAIL = 'ndiayeeugene@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/EugeneNdiaye/Gap_Safe_Rules.git'
URL = 'https://github.com/EugeneNdiaye/Gap_Safe_Rules.git'
VERSION = None


# TODO using language='c++' break so far bc of a conversion
ext_modules = [Extension(name.split('.')[0].replace('/', '.'), sources=[name],
                         language='c', include_dirs=[np.get_include()],
                         extra_compile_args=['-O3'])
               for name in glob.glob("gsroptim/*.pyx")]

setup(name='gsroptim',
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
            install_requires=['numpy>=1.12', 'scipy>=0.18.0',
                        'matplotlib>=2.0.0', 'Cython>=0.26',
                        'scikit-learn>=0.21', 'xarray'],
      packages=['gsroptim'],
      cmdclass={'build_ext': build_ext},
      # ext_modules=ext_modules,
      ext_modules=cythonize(ext_modules)
      )

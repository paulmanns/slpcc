import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [Extension("slpcc.quadratic_cauchy_improvement", ["src/slpcc/quadratic_cauchy_improvement.pyx"],include_dirs=[np.get_include()])]
setup(ext_modules = cythonize(ext_modules))

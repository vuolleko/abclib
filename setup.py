from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension("abclib", ["abclib.pyx"], include_dirs = [numpy.get_include()])
             ]

setup(
    name = "ABC project",
    ext_modules = cythonize(extensions, gdb_debug=False)
     )

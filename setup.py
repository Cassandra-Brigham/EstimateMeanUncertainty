from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize([
        Extension("variogram_tools", ["variogram_tools.pyx"], include_dirs=[numpy.get_include()]),
    ]),
)


from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.extension import Extension

ext_modules = [Extension("pycgp", sources=["pycgp.pyx"], libraries=["cgp"])]

setup(
    cmdclass = {'build_ext': build_ext},
    name = 'pycgp',
    ext_modules = ext_modules
)
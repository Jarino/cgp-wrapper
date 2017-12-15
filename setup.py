from distutils.core import setup

from Cython.Distutils import build_ext
from distutils.extension import Extension

ext_modules = [Extension("cgpwrapper", sources=["cgpwrapper.pyx"], libraries=["cgp"])]

setup(
    cmdclass = {'build_ext': build_ext},
    name = 'cgpwrapper',
    ext_modules = ext_modules
)

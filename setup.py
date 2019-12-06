#!/usr/bin/env python3

from Cython.Build import build_ext
from setuptools import Extension, find_packages, setup


def extension(*args, **kwags):
    return Extension(
        # *args, **kwags, language="c++", extra_compile_args=["-std=c++11"])
        *args, **kwags, language="c++")


# return Extension(
#             *args, **kwargs,
#             language="c++",
#             libraries=["primitiv"],
#             include_dirs=[
#                 np.get_include(),
#                 os.path.join(dirname, "primitiv"),
#             ],
#             extra_compile_args=["-std=c++11"],
#         )

ext_modules = [
    extension("minn.core._core", sources=["minn/core/_core.pyx"]),
]


setup(
    name='minn',
    version='0.1.0',
    author='Hiroki Teranishi',
    author_email='teranishihiroki@gmail.com',
    description='A Minimum Neural Network Toolkit',
    url='https://github.com/chantera/minn',
    license='MIT License',
    install_requires=['numpy>=1.11.0'],
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)

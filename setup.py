from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "cheburaxa",
        ["python/cheburaxa.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name="cheburaxa",
    version=__version__,
    author="Stanislav Morozov",
    author_email="sylvain.corlay@gmail.com",
    url="https://github.com/cheburaxa",
    description="Library for low-rank Chebyshev approximation",
    long_description="",
    ext_modules=ext_modules,
    # extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)

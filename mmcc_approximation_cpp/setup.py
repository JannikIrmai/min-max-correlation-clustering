from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension("mmcc_approximation",
                      ["src/python_interface.cxx"],
                      include_dirs=["include"],
                      define_macros=[('VERSION_INFO', __version__)])
]

setup(
    name="mmcc_approximation",
    version=__version__,
    description="A four approximation algorithm for min max correlation clustering",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
# Author : Meghraj Pardesi
# License: MIT
# The setup file is used for adding a custom reduction operation in torch lib

from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="reduction_op",
    ext_modules=[
        cpp_extension.CppExtension(
            "reduction_op",
            ["reduction.cpp"],
            include_dirs=["[YOUR PATH TO DIRECTORY]/inference/lib/libtorch"],
        )
    ],
    license="Apache License v2.0",
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)

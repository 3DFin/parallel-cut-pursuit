# ----------------------------------------------------------------------#
#  setuptools setup script for compiling cut-pursuit python extensions  #
# ----------------------------------------------------------------------#
""" 
Compilation command: python setup.py build_ext

Camille Baudoin 2019
"""

from setuptools import setup, Extension
import numpy
import os
import platform

# Include directories
include_dirs = [
    numpy.get_include(),  # find the Numpy headers
    "include",
]

prox_include_dirs = [
    "pcd-prox-split/include",
    "pcd-prox-split/matrix-tools/include",
    "pcd-prox-split/proj-simplex/include",
    "wth-element/include",
] 

# Compilation and linkage options
# _GLIBCXX_PARALLEL is only useful for libstdc++ users
# MIN_OPS_PER_THREAD roughly controls parallelization, see doc in README.md
if platform.system() == "Windows":
    extra_compile_args = ["/openmp", "-DMIN_OPS_PER_THREAD=10000"]
    extra_link_args = []
elif platform.system() == "Linux":
    extra_compile_args = [
        "-std=c++11",
        "-fopenmp",
        "-D_GLIBCXX_PARALLEL",
        "-DMIN_OPS_PER_THREAD=10000",
    ]
    extra_link_args = ["-lgomp"]
elif platform.system() == "Darwin":
    extra_compile_args = [
        "-Xpreprocessor -fopenmp",
        "-DMIN_OPS_PER_THREAD=10000",
    ]
    extra_link_args = ["-lomp"]
# It is more a matter of GCC vs. Clang more than a macOS vs Linux issue.
# In the future we should use something like meson/cmake to handle this in a proper way
else:
    raise NotImplementedError("OS not yet supported.")

COMP_T_ON_32_BITS = os.environ.get("COMP_T_ON_32_BITS", None)

#if COMP_T_ON_32_BITS is not None and COMP_T_ON_32_BITS == "1":
extra_compile_args.append("-DCOMP_T_ON_32_BITS")

# Compilation
mod_cp_d1_ql1b = Extension(
    "pycut_pursuit.cp_d1_ql1b_cpy",
    # list source files
    [
        "python/cpython/cp_d1_ql1b_cpy.cpp",
        "src/cp_d1_ql1b.cpp",
        "src/cut_pursuit_d1.cpp",
        "src/cut_pursuit.cpp",
        "src/maxflow.cpp",
        "pcd-prox-split/src/pfdr_d1_ql1b.cpp",
        "pcd-prox-split/src/pfdr_graph_d1.cpp",
        "pcd-prox-split/src/pcd_fwd_doug_rach.cpp",
        "pcd-prox-split/src/pcd_prox_split.cpp",
        "pcd-prox-split/matrix-tools/src/matrix_tools.cpp",
    ],
    include_dirs=include_dirs + prox_include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

mod_cp_d1_lsx = Extension(
    "pycut_pursuit.cp_d1_lsx_cpy",
    # list source files
    [
        "python/cpython/cp_d1_lsx_cpy.cpp",
        "src/cp_d1_lsx.cpp",
        "src/cut_pursuit_d1.cpp",
        "src/cut_pursuit.cpp",
        "src/maxflow.cpp",
        "pcd-prox-split/src/pfdr_d1_lsx.cpp",
        "pcd-prox-split/src/pfdr_graph_d1.cpp",
        "pcd-prox-split/src/pcd_fwd_doug_rach.cpp",
        "pcd-prox-split/src/pcd_prox_split.cpp",
        "pcd-prox-split/proj-simplex/src/proj_simplex.cpp",
    ],
    include_dirs=include_dirs + prox_include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

mod_cp_d0_dist = Extension(
    "pycut_pursuit.cp_d0_dist_cpy",
    # list source files
    [
        "python/cpython/cp_d0_dist_cpy.cpp",
        "src/cp_d0_dist.cpp",
        "src/cut_pursuit_d0.cpp",
        "src/cut_pursuit.cpp",
        "src/maxflow.cpp",
    ],
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

mod_cp_prox_tv = Extension(
    "pycut_pursuit.cp_prox_tv_cpy",
    # list source files
    [
        "python/cpython/cp_prox_tv_cpy.cpp",
        "src/cp_prox_tv.cpp",
        "src/cut_pursuit_d1.cpp",
        "src/cut_pursuit.cpp",
        "src/maxflow.cpp",
        "pcd-prox-split/src/pfdr_prox_tv.cpp",
        "pcd-prox-split/src/pfdr_graph_d1.cpp",
        "pcd-prox-split/src/pcd_fwd_doug_rach.cpp",
        "pcd-prox-split/src/pcd_prox_split.cpp",
    ],
    include_dirs=include_dirs + prox_include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    package_dir={"pycut_pursuit": "python/wrappers"},
    ext_modules=[mod_cp_d1_ql1b, mod_cp_d1_lsx, mod_cp_d0_dist, mod_cp_prox_tv]
)

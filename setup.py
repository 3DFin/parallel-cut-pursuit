   #----------------------------------------------------------------------#
   #  setuptools setup script for compiling cut-pursuit python extensions  #
   #----------------------------------------------------------------------#
""" 
Compilation command: python setup.py build_ext

Camille Baudoin 2019
"""

from setuptools import setup, Extension
import numpy
import os

# Include directories
include_dirs = [numpy.get_include(), # find the Numpy headers
                "include"]

# Compilation and linkage options
# _GLIBCXX_PARALLEL is only useful for libstdc++ users
# MIN_OPS_PER_THREAD roughly controls parallelization, see doc in README.md
if os.name == 'nt': # windows
    extra_compile_args = ["/openmp",
                          "-DMIN_OPS_PER_THREAD=10000"]
    extra_link_args = []
elif os.name == 'posix': # linux
    extra_compile_args = ["-std=c++11", "-fopenmp", "-D_GLIBCXX_PARALLEL",
                          "-DMIN_OPS_PER_THREAD=10000"]
    extra_link_args = ["-lgomp"]
else:
    raise NotImplementedError('OS not yet supported.')

CP_NPY_COMP_32 = os.environ.get("CP_NPY_COMP_32", None)

if CP_NPY_COMP_32 is not None and CP_NPY_COMP_32 == "1":
    extra_compile_args.append("-DCP_NPY_COMP_32")

# Compilation
mod_cp_d1_ql1b = Extension(
        "pycut_pursuit.cp_d1_ql1b_cpy",
        # list source files
        ["python/cpython/cp_d1_ql1b_cpy.cpp", "src/cp_d1_ql1b.cpp",
            "src/cut_pursuit_d1.cpp", "src/cut_pursuit.cpp",
            "src/maxflow.cpp", "src/pfdr_d1_ql1b.cpp",
            "src/matrix_tools.cpp", "src/pfdr_graph_d1.cpp", 
            "src/pcd_fwd_doug_rach.cpp", "src/pcd_prox_split.cpp"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)

mod_cp_d1_lsx = Extension(
        "pycut_pursuit.cp_d1_lsx_cpy",
        # list source files
        ["python/cpython/cp_d1_lsx_cpy.cpp", "src/cp_d1_lsx.cpp",
            "src/cut_pursuit_d1.cpp", "src/cut_pursuit.cpp",
            "src/maxflow.cpp", "src/pfdr_d1_lsx.cpp",
            "src/proj_simplex.cpp", "src/pfdr_graph_d1.cpp",
            "src/pcd_fwd_doug_rach.cpp", "src/pcd_prox_split.cpp"], 
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)

mod_cp_d0_dist = Extension(
        "pycut_pursuit.cp_d0_dist_cpy",
        # list source files
        ["python/cpython/cp_d0_dist_cpy.cpp", "src/cp_d0_dist.cpp",
            "src/cut_pursuit_d0.cpp", "src/cut_pursuit.cpp",
            "src/maxflow.cpp"], 
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)

mod_cp_prox_tv = Extension(
        "pycut_pursuit.cp_prox_tv_cpy",
        # list source files
        ["python/cpython/cp_prox_tv_cpy.cpp", "src/cp_prox_tv.cpp",
            "src/cut_pursuit_d1.cpp", "src/cut_pursuit.cpp",
            "src/maxflow.cpp", "src/pfdr_d1_ql1b.cpp",
            "src/pfdr_graph_d1.cpp", "src/pcd_fwd_doug_rach.cpp",
            "src/pcd_prox_split.cpp", "src/matrix_tools.cpp"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)

setup(package_dir = {'pycut_pursuit': 'python/wrappers'}, ext_modules=[mod_cp_d1_ql1b, mod_cp_d1_lsx, mod_cp_d0_dist])

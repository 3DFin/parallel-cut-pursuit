   #----------------------------------------------------------------------#
   #  distutils setup script for compiling cut-pursuit python extensions  #
   #----------------------------------------------------------------------#
""" 
Compilation command: python setup.py build_ext

Camille Baudoin 2019
"""

from distutils.core import setup, Extension
from distutils.command.build import build
import numpy
import shutil # for rmtree, os.rmdir can only remove _empty_ directory
import os 
import re

###  targets and compile options  ###
to_compile = [ # comment undesired extension modules
    "cp_pfdr_d1_ql1b_cpy",
    "cp_pfdr_d1_lsx_cpy",
    "cp_kmpp_d0_dist_cpy"
]
include_dirs = [numpy.get_include(), # find the Numpy headers
                "../include"]
# compilation and linkage options
# _GLIBCXX_PARALLEL is only useful for libstdc++ users
# MIN_OPS_PER_THREAD roughly controls parallelization, see doc in README.md
if os.name == 'nt': # windows
    extra_compile_args = ["/std:c++11", "/openmp", "-D_GLIBCXX_PARALLEL",
                          "-DMIN_OPS_PER_THREAD=10000"]
    extra_link_args = ["/lgomp"]
elif os.name == 'posix': # linux
    extra_compile_args = ["-std=c++11", "-fopenmp", "-D_GLIBCXX_PARALLEL",
                          "-DMIN_OPS_PER_THREAD=10000"]
    extra_link_args = ["-lgomp"]
else:
    raise NotImplementedError('OS not yet supported.')

###  auxiliary functions  ###
class build_class(build):
    def initialize_options(self):
        build.initialize_options(self)
        self.build_lib = "bin" 
    def run(self):
        build_path = self.build_lib

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))

###  preprocessing  ###
# ensure right working directory
tmp_work_dir = os.path.realpath(os.curdir)
os.chdir(os.path.realpath(os.path.dirname(__file__)))

if not os.path.exists("bin"):
    os.mkdir("bin")

# remove previously compiled lib
for shared_obj in to_compile: 
    purge("bin/", shared_obj)

###  compilation  ###

name = "cp_pfdr_d1_ql1b_cpy"
if name in to_compile:
    mod = Extension(
            name,
            # list source files
            ["cpython/cp_pfdr_d1_ql1b_cpy.cpp", "../src/cp_pfdr_d1_ql1b.cpp",
             "../src/cut_pursuit_d1.cpp", "../src/cut_pursuit.cpp",
             "../src/maxflow.cpp", "../src/pfdr_d1_ql1b.cpp",
             "../src/matrix_tools.cpp", "../src/pfdr_graph_d1.cpp", 
             "../src/pcd_fwd_doug_rach.cpp", "../src/pcd_prox_split.cpp"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args)
    setup(name=name, ext_modules=[mod], cmdclass=dict(build=build_class))


name = "cp_pfdr_d1_lsx_cpy"
if name in to_compile:
    mod = Extension(
            name,
            # list source files
            ["cpython/cp_pfdr_d1_lsx_cpy.cpp", "../src/cp_pfdr_d1_lsx.cpp",
             "../src/cut_pursuit_d1.cpp", "../src/cut_pursuit.cpp",
             "../src/maxflow.cpp", "../src/pfdr_d1_lsx.cpp",
             "../src/proj_simplex.cpp", "../src/pfdr_graph_d1.cpp",
             "../src/pcd_fwd_doug_rach.cpp", "../src/pcd_prox_split.cpp"], 
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args)
    setup(name=name, ext_modules=[mod], cmdclass=dict(build=build_class))


name = "cp_kmpp_d0_dist_cpy"
if name in to_compile:
    mod = Extension(
            name,
            # list source files
            ["cpython/cp_kmpp_d0_dist_cpy.cpp", "../src/cp_kmpp_d0_dist.cpp",
             "../src/cut_pursuit_d0.cpp", "../src/cut_pursuit.cpp",
             "../src/maxflow.cpp"], 
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args)
    setup(name=name, ext_modules=[mod], cmdclass=dict(build=build_class))

###  postprocessing  ###
try:
    shutil.rmtree("build") # remove temporary compilation products
except FileNotFoundError:
    pass

os.chdir(tmp_work_dir) # get back to initial working directory

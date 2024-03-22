from setuptools import setup
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = os.path.dirname(__file__)

include_dirs = [
    os.path.join(ROOT_DIR, 'include'),
    os.path.join(ROOT_DIR, 'dependencies', 'eigen'),
    os.path.join(ROOT_DIR, 'dependencies', 'pybind11', 'include'),
    '/usr/local/include',
    '/usr/include',
    '/usr/local/cuda-11.1/targets/x86_64-linux/include'
]

print(include_dirs)

module = CUDAExtension(
    name='pyngpmesh',
    sources=['src/python-api.cu', 'src/NGPMesh.cu', 'src/TriangleBvh.cu'],
    include_dirs=include_dirs,
    define_macros=[('EIGEN_GPUCC', None)],
    extra_compile_args={
        'nvcc': ['--extended-lambda', 
                #  "-Xcompiler=-mf16c",
                 "-Xcompiler=-Wno-float-conversion",
                 "-Xcompiler=-fno-strict-aliasing",
                 "-Xcompiler=-fPIC",
                 "--expt-relaxed-constexpr"]
    }
)


setup(name='pyngpmesh',
      version='1.0',
      author='Dray Ken',
      author_email='dray_ken@163.com',
      description='Tracer extracted from instant ngp',
      ext_modules=[module],
      cmdclass={
          'build_ext': BuildExtension
      })
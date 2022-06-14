import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# If you have gcc version 7, change below to 'gcc-7'
os.environ["CC"] = "gcc-8"
os.environ["CXX"] = "gcc-8"

setup(
    name='rasterize_cuda',
    ext_modules=[
	CUDAExtension('rasterize_cuda', [
        'rasterize_cuda.cpp',
        'rasterize_cuda_kernel.cu',
        ])
	],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}
)
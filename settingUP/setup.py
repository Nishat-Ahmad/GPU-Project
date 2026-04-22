import os

# 1. FORCE THE PATHS FIRST! BEFORE ANY IMPORTS!
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
os.environ["CUDA_HOME"] = cuda_path
os.environ["CUDA_PATH"] = cuda_path

# 2. NOW we are allowed to import PyTorch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 3. Build the extension
setup(
    name='dummy_ext',
    ext_modules=[
        CUDAExtension(
            name='dummy_ext', 
            sources=['dummy.cu']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
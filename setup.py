# -*- coding: utf-8 -*-
from os import sys

try:
    from skbuild import setup
except ImportError:
    print('scikit-build is required to build from source.', file=sys.stderr)
    print('Please run:', file=sys.stderr)
    print('', file=sys.stderr)
    print('  python -m pip install scikit-build')
    sys.exit(1)

# Configure wheel name if CUDA is used
wheel_name='itk-cudacommon'
# Extract cuda version from the CUDACOMMON_CUDA_VERSION cmake option
for arg in sys.argv:
  if "CUDACOMMON_CUDA_VERSION" in str(arg):
    cuda_version = arg.rsplit('CUDACOMMONG_CUDA_VERSION=', 1)[-1]
    wheel_name += '-cuda' + cuda_version.replace('.', '')

setup(
    name=wheel_name,
    version='1.0.0',
    author='RTK Consortium',
    author_email='simon.rit@creatis.insa-lyon.fr',
    packages=['itk'],
    package_dir={'itk': 'itk'},
    download_url=r'https://github.com/RTKConsortium/ITKCudaCommon',
    description=r'Framework for processing images with Cuda.',
    long_description='This module contains the framework for processing images with Cuda. This includes base classes for the Cuda image filters and classes to manage memory transfers between the CPU and GPU.',
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: C++",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries",
        "Operating System :: Android",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS"
        ],
    license='Apache',
    url=r'https://github.com/RTKConsortium/ITKCudaCommon',
    install_requires=[
        r'itk>=5.3rc04'
    ]
    )

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

setup(
    name='itk-cudacommon',
    version='0.1.0',
    author='RTK Consortium',
    author_email='simon.rit@creatis.insa-lyon.fr',
    packages=['itk'],
    package_dir={'itk': 'itk'},
    download_url=r'https://github.com/SimonRit/ITKCudaCommon',
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
    keywords='RTK Reconstruction Toolkit',
    url=r'https://www.openrtk.org/',
    install_requires=[
        r'itk>=5.2.1.post1'
    ]
    )

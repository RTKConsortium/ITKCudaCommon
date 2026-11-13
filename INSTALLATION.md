# CudaCommon installation

## Configuration, compilation and installation with ITK
CudaCommon is a module of [ITK](https://www.itk.org), the Insight Toolkit. Follow the instructions of the [ITK software guide](https://itk.org/ITKSoftwareGuide/html) ([chapter 2](https://itk.org/ITKSoftwareGuide/html/Book1/ITKSoftwareGuide-Book1ch2.html) mainly) for configuring and compiling ITK. The following CMake options are CudaCommon specific:

* `Module_CudaCommon`: Activates CudaCommon download and compilation. Default is `OFF`. Turn it `ON` to activate CudaCommon or compile CudaCommon independently (see below).
* `Module_CudaCommon_GIT_TAG`: Git tag for the CudaCommon download. By default, the CudaCommon version which is downloaded and compiled is the one given in the [CudaCommon.remote.cmake](https://github.com/InsightSoftwareConsortium/ITK/blob/main/Modules/Remote/CudaCommon.remote.cmake). Change this option to build another version. For example, you can change it to `main` to build the latest CudaCommon version. CudaCommon is only maintained to be backward compatible with the latest ITK release and ITK main branch.
* `CUDACOMMON_CUDA_VERSION`: Specifies the exact CUDA version that must be used by CudaCommon.

CudaCommon will automatically be installed when installing ITK.

### CUDA GPU architectures

The default CUDA architecture is `75` (Turing). Builds using this default target Turing and newer GPUs only. To build from source for an older GPU (Maxwell, Pascal or Volta), explicitly set `CMAKE_CUDA_ARCHITECTURES` to its compute capability (for example, `-DCMAKE_CUDA_ARCHITECTURES=52` for compute capability 5.2) and use a compatible CUDA toolkit. CUDA 13 no longer supports compilation for architectures below `75`. Use CUDA 11 or 12 for those GPUs.

## Independent configuration and compilation
For CudaCommon developers, it may be useful to compile CudaCommon independently from ITK. This is possible, simply:
* Compile ITK with `Module_CudaCommon=OFF`.
* Manually download CudaCommon's source repository from [GitHub](https://github.com/RTKConsortium/ITKCudaCommon) with `git` (recommended) or as a [zip package](https://codeload.github.com/RTKConsortium/ITKCudaCommon/zip/main).
* Configure the project with CMake pointing to CudaCommon's source directory and setting the CMake option `ITK_DIR` to ITK's compilation directory. All CMake options above can be set except `Module_CudaCommon`.

### Installing the Python wrapping into a Python environment

The recommended way is to create a self-contained *Python packages folder* holding ITK's compiled Python extensions and install ITK's Python package into it, built with `-DITK_WRAP_PYTHON=ON -DPY_SITE_PACKAGES_PATH=/path/to/python_packages_folder` and installed with `cmake --install <itk-build-dir> --component RuntimeLibraries`. The extensions are built for a specific Python ABI and platform, so the folder is portable across environments that support that ABI/platform; using a different Python version or OS requires rebuilding ITK (and CudaCommon).

[utilities/cudacommon_install_python_packages.sh](https://github.com/RTKConsortium/ITKCudaCommon/blob/main/utilities/cudacommon_install_python_packages.sh) automates the CudaCommon side: it configures, builds and installs CudaCommon into the same folder with CMake (`-DPY_SITE_PACKAGES_PATH=$DEST_DIR`, `cmake --install --component RuntimeLibraries`), makes the folder importable with a `.pth` file, and symlinks CudaCommon's Python modules to the source tree so their edits are taken into account without reinstalling. Run it with `DEST_DIR` set to the same folder given by `-DPY_SITE_PACKAGES_PATH` above:

```
ITK_DIR=<itk-build-dir> DEST_DIR=/path/to/python_packages_folder utilities/cudacommon_install_python_packages.sh
```

The packages folder holds compiled extensions for a specific Python ABI/platform, but the `.pth` file that makes it importable is added by the script to the `site-packages` of the Python interpreter it runs with (the one given by `PYTHON`, `python3` by default), or to its user `site-packages` if the former is not writable, e.g. for a system Python outside a virtual environment. To use it in another environment, run the script with `PYTHON=/path/to/other/env/bin/python ...` or export `PYTHONPATH=$DEST_DIR`.

CudaCommon is built in the `Release` configuration. Set `CMAKE_BUILD_TYPE` (e.g. `CMAKE_BUILD_TYPE=Debug`) to match another configuration of ITK.

## Python pre-compiled binaries
We only provide pre-compiled binaries for the Python package which depends on ITK. They require an installed version of CUDA compatible with the package. CudaCommon is available on PyPI as one package per CUDA version, `itk-cudacommon-cuda<version>`: CUDA 11.8 and 12.4 for Linux, CUDA 12.8 and 13.0 for Linux and Windows. For example, for CUDA 12.8:
```
python -m pip install --upgrade pip
python -m pip install itk-cudacommon-cuda128
```
The supported Python versions and platforms are listed with each package's files on PyPI, e.g. [itk-cudacommon-cuda128](https://pypi.org/project/itk-cudacommon-cuda128/#files).

To try the newest features before they are published on PyPI, download the most recent wheels produced by our [CI workflow](https://github.com/RTKConsortium/ITKCudaCommon/actions/workflows/build-test-package-python-cuda.yml?query=branch%3Amain+is%3Asuccess)

Each workflow shows a list of *Artifacts* at the bottom of its front page if you are logged in on GitHub. Just select the version corresponding to your operating system and Python version, unzip the wheel and `pip install wheel_filename`.

### Testing
To run CudaCommon Python tests with pytest, install the test dependencies from the directory containing `pyproject.toml` (`--group` requires pip 25.1 or later):

```
python -m pip install --group test-cuda12 itk-cudacommon-cuda128
```

This installs pytest, torch and cupy, which the Python tests need. Use the group matching the major CUDA version of your package (`test-cuda11`, `test-cuda12` or `test-cuda13`) so that the CuPy build relies on the same CUDA runtime.

After installing, run the tests with:
```
pytest -o python_files="*Test.py" test
```

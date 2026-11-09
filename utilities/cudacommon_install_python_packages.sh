#!/usr/bin/env bash

# Install CudaCommon's Python packages, by CMake, into DEST_DIR (a folder
# holding the ITK Python package). Does not build ITK; ITK_DIR is the ITK build
# tree (with Python wrapping). CMAKE_BUILD_TYPE (Release by default) should
# match the configuration of ITK.
# Set inputs via environment variables, e.g.:
#   ITK_DIR=/path/to/itk/build DEST_DIR=/path/to/python_packages \
#   cudacommon_install_python_packages.sh

set -xe

PYTHON=${PYTHON:-python3}
NTHREADS=${NTHREADS:-$("$PYTHON" -c "import os; print(os.cpu_count() or 1)")}
CudaCommon_BUILD_DIR=${CudaCommon_BUILD_DIR:-$PWD/build-cudacommon-python}
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}

ITK_DIR=${ITK_DIR:?"ITK_DIR must be set (ITK build tree with Python wrapping)"}
DEST_DIR=${DEST_DIR:?"DEST_DIR must be set (folder that already holds the itk package)"}

CudaCommon_SRC=$(cd "$(dirname "$0")/.." && pwd)
ITK_DIR=$(cd "$ITK_DIR" && pwd)
mkdir -p "$DEST_DIR"
DEST_DIR=$(cd "$DEST_DIR" && pwd)
mkdir -p "$CudaCommon_BUILD_DIR"
CudaCommon_BUILD_DIR=$(cd "$CudaCommon_BUILD_DIR" && pwd)

# Pick the site-packages folder that will receive the .pth file before building:
# $PYTHON's own if writable (e.g. a virtual environment), else the user site.
SITE_PACKAGES=$("$PYTHON" -c '
import os, site, sys
d = site.getsitepackages()[0]
if not os.access(d, os.W_OK):
    if not site.ENABLE_USER_SITE:
        sys.exit(f"Error: {d} is not writable and the user site is disabled. "
                 "Set PYTHON to the python of a virtual environment.")
    d = site.getusersitepackages()
print(d)
')
mkdir -p "$SITE_PACKAGES"

echo "ITK build tree:       $ITK_DIR"
echo "CudaCommon source:    $CudaCommon_SRC"
echo "CudaCommon build dir: $CudaCommon_BUILD_DIR"
echo "Install folder:       $DEST_DIR"
echo "Python:               $PYTHON"
echo "Build type:           $CMAKE_BUILD_TYPE"
echo ".pth folder:          $SITE_PACKAGES"

# The Python wrapping must already be installed in DEST_DIR.
if [ ! -f "$DEST_DIR/itk/__init__.py" ]; then
  echo "Error: no Python wrapping found in $DEST_DIR/itk. Configure ITK with" >&2
  echo "-DITK_WRAP_PYTHON=ON -DPY_SITE_PACKAGES_PATH=$DEST_DIR, build it, then" >&2
  echo "install the wrapping into $DEST_DIR with:" >&2
  echo "  cmake --install $ITK_DIR --component RuntimeLibraries" >&2
  echo "and run this script again." >&2
  exit 1
fi

# Building CudaCommon (an external ITK module) requires an ITK build tree.
if [ ! -f "$ITK_DIR/ITKConfig.cmake" ]; then
  echo "Error: $ITK_DIR is not an ITK build tree (missing ITKConfig.cmake)." >&2
  exit 1
fi

# Configure, build and install CudaCommon. PY_SITE_PACKAGES_PATH merges the wrapping
# into the itk package in DEST_DIR; RuntimeLibraries matches the itk install.
# CMAKE_BUILD_TYPE is passed to every step so that multi-configuration
# generators (Visual Studio, Ninja Multi-Config) build and install the same
# configuration.
cmake -S "$CudaCommon_SRC" -B "$CudaCommon_BUILD_DIR" \
  -DITK_DIR="$ITK_DIR" \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -DBUILD_TESTING=OFF \
  -DPY_SITE_PACKAGES_PATH="$DEST_DIR"
cmake --build "$CudaCommon_BUILD_DIR" --config "$CMAKE_BUILD_TYPE" -j "$NTHREADS"
cmake --install "$CudaCommon_BUILD_DIR" --config "$CMAKE_BUILD_TYPE" --component RuntimeLibraries

# Symlink hand-written Python (apps and support modules) to the source so edits
# are picked up without reinstalling; generated wrapping still needs a rebuild.
for src in \
    "$CudaCommon_SRC/wrapping/__init_cudacommon__.py" \
    "$CudaCommon_SRC/wrapping/itkCudaImageFromCudaArray.py" \
    "$CudaCommon_SRC/wrapping/itkCudaImageFromImage.py" \
    "$CudaCommon_BUILD_DIR/Wrapping/Generators/Python/cudacommonConfig.py"; do
  [ -f "$src" ] || continue
  ln -sf "$src" "$DEST_DIR/itk/$(basename "$src")"
done
echo "CudaCommon Python modules in $DEST_DIR/itk are symlinked to the CudaCommon source tree, so their edits are taken into account immediately."

# Make DEST_DIR importable and give it priority over installed packages.
PTH_FILE="$SITE_PACKAGES/cudacommon_python_packages.pth"

"$PYTHON" -c 'import pathlib, sys; pathlib.Path(sys.argv[2]).write_text(f"import sys; sys.path.insert(0, {sys.argv[1]!r})\n")' \
  "$DEST_DIR" "$PTH_FILE"

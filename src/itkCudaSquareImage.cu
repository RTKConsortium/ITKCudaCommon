/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "itkCudaSquareImage.hcu"

namespace itk
{

template<class PixelType>
__global__ void CudaSquareImage3D_kernel(int3 imSize, PixelType* in, PixelType* out)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= imSize.x || j >= imSize.y || k >= imSize.z)
  {
    return;
  }
  unsigned int gidx = i + j * imSize.x + k * imSize.x * imSize.y;

  out[gidx] = in[gidx] * in[gidx];
}

template<class PixelType>
void CudaSquareImage3D(int imSize[3], PixelType* in, PixelType* out)
{
  // Thread Block Dimensions
  constexpr int tBlock_x = 16;
  constexpr int tBlock_y = 4;
  constexpr int tBlock_z = 4;

  unsigned int blocksInX = (imSize[0] - 1) / tBlock_x + 1;
  unsigned int blocksInY = (imSize[1] - 1) / tBlock_y + 1;
  unsigned int blocksInZ = (imSize[2] - 1) / tBlock_z + 1;

  // Compute block and grid sizes
  dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

  int3 imageSize = make_int3(imSize[0], imSize[1], imSize[2]);

  CudaSquareImage3D_kernel<PixelType> << <dimGrid, dimBlock >> > (imageSize, in, out);
}

template void CudaSquareImage3D<float>(int imSize[3], float* in, float* out);
template void CudaSquareImage3D<double>(int imSize[3], double* in, double* out);

} // end namespace itk

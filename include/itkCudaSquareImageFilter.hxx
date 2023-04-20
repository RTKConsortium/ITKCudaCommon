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
#ifndef itkCudaSquareImageFilter_hxx
#define itkCudaSquareImageFilter_hxx

#include "itkCudaSquareImage.hcu"

namespace itk
{
  template <class ImageType>
  void CudaSquareImageFilter<ImageType>::GPUGenerateData()
  {
    int size[3] = {1, 1, 1};
    for (unsigned int i = 0; i < ImageDimension; i++)
    {
      size[i] = this->GetInput()->GetBufferedRegion().GetSize()[i];
    }

    typename ImageType::PixelType* pin0 = *(typename ImageType::PixelType**)(this->GetInput(0)->GetCudaDataManager()->GetGPUBufferPointer());
    typename ImageType::PixelType* pout = *(typename ImageType::PixelType**)(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());

    CudaSquareImage3D<typename ImageType::PixelType>(size, pin0, pout);
  }

} // end namespace itk

#endif // end conditional definition of the class

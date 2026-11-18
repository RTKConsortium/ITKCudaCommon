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
#ifndef itkCudaXorImageFilter_h
#define itkCudaXorImageFilter_h

#include "itkCudaBinaryImageToImageFilter.h"
#include "itkCudaBinaryImageOperators.h"
#include "itkXorImageFilter.h"

namespace itk
{

/** \class CudaXorImageFilter
 * \brief CUDA implementation of itk::XorImageFilter for integer pixels.
 *
 * Supports two images or an image and a constant, with the same pixel type.
 *
 * \ingroup ITKCudaCommon
 */
template <class TImage>
class ITK_TEMPLATE_EXPORT CudaXorImageFilter
  : public CudaBinaryImageToImageFilter<TImage, TImage, XorImageFilter<TImage, TImage, TImage>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaXorImageFilter);

  using Self = CudaXorImageFilter;
  using Superclass = CudaBinaryImageToImageFilter<TImage, TImage, XorImageFilter<TImage, TImage, TImage>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(CudaXorImageFilter);

protected:
  CudaXorImageFilter() = default;
  ~CudaXorImageFilter() override = default;

  using PixelType = typename Superclass::PixelType;

  void
  GenerateBinaryData(const PixelType *  input1,
                     const PixelType *  input2,
                     PixelType          constant1,
                     PixelType          constant2,
                     PixelType *        output,
                     itk::SizeValueType numberOfPixels) override
  {
    CudaXorImage(numberOfPixels, input1, input2, output, constant1, constant2);
  }
};

} // namespace itk

#endif

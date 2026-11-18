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
#ifndef itkCudaMultiplyImageFilter_h
#define itkCudaMultiplyImageFilter_h

#include "itkCudaBinaryImageToImageFilter.h"
#include "itkCudaBinaryImageOperators.h"
#include "itkMultiplyImageFilter.h"

namespace itk
{

/** \class CudaMultiplyImageFilter
 * \brief CUDA implementation of itk::MultiplyImageFilter.
 *
 * Supports two images or an image and a constant, with the same pixel type.
 * CPU execution uses itk::MultiplyImageFilter. Common GPU buffer checks are
 * provided by CudaBinaryImageToImageFilter.
 *
 * \ingroup ITKCudaCommon
 */
template <class TImage>
class ITK_TEMPLATE_EXPORT CudaMultiplyImageFilter
  : public CudaBinaryImageToImageFilter<TImage, TImage, MultiplyImageFilter<TImage, TImage, TImage>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaMultiplyImageFilter);

  using Self = CudaMultiplyImageFilter;
  using Superclass = CudaBinaryImageToImageFilter<TImage, TImage, MultiplyImageFilter<TImage, TImage, TImage>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(CudaMultiplyImageFilter);

protected:
  CudaMultiplyImageFilter() = default;
  ~CudaMultiplyImageFilter() override = default;

  using PixelType = typename Superclass::PixelType;

  void
  GenerateBinaryData(const PixelType *  input1,
                     const PixelType *  input2,
                     PixelType          constant1,
                     PixelType          constant2,
                     PixelType *        output,
                     itk::SizeValueType numberOfPixels) override
  {
    CudaMultiplyImage(numberOfPixels, input1, input2, output, constant1, constant2);
  }
};

} // namespace itk

#endif

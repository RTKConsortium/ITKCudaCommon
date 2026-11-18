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
#ifndef itkCudaSubtractImageFilter_h
#define itkCudaSubtractImageFilter_h

#include "itkCudaBinaryImageToImageFilter.h"
#include "itkCudaBinaryImageOperators.h"
#include "itkSubtractImageFilter.h"

namespace itk
{

/** \class CudaSubtractImageFilter
 * \brief CUDA implementation of itk::SubtractImageFilter.
 *
 * Supports two images or an image and a constant, with the same pixel type.
 * CPU execution uses itk::SubtractImageFilter. Common GPU buffer checks are
 * provided by CudaBinaryImageToImageFilter.
 *
 * \ingroup ITKCudaCommon
 */
template <class TImage>
class ITK_TEMPLATE_EXPORT CudaSubtractImageFilter
  : public CudaBinaryImageToImageFilter<TImage, TImage, SubtractImageFilter<TImage, TImage, TImage>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaSubtractImageFilter);

  using Self = CudaSubtractImageFilter;
  using Superclass = CudaBinaryImageToImageFilter<TImage, TImage, SubtractImageFilter<TImage, TImage, TImage>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(CudaSubtractImageFilter);

protected:
  CudaSubtractImageFilter() = default;
  ~CudaSubtractImageFilter() override = default;

  using PixelType = typename Superclass::PixelType;

  void
  GenerateBinaryData(const PixelType *  input1,
                     const PixelType *  input2,
                     PixelType          constant1,
                     PixelType          constant2,
                     PixelType *        output,
                     itk::SizeValueType numberOfPixels) override
  {
    CudaSubtractImage(numberOfPixels, input1, input2, output, constant1, constant2);
  }
};

} // namespace itk

#endif

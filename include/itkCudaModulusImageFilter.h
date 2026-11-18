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
#ifndef itkCudaModulusImageFilter_h
#define itkCudaModulusImageFilter_h

#include "itkCudaBinaryImageToImageFilter.h"
#include "itkCudaBinaryImageOperators.h"
#include "itkModulusImageFilter.h"

namespace itk
{

/** \class CudaModulusImageFilter
 * \brief CUDA implementation of itk::ModulusImageFilter for integer pixels.
 *
 * Supports two images or an image and a constant. A zero divisor returns
 * the maximum pixel value, matching ITK. SetDividend() sets the second constant.
 *
 * \ingroup ITKCudaCommon
 */
template <class TImage>
class ITK_TEMPLATE_EXPORT CudaModulusImageFilter
  : public CudaBinaryImageToImageFilter<TImage, TImage, ModulusImageFilter<TImage, TImage, TImage>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaModulusImageFilter);

  using Self = CudaModulusImageFilter;
  using Superclass = CudaBinaryImageToImageFilter<TImage, TImage, ModulusImageFilter<TImage, TImage, TImage>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(CudaModulusImageFilter);

protected:
  CudaModulusImageFilter() = default;
  ~CudaModulusImageFilter() override = default;

  using PixelType = typename Superclass::PixelType;

  void
  GenerateBinaryData(const PixelType *  input1,
                     const PixelType *  input2,
                     PixelType          constant1,
                     PixelType          constant2,
                     PixelType *        output,
                     itk::SizeValueType numberOfPixels) override
  {
    CudaModulusImage(numberOfPixels, input1, input2, output, constant1, constant2);
  }
};

} // namespace itk

#endif

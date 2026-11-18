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
#ifndef itkCudaMaskImageFilter_h
#define itkCudaMaskImageFilter_h

#include "itkCudaBinaryImageToImageFilter.h"
#include "itkCudaBinaryImageOperators.h"
#include "itkMaskImageFilter.h"

namespace itk
{

/** \class CudaMaskImageFilter
 * \brief CUDA implementation of itk::MaskImageFilter.
 *
 * The image and mask may have different scalar pixel types. Output has the
 * image's type. Uses the inherited MaskingValue and
 * OutsideValue settings. Supports image/image and image/constant operands.
 *
 * \note Mask pixel types are limited to unsigned char, unsigned short, short
 * and the image pixel type, see CudaCommon_INSTANTIATE_MASKS in
 * itkCudaBinaryImageOperators.cu.
 *
 * \ingroup ITKCudaCommon
 */
template <class TImage, class TMaskImage = TImage>
class ITK_TEMPLATE_EXPORT CudaMaskImageFilter
  : public CudaBinaryImageToImageFilter<TImage, TMaskImage, MaskImageFilter<TImage, TMaskImage, TImage>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaMaskImageFilter);

  using Self = CudaMaskImageFilter;
  using Superclass = CudaBinaryImageToImageFilter<TImage, TMaskImage, MaskImageFilter<TImage, TMaskImage, TImage>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(CudaMaskImageFilter);

protected:
  CudaMaskImageFilter() = default;
  ~CudaMaskImageFilter() override = default;

  using PixelType = typename Superclass::PixelType;
  using MaskPixelType = typename Superclass::Input2PixelType;

  void
  GenerateBinaryData(const PixelType *     input,
                     const MaskPixelType * mask,
                     PixelType             constant1,
                     MaskPixelType         constant2,
                     PixelType *           output,
                     itk::SizeValueType    numberOfPixels) override
  {
    CudaMaskImage(
      numberOfPixels, input, mask, output, constant1, constant2, this->GetMaskingValue(), this->GetOutsideValue());
  }
};

} // namespace itk

#endif

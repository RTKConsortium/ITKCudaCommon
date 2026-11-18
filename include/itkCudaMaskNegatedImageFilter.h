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
#ifndef itkCudaMaskNegatedImageFilter_h
#define itkCudaMaskNegatedImageFilter_h

#include "itkCudaBinaryImageToImageFilter.h"
#include "itkCudaBinaryImageOperators.h"
#include "itkMaskNegatedImageFilter.h"

namespace itk
{

/** \class CudaMaskNegatedImageFilter
 * \brief CUDA implementation of itk::MaskNegatedImageFilter.
 *
 * The image and mask may have different scalar pixel types. Output has the
 * image's type. Keeps pixels where the mask equals the
 * inherited MaskingValue and uses OutsideValue elsewhere. Supports image/image
 * and image/constant operands.
 *
 * \note Mask pixel types are limited to unsigned char, unsigned short, short
 * and the image pixel type, see CudaCommon_INSTANTIATE_MASKS in
 * itkCudaBinaryImageOperators.cu.
 *
 * \ingroup ITKCudaCommon
 */
template <class TImage, class TMaskImage = TImage>
class ITK_TEMPLATE_EXPORT CudaMaskNegatedImageFilter
  : public CudaBinaryImageToImageFilter<TImage, TMaskImage, MaskNegatedImageFilter<TImage, TMaskImage, TImage>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaMaskNegatedImageFilter);

  using Self = CudaMaskNegatedImageFilter;
  using Superclass =
    CudaBinaryImageToImageFilter<TImage, TMaskImage, MaskNegatedImageFilter<TImage, TMaskImage, TImage>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(CudaMaskNegatedImageFilter);

protected:
  CudaMaskNegatedImageFilter() = default;
  ~CudaMaskNegatedImageFilter() override = default;

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
    CudaMaskNegatedImage(
      numberOfPixels, input, mask, output, constant1, constant2, this->GetMaskingValue(), this->GetOutsideValue());
  }
};

} // namespace itk

#endif

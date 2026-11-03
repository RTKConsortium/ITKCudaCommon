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
#ifndef itkCudaBinaryGeneratorImageFilter_h
#define itkCudaBinaryGeneratorImageFilter_h

#include "itkCudaBinaryGeneratorImageFilter.hcu"
#include "itkCudaImageToImageFilter.h"
#include "itkBinaryGeneratorImageFilter.h"

#include <type_traits>

namespace itk
{

/** \class CudaBinaryGeneratorImageFilter
 * Shared CUDA execution path for ITK binary generator filters. The concrete
 * filter supplies the matching ITK parent (for CPU fallback) and its GPU
 * operation, while this class handles image/constant inputs and GPU buffers.
 *
 * \ingroup ITKCudaCommon
 */
template <typename TInputImage1, typename TInputImage2, typename TOutputImage, typename TParentImageFilter>
class ITK_TEMPLATE_EXPORT CudaBinaryGeneratorImageFilter
  : public CudaImageToImageFilter<TInputImage1, TOutputImage, TParentImageFilter>
{
public:
  using Self = CudaBinaryGeneratorImageFilter;
  using Superclass = CudaImageToImageFilter<TInputImage1, TOutputImage, TParentImageFilter>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  using Input1ImageType = TInputImage1;
  using Input1ImagePixelType = typename Input1ImageType::PixelType;
  using Input2ImageType = TInputImage2;
  using Input2ImagePixelType = typename Input2ImageType::PixelType;
  using OutputImageType = TOutputImage;
  using OutputImagePixelType = typename OutputImageType::PixelType;

  static_assert(std::is_same_v<Input1ImagePixelType, Input2ImagePixelType> &&
                  std::is_same_v<Input1ImagePixelType, OutputImagePixelType>,
                "CUDA binary arithmetic requires matching input and output pixel types");

  static constexpr unsigned int InputImage1Dimension = TInputImage1::ImageDimension;
  static constexpr unsigned int InputImage2Dimension = TInputImage2::ImageDimension;
  static constexpr unsigned int OutputImageDimension = TOutputImage::ImageDimension;

protected:
  CudaBinaryGeneratorImageFilter() = default;
  ~CudaBinaryGeneratorImageFilter() override = default;

  void
  GPUGenerateData() override = 0;

  template <CudaBinaryImageOperation VOperation>
  void
  LaunchOperation();
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCudaBinaryGeneratorImageFilter.hxx"
#endif

#endif

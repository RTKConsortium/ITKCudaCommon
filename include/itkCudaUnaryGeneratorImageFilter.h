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
#ifndef itkCudaUnaryGeneratorImageFilter_h
#define itkCudaUnaryGeneratorImageFilter_h

#include "itkCudaImageToImageFilter.h"
#include "itkCudaUnaryGeneratorImageFilter.hcu"

#include <type_traits>

namespace itk
{
/** \class CudaUnaryGeneratorImageFilter
 * Shared CUDA execution path for ITK unary image filters. The concrete filter
 * supplies the matching ITK parent (for CPU fallback) and its GPU operation;
 * this class handles the CUDA image buffers and output region.
 *
 * \ingroup ITKCudaCommon
 */
template <typename TInputImage, typename TOutputImage, typename TParentImageFilter>
class ITK_TEMPLATE_EXPORT CudaUnaryGeneratorImageFilter
  : public CudaImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>
{
public:
  using Self = CudaUnaryGeneratorImageFilter;
  using Superclass = CudaImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;
  using InputImageType = TInputImage;
  using InputImagePixelType = typename TInputImage::PixelType;
  using OutputImageType = TOutputImage;
  using OutputImagePixelType = typename TOutputImage::PixelType;

  static_assert(std::is_same_v<InputImagePixelType, OutputImagePixelType>,
                "CUDA unary arithmetic requires matching input and output pixel types");

  static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;
  static constexpr unsigned int OutputImageDimension = TOutputImage::ImageDimension;

protected:
  CudaUnaryGeneratorImageFilter() = default;
  ~CudaUnaryGeneratorImageFilter() override = default;

  void
  GPUGenerateData() override = 0;

  template <CudaUnaryImageOperation VOperation>
  void
  LaunchOperation();
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCudaUnaryGeneratorImageFilter.hxx"
#endif

#endif

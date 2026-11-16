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
#ifndef itkCudaUnaryImageToImageFilter_h
#define itkCudaUnaryImageToImageFilter_h

#include "itkCudaImageToImageFilter.h"

namespace itk
{

/** \class CudaUnaryImageToImageFilter
 *
 * \brief Base class for pixel-wise CUDA filters with one input image.
 *
 * This class mirrors itk::UnaryGeneratorImageFilter: it owns the common GPU
 * plumbing for one-input, one-output filters, so that concrete operators only
 * implement the device operation.
 *
 * GPU execution requires the input and output buffered regions to exactly
 * match the output requested region. This allows streaming in pieces as long
 * as each piece has its own equally sized input and output buffers.
 *
 * \ingroup ITKCudaCommon
 */
template <class TInputImage,
          class TOutputImage = TInputImage,
          class TParentImageFilter = ImageToImageFilter<TInputImage, TOutputImage>>
class ITK_TEMPLATE_EXPORT CudaUnaryImageToImageFilter
  : public CudaImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>
{
public:
  /** Standard class type alias. */
  using Self = CudaUnaryImageToImageFilter;
  using Superclass = CudaImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(CudaUnaryImageToImageFilter);

  ITK_DISALLOW_COPY_AND_MOVE(CudaUnaryImageToImageFilter);

  using InputPixelType = typename TInputImage::PixelType;
  using OutputPixelType = typename TOutputImage::PixelType;

protected:
  CudaUnaryImageToImageFilter() = default;
  ~CudaUnaryImageToImageFilter() override = default;

  /** Validate the regions and forward the device buffers to the concrete
   * operation. This is the only place the unary checks are written.
   */
  void
  GPUGenerateData() final;

  /** Implement the device operation. The input and output buffers cover exactly
   * the requested region and contain numberOfPixels elements.
   */
  virtual void
  GenerateUnaryData(const InputPixelType * input, OutputPixelType * output, itk::SizeValueType numberOfPixels) = 0;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCudaUnaryImageToImageFilter.hxx"
#endif

#endif

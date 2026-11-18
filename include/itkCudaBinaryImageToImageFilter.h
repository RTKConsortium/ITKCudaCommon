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
#ifndef itkCudaBinaryImageToImageFilter_h
#define itkCudaBinaryImageToImageFilter_h

#include "itkCudaImageToImageFilter.h"
#include "itkBinaryGeneratorImageFilter.h"

namespace itk
{

/** \class CudaBinaryImageToImageFilter
 * \brief Common GPU handling for two images or an image and a constant.
 *
 * Each image's buffered region must match the output requested region exactly.
 * Concrete filters implement GenerateBinaryData() using the prepared buffers.
 * The first input and output use TImage; only the second input (e.g. a mask)
 * may have a different pixel type.
 * CPU execution uses TParentImageFilter.
 *
 * \ingroup ITKCudaCommon
 */
template <class TImage,
          class TInputImage2 = TImage,
          class TParentImageFilter = BinaryGeneratorImageFilter<TImage, TInputImage2, TImage>>
class ITK_TEMPLATE_EXPORT CudaBinaryImageToImageFilter
  : public CudaImageToImageFilter<TImage, TImage, TParentImageFilter>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaBinaryImageToImageFilter);

  using Self = CudaBinaryImageToImageFilter;
  using Superclass = CudaImageToImageFilter<TImage, TImage, TParentImageFilter>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkOverrideGetNameOfClassMacro(CudaBinaryImageToImageFilter);

  using PixelType = typename TImage::PixelType;
  using Input2PixelType = typename TInputImage2::PixelType;

protected:
  CudaBinaryImageToImageFilter() = default;
  ~CudaBinaryImageToImageFilter() override = default;

  void
  GPUGenerateData() final;

  /** A null input pointer represents the corresponding constant operand. */
  virtual void
  GenerateBinaryData(const PixelType *       input1,
                     const Input2PixelType * input2,
                     PixelType               constant1,
                     Input2PixelType         constant2,
                     PixelType *             output,
                     itk::SizeValueType      numberOfPixels) = 0;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCudaBinaryImageToImageFilter.hxx"
#endif

#endif

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
#ifndef itkCudaImageFromImageFilter_h
#define itkCudaImageFromImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkCudaImage.h"

namespace itk
{

/** \class CudaImageFromImageFilter
 * \brief Adaptor producing an itk::CudaImage that zero‑copy shares the input host buffer.
 *
 * This filter takes an \c itk::Image as input and outputs an \c itk::CudaImage with the
 * same pixel type and dimension. The output \c PixelContainer pointer is set to the
 * input's container (no allocation or deep copy). Image meta‑information (origin,
 * spacing, direction, largest & buffered regions) is preserved through the standard
 * pipeline.
 *
 * \ingroup ITKCudaCommon
 */
template <typename TInputImage>
class ITK_TEMPLATE_EXPORT CudaImageFromImageFilter
  : public ImageToImageFilter<TInputImage, CudaImage<typename TInputImage::PixelType, TInputImage::ImageDimension>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CudaImageFromImageFilter);

  using Self = CudaImageFromImageFilter;
  using Superclass =
    ImageToImageFilter<TInputImage, CudaImage<typename TInputImage::PixelType, TInputImage::ImageDimension>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(CudaImageFromImageFilter);

protected:
  CudaImageFromImageFilter() = default;
  ~CudaImageFromImageFilter() override = default;

  void
  GenerateData() override;
};

} // namespace itk
#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCudaImageFromImageFilter.hxx"
#endif

#endif // itkCudaImageFromImageFilter_h

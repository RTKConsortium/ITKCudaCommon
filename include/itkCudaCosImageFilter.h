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
#ifndef itkCudaCosImageFilter_h
#define itkCudaCosImageFilter_h

#include "itkCudaUnaryImageToImageFilter.h"
#include "itkCudaUnaryImageOperators.hcu"
#include "itkCosImageFilter.h"

namespace itk
{

/** \class CudaCosImageFilter
 *
 * \brief CUDA implementation of itk::CosImageFilter.
 *
 * Computes the cosine of each pixel using a device lambda. CPU execution
 * uses itk::CosImageFilter. GPU execution and the common buffer checks are
 * provided by CudaUnaryImageToImageFilter.
 *
 * \ingroup ITKCudaCommon
 */
template <class TImage>
class ITK_TEMPLATE_EXPORT CudaCosImageFilter
  : public CudaUnaryImageToImageFilter<TImage, TImage, CosImageFilter<TImage, TImage>>
{
public:
  /** Standard class type alias. */
  using Self = CudaCosImageFilter;
  using Superclass = CudaUnaryImageToImageFilter<TImage, TImage, CosImageFilter<TImage, TImage>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(CudaCosImageFilter);

  ITK_DISALLOW_COPY_AND_MOVE(CudaCosImageFilter);

protected:
  CudaCosImageFilter() = default;
  ~CudaCosImageFilter() override = default;

  using PixelType = typename Superclass::InputPixelType;

  void
  GenerateUnaryData(const PixelType * input, PixelType * output, itk::SizeValueType numberOfPixels) override
  {
    CudaCosImage(numberOfPixels, input, output);
  }
};

} // end namespace itk

#endif

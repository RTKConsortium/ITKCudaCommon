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
#ifndef itkCudaThresholdImageFilter_h
#define itkCudaThresholdImageFilter_h

#include "itkCudaUnaryImageToImageFilter.h"
#include "itkCudaUnaryImageOperators.h"
#include "itkThresholdImageFilter.h"

namespace itk
{

/** \class CudaThresholdImageFilter
 *
 * \brief CUDA implementation of itk::ThresholdImageFilter.
 *
 * Replaces each pixel outside the inherited lower and upper thresholds with
 * the inherited outside value using a device lambda. Both thresholds are
 * inclusive, as in itk::ThresholdImageFilter. CPU execution uses
 * itk::ThresholdImageFilter. GPU execution and the common buffer checks are
 * provided by CudaUnaryImageToImageFilter.
 *
 * \ingroup ITKCudaCommon
 */
template <class TImage>
class ITK_TEMPLATE_EXPORT CudaThresholdImageFilter
  : public CudaUnaryImageToImageFilter<TImage, TImage, ThresholdImageFilter<TImage>>
{
public:
  /** Standard class type alias. */
  using Self = CudaThresholdImageFilter;
  using Superclass = CudaUnaryImageToImageFilter<TImage, TImage, ThresholdImageFilter<TImage>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(CudaThresholdImageFilter);

  ITK_DISALLOW_COPY_AND_MOVE(CudaThresholdImageFilter);

protected:
  CudaThresholdImageFilter() = default;
  ~CudaThresholdImageFilter() override = default;

  using PixelType = typename Superclass::InputPixelType;

  void
  GenerateUnaryData(const PixelType * input, PixelType * output, itk::SizeValueType numberOfPixels) override
  {
    CudaThresholdImage(numberOfPixels, input, output, this->GetLower(), this->GetUpper(), this->GetOutsideValue());
  }
};

} // end namespace itk

#endif

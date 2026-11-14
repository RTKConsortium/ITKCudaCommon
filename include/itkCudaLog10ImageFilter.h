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
#ifndef itkCudaLog10ImageFilter_h
#define itkCudaLog10ImageFilter_h

#include "itkCudaUnaryImageToImageFilter.h"
#include "itkCudaUnaryImageOperators.hcu"
#include "itkLog10ImageFilter.h"

namespace itk
{

/** \class CudaLog10ImageFilter
 *
 * \brief CUDA implementation of itk::Log10ImageFilter.
 *
 * Computes the base-10 logarithm of each pixel using a device lambda. CPU execution
 * uses itk::Log10ImageFilter. GPU execution and the common buffer checks are
 * provided by CudaUnaryImageToImageFilter.
 *
 * \ingroup ITKCudaCommon
 */
template <class TImage>
class ITK_TEMPLATE_EXPORT CudaLog10ImageFilter
  : public CudaUnaryImageToImageFilter<TImage, TImage, Log10ImageFilter<TImage, TImage>>
{
public:
  /** Standard class type alias. */
  using Self = CudaLog10ImageFilter;
  using Superclass = CudaUnaryImageToImageFilter<TImage, TImage, Log10ImageFilter<TImage, TImage>>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(CudaLog10ImageFilter);

  ITK_DISALLOW_COPY_AND_MOVE(CudaLog10ImageFilter);

protected:
  CudaLog10ImageFilter() = default;
  ~CudaLog10ImageFilter() override = default;

  using PixelType = typename Superclass::InputPixelType;

  void
  GenerateUnaryData(const PixelType * input, PixelType * output, itk::SizeValueType numberOfPixels) override
  {
    CudaLog10Image(numberOfPixels, input, output);
  }
};

} // end namespace itk

#endif

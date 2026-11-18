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
#ifndef itkCudaBinaryImageOperators_h
#define itkCudaBinaryImageOperators_h

// Omit ITK DLL annotations when compiling CUDA, as in RTK's CUDA utilities.
#if defined(__CUDACC__) && !defined(ITK_STATIC)
#  define ITK_STATIC
#  include "itkMacro.h"
#  undef ITK_STATIC
#endif

#include "itkIntTypes.h"
#include "itkNumericTraits.h"

namespace itk
{

template <class TPixel>
void
CudaAddImage(itk::SizeValueType numberOfPixels,
             const TPixel *     input1,
             const TPixel *     input2,
             TPixel *           output,
             TPixel             constant1,
             TPixel             constant2);

template <class TPixel>
void
CudaSubtractImage(itk::SizeValueType numberOfPixels,
                  const TPixel *     input1,
                  const TPixel *     input2,
                  TPixel *           output,
                  TPixel             constant1,
                  TPixel             constant2);

template <class TPixel>
void
CudaMultiplyImage(itk::SizeValueType numberOfPixels,
                  const TPixel *     input1,
                  const TPixel *     input2,
                  TPixel *           output,
                  TPixel             constant1,
                  TPixel             constant2);

template <class TPixel>
void
CudaDivideImage(itk::SizeValueType numberOfPixels,
                const TPixel *     input1,
                const TPixel *     input2,
                TPixel *           output,
                TPixel             constant1,
                TPixel             constant2);

template <class TPixel>
void
CudaMaximumImage(itk::SizeValueType numberOfPixels,
                 const TPixel *     input1,
                 const TPixel *     input2,
                 TPixel *           output,
                 TPixel             constant1,
                 TPixel             constant2);

template <class TPixel>
void
CudaMinimumImage(itk::SizeValueType numberOfPixels,
                 const TPixel *     input1,
                 const TPixel *     input2,
                 TPixel *           output,
                 TPixel             constant1,
                 TPixel             constant2);

template <class TPixel>
void
CudaPowImage(itk::SizeValueType numberOfPixels,
             const TPixel *     input1,
             const TPixel *     input2,
             TPixel *           output,
             TPixel             constant1,
             TPixel             constant2);

template <class TPixel>
void
CudaAtan2Image(itk::SizeValueType numberOfPixels,
               const TPixel *     input1,
               const TPixel *     input2,
               TPixel *           output,
               TPixel             constant1,
               TPixel             constant2);

template <class TPixel>
void
CudaDivideOrZeroOutImage(itk::SizeValueType numberOfPixels,
                         const TPixel *     input1,
                         const TPixel *     input2,
                         TPixel *           output,
                         TPixel             constant1,
                         TPixel             constant2,
                         TPixel             threshold,
                         TPixel             outsideValue);

template <class TPixel>
void
CudaWeightedAddImage(itk::SizeValueType                       numberOfPixels,
                     const TPixel *                           input1,
                     const TPixel *                           input2,
                     TPixel *                                 output,
                     TPixel                                   constant1,
                     TPixel                                   constant2,
                     typename NumericTraits<TPixel>::RealType alpha,
                     typename NumericTraits<TPixel>::RealType beta);

template <class TPixel>
void
CudaModulusImage(itk::SizeValueType numberOfPixels,
                 const TPixel *     input1,
                 const TPixel *     input2,
                 TPixel *           output,
                 TPixel             constant1,
                 TPixel             constant2);

template <class TPixel>
void
CudaAndImage(itk::SizeValueType numberOfPixels,
             const TPixel *     input1,
             const TPixel *     input2,
             TPixel *           output,
             TPixel             constant1,
             TPixel             constant2);

template <class TPixel>
void
CudaOrImage(itk::SizeValueType numberOfPixels,
            const TPixel *     input1,
            const TPixel *     input2,
            TPixel *           output,
            TPixel             constant1,
            TPixel             constant2);

template <class TPixel>
void
CudaXorImage(itk::SizeValueType numberOfPixels,
             const TPixel *     input1,
             const TPixel *     input2,
             TPixel *           output,
             TPixel             constant1,
             TPixel             constant2);

template <class TPixel>
void
CudaBinaryMagnitudeImage(itk::SizeValueType numberOfPixels,
                         const TPixel *     input1,
                         const TPixel *     input2,
                         TPixel *           output,
                         TPixel             constant1,
                         TPixel             constant2);

template <class TPixel>
void
CudaConstrainedValueAdditionImage(itk::SizeValueType numberOfPixels,
                                  const TPixel *     input1,
                                  const TPixel *     input2,
                                  TPixel *           output,
                                  TPixel             constant1,
                                  TPixel             constant2);

template <class TPixel>
void
CudaConstrainedValueDifferenceImage(itk::SizeValueType numberOfPixels,
                                    const TPixel *     input1,
                                    const TPixel *     input2,
                                    TPixel *           output,
                                    TPixel             constant1,
                                    TPixel             constant2);

template <class TPixel, class TMaskPixel>
void
CudaMaskImage(itk::SizeValueType numberOfPixels,
              const TPixel *     input,
              const TMaskPixel * mask,
              TPixel *           output,
              TPixel             constant1,
              TMaskPixel         constant2,
              TMaskPixel         maskingValue,
              TPixel             outsideValue);

template <class TPixel, class TMaskPixel>
void
CudaMaskNegatedImage(itk::SizeValueType numberOfPixels,
                     const TPixel *     input,
                     const TMaskPixel * mask,
                     TPixel *           output,
                     TPixel             constant1,
                     TMaskPixel         constant2,
                     TMaskPixel         maskingValue,
                     TPixel             outsideValue);

} // namespace itk

#include "itkCudaBinaryImageOperators.hcu"

#endif

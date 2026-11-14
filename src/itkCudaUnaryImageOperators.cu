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
// Omit ITK DLL annotations when compiling CUDA, as in RTK's CUDA utilities.
#ifndef ITK_STATIC
#  define ITK_STATIC
#  include "itkMacro.h"
#  undef ITK_STATIC
#endif

#include "itkCudaUnaryImageOperators.hcu"
#include "itkCudaImageOperators.hcu"

#include <cmath>
#include <type_traits>

namespace itk
{

template <class PixelType>
void
CudaSquareImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto square = [] __device__(PixelType value) { return static_cast<PixelType>(value * value); };
  LaunchCudaImageOperation(numberOfPixels, square, output, input);
}
CudaCommon_INSTANTIATE_OPERATOR(CudaSquareImage);

template <class PixelType>
void
CudaSqrtImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto sqrtOperation = [] __device__(PixelType value) {
    return static_cast<PixelType>(sqrt(static_cast<double>(value)));
  };
  LaunchCudaImageOperation(numberOfPixels, sqrtOperation, output, input);
}
CudaCommon_INSTANTIATE_OPERATOR(CudaSqrtImage);

template <class PixelType>
void
CudaAbsImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto absOperation = [] __device__(PixelType value) {
    if constexpr (std::is_floating_point_v<PixelType>)
    {
      return fabs(value);
    }
    else if constexpr (std::is_signed_v<PixelType>)
    {
      return static_cast<PixelType>(value < PixelType{} ? -value : value);
    }
    else
    {
      return value;
    }
  };
  LaunchCudaImageOperation(numberOfPixels, absOperation, output, input);
}
CudaCommon_INSTANTIATE_OPERATOR(CudaAbsImage);

template <class PixelType>
void
CudaAcosImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto acosOperation = [] __device__(PixelType value) {
    return static_cast<PixelType>(acos(static_cast<double>(value)));
  };
  LaunchCudaImageOperation(numberOfPixels, acosOperation, output, input);
}
CudaCommon_INSTANTIATE_OPERATOR(CudaAcosImage);

template <class PixelType>
void
CudaAsinImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto asinOperation = [] __device__(PixelType value) {
    return static_cast<PixelType>(asin(static_cast<double>(value)));
  };
  LaunchCudaImageOperation(numberOfPixels, asinOperation, output, input);
}
CudaCommon_INSTANTIATE_OPERATOR(CudaAsinImage);

template <class PixelType>
void
CudaAtanImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto atanOperation = [] __device__(PixelType value) {
    return static_cast<PixelType>(atan(static_cast<double>(value)));
  };
  LaunchCudaImageOperation(numberOfPixels, atanOperation, output, input);
}
CudaCommon_INSTANTIATE_REAL_OPERATOR(CudaAtanImage);

template <class PixelType>
void
CudaBoundedReciprocalImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto boundedReciprocalOperation = [] __device__(PixelType value) {
    return static_cast<PixelType>(1.0 / (1.0 + static_cast<double>(value)));
  };
  LaunchCudaImageOperation(numberOfPixels, boundedReciprocalOperation, output, input);
}
CudaCommon_INSTANTIATE_OPERATOR(CudaBoundedReciprocalImage);

template <class PixelType>
void
CudaCosImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto cosOperation = [] __device__(PixelType value) {
    return static_cast<PixelType>(cos(static_cast<double>(value)));
  };
  LaunchCudaImageOperation(numberOfPixels, cosOperation, output, input);
}
CudaCommon_INSTANTIATE_REAL_OPERATOR(CudaCosImage);

template <class PixelType>
void
CudaExpImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto expOperation = [] __device__(PixelType value) {
    return static_cast<PixelType>(exp(static_cast<double>(value)));
  };
  LaunchCudaImageOperation(numberOfPixels, expOperation, output, input);
}
CudaCommon_INSTANTIATE_OPERATOR(CudaExpImage);

template <class PixelType>
void
CudaLogImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto logOperation = [] __device__(PixelType value) {
    return static_cast<PixelType>(log(static_cast<double>(value)));
  };
  LaunchCudaImageOperation(numberOfPixels, logOperation, output, input);
}
CudaCommon_INSTANTIATE_OPERATOR(CudaLogImage);

template <class PixelType>
void
CudaLog10Image(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto log10Operation = [] __device__(PixelType value) {
    return static_cast<PixelType>(log10(static_cast<double>(value)));
  };
  LaunchCudaImageOperation(numberOfPixels, log10Operation, output, input);
}
CudaCommon_INSTANTIATE_OPERATOR(CudaLog10Image);

template <class PixelType>
void
CudaRoundImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  // Match ITK's half-up rounding without changing large integral values.
  const auto roundOperation = [] __device__(PixelType value) {
    PixelType integralLimit = 0x1p52;
    if constexpr (std::is_same_v<PixelType, float>)
    {
      integralLimit = 0x1p23f;
    }
    if (fabs(value) >= integralLimit)
    {
      return value;
    }
    return static_cast<PixelType>(floor(value + PixelType{ 0.5 }));
  };
  LaunchCudaImageOperation(numberOfPixels, roundOperation, output, input);
}
CudaCommon_INSTANTIATE_REAL_OPERATOR(CudaRoundImage);

template <class PixelType>
void
CudaSinImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto sinOperation = [] __device__(PixelType value) {
    return static_cast<PixelType>(sin(static_cast<double>(value)));
  };
  LaunchCudaImageOperation(numberOfPixels, sinOperation, output, input);
}
CudaCommon_INSTANTIATE_REAL_OPERATOR(CudaSinImage);

template <class PixelType>
void
CudaTanImage(itk::SizeValueType numberOfPixels, const PixelType * input, PixelType * output)
{
  const auto tanOperation = [] __device__(PixelType value) {
    return static_cast<PixelType>(tan(static_cast<double>(value)));
  };
  LaunchCudaImageOperation(numberOfPixels, tanOperation, output, input);
}
CudaCommon_INSTANTIATE_REAL_OPERATOR(CudaTanImage);

} // namespace itk

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
#include "itkCudaBinaryImageOperators.h"
#include "CudaCommonExport.h"

namespace itk
{

// Modulus and bitwise operators require integer pixel types.
#define CudaCommon_INSTANTIATE_INTEGER_OPERATOR(OP)                               \
  template decltype(OP<signed char>) CudaCommon_EXPORT        OP<signed char>;    \
  template decltype(OP<unsigned char>) CudaCommon_EXPORT      OP<unsigned char>;  \
  template decltype(OP<short>) CudaCommon_EXPORT              OP<short>;          \
  template decltype(OP<unsigned short>) CudaCommon_EXPORT     OP<unsigned short>; \
  template decltype(OP<int>) CudaCommon_EXPORT                OP<int>;            \
  template decltype(OP<unsigned int>) CudaCommon_EXPORT       OP<unsigned int>;   \
  template decltype(OP<long>) CudaCommon_EXPORT               OP<long>;           \
  template decltype(OP<unsigned long>) CudaCommon_EXPORT      OP<unsigned long>;  \
  template decltype(OP<long long>) CudaCommon_EXPORT          OP<long long>;      \
  template decltype(OP<unsigned long long>) CudaCommon_EXPORT OP<unsigned long long>;

// Other operators are instantiated for every ITK scalar pixel type.
#define CudaCommon_INSTANTIATE_OPERATOR(OP)                   \
  template decltype(OP<float>) CudaCommon_EXPORT  OP<float>;  \
  template decltype(OP<double>) CudaCommon_EXPORT OP<double>; \
  CudaCommon_INSTANTIATE_INTEGER_OPERATOR(OP)

CudaCommon_INSTANTIATE_OPERATOR(CudaAddImage);
CudaCommon_INSTANTIATE_OPERATOR(CudaSubtractImage);
CudaCommon_INSTANTIATE_OPERATOR(CudaMultiplyImage);
CudaCommon_INSTANTIATE_OPERATOR(CudaDivideImage);
CudaCommon_INSTANTIATE_OPERATOR(CudaMaximumImage);
CudaCommon_INSTANTIATE_OPERATOR(CudaMinimumImage);
CudaCommon_INSTANTIATE_OPERATOR(CudaPowImage);
CudaCommon_INSTANTIATE_OPERATOR(CudaAtan2Image);
CudaCommon_INSTANTIATE_OPERATOR(CudaDivideOrZeroOutImage);
CudaCommon_INSTANTIATE_OPERATOR(CudaWeightedAddImage);
CudaCommon_INSTANTIATE_OPERATOR(CudaBinaryMagnitudeImage);
CudaCommon_INSTANTIATE_OPERATOR(CudaConstrainedValueAdditionImage);
CudaCommon_INSTANTIATE_OPERATOR(CudaConstrainedValueDifferenceImage);
CudaCommon_INSTANTIATE_INTEGER_OPERATOR(CudaModulusImage);
CudaCommon_INSTANTIATE_INTEGER_OPERATOR(CudaAndImage);
CudaCommon_INSTANTIATE_INTEGER_OPERATOR(CudaOrImage);
CudaCommon_INSTANTIATE_INTEGER_OPERATOR(CudaXorImage);

// Masks are limited to the integer types wrapped by default in ITK, plus the
// image pixel type, to bound the number of kernels. Wrapping uses the same
// types (CudaCommon_WRAP_MASK_TYPES in wrapping/CMakeLists.txt).
#define CudaCommon_INSTANTIATE_MASK(PIXEL, MASK)                                                     \
  template decltype(CudaMaskImage<PIXEL, MASK>) CudaCommon_EXPORT        CudaMaskImage<PIXEL, MASK>; \
  template decltype(CudaMaskNegatedImage<PIXEL, MASK>) CudaCommon_EXPORT CudaMaskNegatedImage<PIXEL, MASK>;

#define CudaCommon_INSTANTIATE_INTEGER_MASKS(PIXEL)                                                    \
  CudaCommon_INSTANTIATE_MASK(PIXEL, unsigned char) CudaCommon_INSTANTIATE_MASK(PIXEL, unsigned short) \
    CudaCommon_INSTANTIATE_MASK(PIXEL, short)

// For pixel types that are not already an instantiated mask type.
#define CudaCommon_INSTANTIATE_MASKS(PIXEL) \
  CudaCommon_INSTANTIATE_INTEGER_MASKS(PIXEL) CudaCommon_INSTANTIATE_MASK(PIXEL, PIXEL)

CudaCommon_INSTANTIATE_INTEGER_MASKS(unsigned char);
CudaCommon_INSTANTIATE_INTEGER_MASKS(unsigned short);
CudaCommon_INSTANTIATE_INTEGER_MASKS(short);
CudaCommon_INSTANTIATE_MASKS(signed char);
CudaCommon_INSTANTIATE_MASKS(int);
CudaCommon_INSTANTIATE_MASKS(unsigned int);
CudaCommon_INSTANTIATE_MASKS(long);
CudaCommon_INSTANTIATE_MASKS(unsigned long);
CudaCommon_INSTANTIATE_MASKS(long long);
CudaCommon_INSTANTIATE_MASKS(unsigned long long);
CudaCommon_INSTANTIATE_MASKS(float);
CudaCommon_INSTANTIATE_MASKS(double);

#undef CudaCommon_INSTANTIATE_MASKS
#undef CudaCommon_INSTANTIATE_INTEGER_MASKS
#undef CudaCommon_INSTANTIATE_MASK
#undef CudaCommon_INSTANTIATE_OPERATOR
#undef CudaCommon_INSTANTIATE_INTEGER_OPERATOR

} // namespace itk

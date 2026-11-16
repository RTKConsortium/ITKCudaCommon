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
#include "itkCudaUnaryImageOperators.h"
#include "CudaCommonExport.h"

namespace itk
{

// Every unary operator provided by the library. Adding an operator here is all
// that is needed to instantiate it for the scalar pixel types below.
#define CudaCommon_UNARY_OPERATORS(M) M(CudaSquareImage)

// Instantiate an operator for every ITK scalar pixel type.
#define CudaCommon_INSTANTIATE_OPERATOR(OP)                                           \
  template decltype(OP<signed char>) CudaCommon_EXPORT        OP<signed char>;        \
  template decltype(OP<unsigned char>) CudaCommon_EXPORT      OP<unsigned char>;      \
  template decltype(OP<short>) CudaCommon_EXPORT              OP<short>;              \
  template decltype(OP<unsigned short>) CudaCommon_EXPORT     OP<unsigned short>;     \
  template decltype(OP<int>) CudaCommon_EXPORT                OP<int>;                \
  template decltype(OP<unsigned int>) CudaCommon_EXPORT       OP<unsigned int>;       \
  template decltype(OP<long>) CudaCommon_EXPORT               OP<long>;               \
  template decltype(OP<unsigned long>) CudaCommon_EXPORT      OP<unsigned long>;      \
  template decltype(OP<long long>) CudaCommon_EXPORT          OP<long long>;          \
  template decltype(OP<unsigned long long>) CudaCommon_EXPORT OP<unsigned long long>; \
  template decltype(OP<float>) CudaCommon_EXPORT              OP<float>;              \
  template decltype(OP<double>) CudaCommon_EXPORT             OP<double>;

CudaCommon_UNARY_OPERATORS(CudaCommon_INSTANTIATE_OPERATOR);

#undef CudaCommon_INSTANTIATE_OPERATOR
#undef CudaCommon_UNARY_OPERATORS

} // namespace itk

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
#ifndef itkCudaSquareImageFilter_h
#define itkCudaSquareImageFilter_h

#include "itkCudaInPlaceImageFilter.h"

namespace itk
{

/** \class CudaSquareImageFilter
 *
 * \brief class to demonstrate the use of Cuda filters.
 *
 * CudaSquareImageFilter is the Cuda version of ImageToImageFilter that
 * computes the squared value of each pixel.
 *
 * \ingroup ITKCudaCommon
 */
template <class TImage>
class ITK_TEMPLATE_EXPORT CudaSquareImageFilter : public CudaImageToImageFilter<TImage, TImage>
{
public:
  /** Standard class type alias. */
  using Self = CudaSquareImageFilter;
  using Superclass = CudaImageToImageFilter<TImage, TImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
#ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(CudaSquareImageFilter);
#else
  itkTypeMacro(CudaSquareImageFilter, CudaImageToImageFilter);
#endif

  ITK_DISALLOW_COPY_AND_MOVE(CudaSquareImageFilter);

  /** ImageDimension constants */
  static constexpr unsigned int ImageDimension = TImage::ImageDimension;

#ifdef ITK_USE_CONCEPT_CHECKING
  itkConceptMacro(ImageDimensionCheck, (itk::Concept::SameDimensionOrMinusOneOrTwo<3, TImage::ImageDimension>));
#endif

protected:
  CudaSquareImageFilter() = default;
  ~CudaSquareImageFilter() = default;

  virtual void GPUGenerateData();
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCudaSquareImageFilter.hxx"
#endif

#endif

/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkCudaImageToImageFilter_hxx
#define itkCudaImageToImageFilter_hxx

#include "itkCudaImageToImageFilter.h"

namespace itk
{

template <class TInputImage, class TOutputImage, class TParentImageFilter>
CudaImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::CudaImageToImageFilter()
  : m_GPUEnabled(true)
{}

template <class TInputImage, class TOutputImage, class TParentImageFilter>
CudaImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::~CudaImageToImageFilter()
{}

template <class TInputImage, class TOutputImage, class TParentImageFilter>
void
CudaImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "GPU: " << (m_GPUEnabled ? "Enabled" : "Disabled") << std::endl;
}

template <class TInputImage, class TOutputImage, class TParentImageFilter>
void
CudaImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GenerateData()
{
  if (!m_GPUEnabled) // call CPU update function
  {
    Superclass::GenerateData();
  }
  else // call Cuda update function
  {
    // Call a method to allocate memory for the filter's outputs
    this->AllocateOutputs();

    GPUGenerateData();
  }
}

template <class TInputImage, class TOutputImage, class TParentImageFilter>
void
CudaImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GraftOutput(
  typename itk::CudaTraits<TOutputImage>::Type * output)
{
  using CudaOutputImage = typename itk::CudaTraits<TOutputImage>::Type;
  typename CudaOutputImage::Pointer cudaImage = dynamic_cast<CudaOutputImage *>(this->GetOutput());

  cudaImage->Graft(output);
}

template <class TInputImage, class TOutputImage, class TParentImageFilter>
void
CudaImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GraftOutput(DataObject * output)
{
  using CudaOutputImage = typename itk::CudaTraits<TOutputImage>::Type;
  CudaOutputImage * cudaImage = dynamic_cast<CudaOutputImage *>(output);
  if (cudaImage)
  {
    this->GraftOutput(cudaImage);
  }
  else
  {
    itkExceptionMacro(<< "itk::CudaImageToImageFilter::GraftOutput() cannot cast " << typeid(output).name() << " to "
                      << typeid(CudaOutputImage *).name());
  }
}

template <class TInputImage, class TOutputImage, class TParentImageFilter>
void
CudaImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GraftOutput(
  const DataObjectIdentifierType &               key,
  typename itk::CudaTraits<TOutputImage>::Type * output)
{
  using CudaOutputImage = typename itk::CudaTraits<TOutputImage>::Type;
  typename CudaOutputImage::Pointer cudaImage = dynamic_cast<CudaOutputImage *>(this->ProcessObject::GetOutput(key));

  cudaImage->Graft(output);
}

template <class TInputImage, class TOutputImage, class TParentImageFilter>
void
CudaImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GraftOutput(const DataObjectIdentifierType & key,
                                                                                   DataObject * output)
{
  using CudaOutputImage = typename itk::CudaTraits<TOutputImage>::Type;
  CudaOutputImage * cudaImage = dynamic_cast<CudaOutputImage *>(output);
  if (cudaImage)
  {
    this->GraftOutput(key, cudaImage);
  }
  else
  {
    itkExceptionMacro(<< "itk::CudaImageToImageFilter::GraftOutput() cannot cast " << typeid(output).name() << " to "
                      << typeid(CudaOutputImage *).name());
  }
}

} // end namespace itk

#endif

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
#ifndef itkCudaInPlaceImageFilter_hxx
#define itkCudaInPlaceImageFilter_hxx

#include "itkCudaInPlaceImageFilter.h"

namespace itk
{
/**
 *
 */
template <class TInputImage, class TOutputImage, class TParentImageFilter>
CudaInPlaceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::CudaInPlaceImageFilter()
{}

/**
 *
 */
template <class TInputImage, class TOutputImage, class TParentImageFilter>
CudaInPlaceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::~CudaInPlaceImageFilter()
{}

template <class TInputImage, class TOutputImage, class TParentImageFilter>
void
CudaInPlaceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::PrintSelf(std::ostream & os, Indent indent) const
{
  GPUSuperclass::PrintSelf(os, indent);
}

template <class TInputImage, class TOutputImage, class TParentImageFilter>
void
CudaInPlaceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::ReleaseInputs()
{
  CPUSuperclass::ReleaseInputs();

  /*
    if (this->GetGPUEnabled())
    {
      // do something
      std::cout << "ToDo: CudaInPlaceImageFilter::ReleaseInputs()" << std::endl;
    }
    else
    {
      CPUSuperclass::ReleaseInputs();
    }
  */
}

template <class TInputImage, class TOutputImage, class TParentImageFilter>
void
CudaInPlaceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::AllocateOutputs()
{
  CPUSuperclass::AllocateOutputs();
  /*
    if (this->GetGPUEnabled())
      {
      // if told to run in place and the types support it,
      if (this->GetInPlace() && this->CanRunInPlace())
        {
        // Graft this first input to the output.  Later, we'll need to
        // remove the input's hold on the bulk data.
        //
        OutputImagePointer inputAsOutput =
          dynamic_cast< TOutputImage * >(const_cast< TInputImage * >(this->GetInput()));
        if (inputAsOutput)
          {
          this->GraftOutput(inputAsOutput);
          }
        else
          {
          // if we cannot cast the input to an output type, then allocate
          // an output usual.
          OutputImagePointer outputPtr;

          outputPtr = this->GetOutput(0);
          outputPtr->SetBufferedRegion(outputPtr->GetRequestedRegion());
          outputPtr->Allocate();
          }

        using ImageBaseType = ImageBase< OutputImageDimension >;
        typename ImageBaseType::Pointer outputPtr;

        // If there are more than one outputs, allocate the remaining outputs
        for (unsigned int i = 1; i < this->GetNumberOfOutputs(); i++)
          {
          // Check whether the output is an image of the appropriate
          // dimension (use ProcessObject's version of the GetInput()
          // method since it returns the input as a pointer to a
          // DataObject as opposed to the subclass version which
          // static_casts the input to an TInputImage).
          outputPtr = dynamic_cast< ImageBaseType * >(this->ProcessObject::GetOutput(i));

          if (outputPtr)
            {
            outputPtr->SetBufferedRegion(outputPtr->GetRequestedRegion());
            outputPtr->Allocate();
            }
          // if the output is not of similar type then it is assumed the
          // the derived class allocated the output if needed.
          }

        }
      else
        {
        CPUSuperclass::AllocateOutputs();
        }
      }
    else
      {
      CPUSuperclass::AllocateOutputs();
      }
      */
}

} // end of namespace itk

#endif

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
#ifndef itkCudaUnaryImageToImageFilter_hxx
#define itkCudaUnaryImageToImageFilter_hxx

namespace itk
{

template <class TInputImage, class TOutputImage, class TParentImageFilter>
void
CudaUnaryImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GPUGenerateData()
{

  const auto * input = this->GetInput();
  auto *       output = this->GetOutput();
  const auto & region = output->GetRequestedRegion();

  const auto numberOfPixels = region.GetNumberOfPixels();
  if (numberOfPixels == 0)
  {
    return;
  }

  if (input->GetBufferedRegion() != region || output->GetBufferedRegion() != region)
  {
    itkExceptionMacro(
      "CUDA unary filter requires each buffered region to match the output requested region (index and size); "
      "stream pieces with equally sized buffers");
  }

  this->GenerateUnaryData(static_cast<const InputPixelType *>(input->GetCudaDataManager()->GetGPUBufferPointer()),
                          static_cast<OutputPixelType *>(output->GetCudaDataManager()->GetGPUBufferPointer()),
                          numberOfPixels);
}

} // end namespace itk

#endif // end conditional definition of the class

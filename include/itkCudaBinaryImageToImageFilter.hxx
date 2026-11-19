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
#ifndef itkCudaBinaryImageToImageFilter_hxx
#define itkCudaBinaryImageToImageFilter_hxx

namespace itk
{

template <class TImage, class TInputImage2, class TParentImageFilter>
void
CudaBinaryImageToImageFilter<TImage, TInputImage2, TParentImageFilter>::GPUGenerateData()
{
  const auto * input1 = dynamic_cast<const TImage *>(this->ProcessObject::GetInput(0));
  const auto * input2 = dynamic_cast<const TInputImage2 *>(this->ProcessObject::GetInput(1));
  if (!input1 && !input2)
  {
    itkExceptionMacro("CUDA binary filters require at least one image input");
  }

  auto *       output = this->GetOutput();
  const auto & region = output->GetRequestedRegion();
  const auto   numberOfPixels = region.GetNumberOfPixels();
  if (numberOfPixels == 0)
  {
    return;
  }

  if ((input1 && input1->GetBufferedRegion() != region) || (input2 && input2->GetBufferedRegion() != region) ||
      output->GetBufferedRegion() != region)
  {
    itkExceptionMacro(
      "CUDA binary filter requires each buffered region to match the output requested region (index and size); "
      "stream pieces with equally sized buffers");
  }

  this->GenerateBinaryData(
    static_cast<const PixelType *>(input1 ? input1->GetCudaDataManager()->GetConstGPUBufferPointer() : nullptr),
    static_cast<const Input2PixelType *>(input2 ? input2->GetCudaDataManager()->GetConstGPUBufferPointer() : nullptr),
    input1 ? PixelType{} : this->GetConstant1(),
    input2 ? Input2PixelType{} : this->GetConstant2(),
    static_cast<PixelType *>(output->GetCudaDataManager()->GetGPUBufferPointer()),
    numberOfPixels);
}

} // namespace itk

#endif

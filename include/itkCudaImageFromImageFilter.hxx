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
#ifndef itkCudaImageFromImageFilter_hxx
#define itkCudaImageFromImageFilter_hxx

namespace itk
{

template <typename TInputImage>
void CudaImageFromImageFilter<TInputImage>::GenerateData()
{
  const auto * in = this->GetInput();
  auto *       out = this->GetOutput();
  out->SetBufferedRegion(in->GetBufferedRegion());
  out->SetPixelContainer(const_cast<typename TInputImage::PixelContainer *>(in->GetPixelContainer()));
}

} // end namespace itk

#endif // itkCudaImageFromImageFilter_hxx

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
#ifndef itkCudaImageDataManager_hxx
#define itkCudaImageDataManager_hxx

#include "itkCudaUtil.h"
// #define VERBOSE

namespace itk
{

template <class ImageType>
void
CudaImageDataManager<ImageType>::SetImagePointer(ImageType * img)
{
  {
    const std::lock_guard<std::mutex> lock(m_Mutex);
    // Unlike the superclass method, CudaImage::GetPixelContainer() marks the
    // GPU buffer dirty.
    m_PixelContainer = img->ImageType::Superclass::GetPixelContainer();
  }

  RegionType region = img->GetBufferedRegion();
  IndexType  index = region.GetIndex();
  SizeType   size = region.GetSize();

  for (unsigned int d = 0; d < ImageType::ImageDimension; d++)
  {
    m_BufferedRegionIndex[d] = index[d];
    m_BufferedRegionSize[d] = size[d];
  }

  m_GPUBufferedRegionIndex = CudaDataManager::New();
  m_GPUBufferedRegionIndex->SetBufferSize(sizeof(int) * ImageType::ImageDimension);
  m_GPUBufferedRegionIndex->SetCPUBufferPointer(&m_BufferedRegionIndex);
  m_GPUBufferedRegionIndex->SetGPUBufferDirty();

  m_GPUBufferedRegionSize = CudaDataManager::New();
  m_GPUBufferedRegionSize->SetBufferSize(sizeof(int) * ImageType::ImageDimension);
  m_GPUBufferedRegionSize->SetCPUBufferPointer(&m_BufferedRegionSize);
  m_GPUBufferedRegionSize->SetGPUBufferDirty();
}

template <class ImageType>
void
CudaImageDataManager<ImageType>::UpdateCPUBuffer()
{
  {
    const std::lock_guard<std::mutex> lock(m_Mutex);
    if (m_PixelContainer && !m_CPUBuffer)
    {
      // Allocate the pixel container shared by the images using this data
      // manager, as Image::Allocate() does.
      m_PixelContainer->Reserve(m_BufferSize / sizeof(typename ImageType::PixelContainer::Element));
      this->SetCPUBufferPointer(m_PixelContainer->GetBufferPointer());
    }
  }

  Superclass::UpdateCPUBuffer();
}

template <class ImageType>
void
CudaImageDataManager<ImageType>::Graft(const CudaDataManager * data)
{
  Superclass::Graft(data);
}

template <class ImageType>
void
CudaImageDataManager<ImageType>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "m_GPUBufferedRegionIndex: " << m_GPUBufferedRegionIndex << std::endl;
  os << indent << "m_GPUBufferedRegionSize: " << m_GPUBufferedRegionSize << std::endl;
}

} // namespace itk

#endif

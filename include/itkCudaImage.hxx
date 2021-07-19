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
#ifndef itkCudaImage_hxx
#define itkCudaImage_hxx

#include "itkCudaImage.h"

namespace itk
{
//
// Constructor
//
template <class TPixel, unsigned int VImageDimension>
CudaImage<TPixel, VImageDimension>::CudaImage()
{
  m_DataManager = CudaImageDataManager<CudaImage<TPixel, VImageDimension>>::New();
}

template <class TPixel, unsigned int VImageDimension>
CudaImage<TPixel, VImageDimension>::~CudaImage()
{}

template <class TPixel, unsigned int VImageDimension>
void
CudaImage<TPixel, VImageDimension>::Allocate(bool initializePixels)
{
  // allocate CPU memory - calling Allocate() in superclass
  Superclass::Allocate(initializePixels);

  // allocate Cuda memory
  this->ComputeOffsetTable();
  SizeValueType numPixel = this->GetOffsetTable()[VImageDimension];
  m_DataManager->SetBufferSize(sizeof(TPixel) * numPixel);
  m_DataManager->SetImagePointer(this);
  m_DataManager->SetCPUBufferPointer(Superclass::GetBufferPointer());

  // When we allocate both buffers are dirty and set so to avoid useless transfers
  // between GPU and CPU.
  m_DataManager->SetGPUDirtyFlag(true);
  m_DataManager->SetCPUDirtyFlag(true);

  // If initialize pixel is set then we set the CPU dirty flag to false
  if (initializePixels)
  {
    m_DataManager->SetCPUDirtyFlag(false);
  }
}

template <class TPixel, unsigned int VImageDimension>
void
CudaImage<TPixel, VImageDimension>::Initialize()
{
  // CPU image initialize
  Superclass::Initialize();

  m_DataManager = CudaImageDataManager<CudaImage<TPixel, VImageDimension>>::New();
}

template <class TPixel, unsigned int VImageDimension>
void
CudaImage<TPixel, VImageDimension>::SetBufferedRegion(const RegionType & region)
{
  // If the regions are the same we don't change
  const RegionType currentRegion = this->GetBufferedRegion();
  bool             sameRegions = true;
  for (unsigned int i = 0; i < VImageDimension; i++)
  {
    if (region.GetSize()[i] != currentRegion.GetSize()[i] || region.GetIndex()[i] != currentRegion.GetIndex()[i])
    {
      sameRegions = false;
      break;
    }
  }

  if (sameRegions)
  {
    return;
  }

  Superclass::SetBufferedRegion(region);
  SizeValueType numPixel = this->GetOffsetTable()[VImageDimension];
  m_DataManager->SetBufferSize(sizeof(TPixel) * numPixel);
  m_DataManager->SetCPUDirtyFlag(false); // prevent the GPU to copy to the CPU
  m_DataManager->SetGPUBufferDirty();
}


template <class TPixel, unsigned int VImageDimension>
void
CudaImage<TPixel, VImageDimension>::FillBuffer(const TPixel & value)
{
  m_DataManager->SetGPUBufferDirty();
  Superclass::FillBuffer(value);
}

template <class TPixel, unsigned int VImageDimension>
void
CudaImage<TPixel, VImageDimension>::SetPixel(const IndexType & index, const TPixel & value)
{
  m_DataManager->SetGPUBufferDirty();
  Superclass::SetPixel(index, value);
}

template <class TPixel, unsigned int VImageDimension>
const TPixel &
CudaImage<TPixel, VImageDimension>::GetPixel(const IndexType & index) const
{
  m_DataManager->UpdateCPUBuffer();
  return Superclass::GetPixel(index);
}

template <class TPixel, unsigned int VImageDimension>
TPixel &
CudaImage<TPixel, VImageDimension>::GetPixel(const IndexType & index)
{
  m_DataManager->UpdateCPUBuffer();
  return Superclass::GetPixel(index);
}

template <class TPixel, unsigned int VImageDimension>
TPixel & CudaImage<TPixel, VImageDimension>::operator[](const IndexType & index)
{
  m_DataManager->UpdateCPUBuffer();
  return Superclass::operator[](index);
}

template <class TPixel, unsigned int VImageDimension>
const TPixel & CudaImage<TPixel, VImageDimension>::operator[](const IndexType & index) const
{
  m_DataManager->UpdateCPUBuffer();
  return Superclass::operator[](index);
}

template <class TPixel, unsigned int VImageDimension>
void
CudaImage<TPixel, VImageDimension>::SetPixelContainer(PixelContainer * container)
{
  Superclass::SetPixelContainer(container);
  m_DataManager->SetImagePointer(this);
  m_DataManager->SetCPUBufferPointer(Superclass::GetBufferPointer());
  m_DataManager->SetCPUDirtyFlag(this->GetBufferPointer() == nullptr);
  m_DataManager->SetGPUDirtyFlag(true);
  SizeValueType numPixel = this->GetOffsetTable()[VImageDimension];
  m_DataManager->SetBufferSize(sizeof(TPixel) * numPixel);
}

template <class TPixel, unsigned int VImageDimension>
void
CudaImage<TPixel, VImageDimension>::UpdateBuffers()
{
  m_DataManager->UpdateCPUBuffer();
  m_DataManager->UpdateGPUBuffer();
}

template <class TPixel, unsigned int VImageDimension>
TPixel *
CudaImage<TPixel, VImageDimension>::GetBufferPointer()
{
  /* less conservative version - if you modify pixel value using
   * this pointer then you must set the image as modified manually!!! */
  m_DataManager->UpdateCPUBuffer();
  m_DataManager->SetGPUDirtyFlag(true);
  return Superclass::GetBufferPointer();
}

template <class TPixel, unsigned int VImageDimension>
const TPixel *
CudaImage<TPixel, VImageDimension>::GetBufferPointer() const
{
  // const does not change buffer, but if CPU is dirty then make it up-to-date
  m_DataManager->UpdateCPUBuffer();
  m_DataManager->SetGPUDirtyFlag(true); // THIS IS NEEDED BECAUSE NON-CONST ITERATORS USE THIS FUNCTION
  return Superclass::GetBufferPointer();
}

template <class TPixel, unsigned int VImageDimension>
CudaDataManager::Pointer
CudaImage<TPixel, VImageDimension>::GetCudaDataManager() const
{
  using CudaImageDataSuperclass = typename CudaImageDataManager<CudaImage>::Superclass;
  using CudaImageDataSuperclassPointer = typename CudaImageDataSuperclass::Pointer;

  return static_cast<CudaImageDataSuperclassPointer>(m_DataManager.GetPointer());
}

template <class TPixel, unsigned int VImageDimension>
void
CudaImage<TPixel, VImageDimension>::Graft(const Self * data)
{
  using CudaImageDataManagerType = CudaImageDataManager<CudaImage>;

  // call the superclass' implementation
  Superclass::Graft(dynamic_cast<const DataObject *>(data));
  m_DataManager = dynamic_cast<CudaImageDataManagerType *>(data->GetCudaDataManager().GetPointer());
  return;
}

template <class TPixel, unsigned int VImageDimension>
void
CudaImage<TPixel, VImageDimension>::Graft(const DataObject * data)
{
  if (data)
  {
    // Attempt to cast data to an Image
    const auto * const cuImgData = dynamic_cast<const Self *>(data);
    if (cuImgData != nullptr)
    {
      this->Graft(cuImgData);
    }
    else
    {
      // pointer could not be cast back down
      itkExceptionMacro(<< "itk::CudaImage::Graft() cannot cast " << typeid(data).name() << " to "
                        << typeid(const Self *).name());
    }
  }
}

} // namespace itk

#endif

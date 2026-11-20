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
#ifndef itkCudaImageDataManager_h
#define itkCudaImageDataManager_h

#include <itkObject.h>
#include <itkLightObject.h>
#include <itkObjectFactory.h>
#include "itkCudaUtil.h"
#include "itkCudaDataManager.h"

namespace itk
{
/**
 * \class CudaImageDataManager
 *
 * DataManager for CudaImage. This class will take care of data synchronization
 * between CPU Image and Cuda Image.
 *
 * \ingroup ITKCudaCommon
 */
template <class ImageType>
class ITK_TEMPLATE_EXPORT CudaImageDataManager : public CudaDataManager
{
public:
  using Self = CudaImageDataManager;
  using Superclass = CudaDataManager;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  using RegionType = typename ImageType::RegionType;
  using IndexType = typename ImageType::IndexType;
  using SizeType = typename ImageType::SizeType;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(CudaImageDataManager);

  itkGetModifiableObjectMacro(GPUBufferedRegionIndex, CudaDataManager);
  itkGetModifiableObjectMacro(GPUBufferedRegionSize, CudaDataManager);

  void
  SetImagePointer(ImageType * img);
  ImageType *
  GetImagePointer()
  {
    return this->m_Image;
  }

#ifndef ITK_FUTURE_LEGACY_REMOVE
  /** Copy the GPU buffer to the CPU buffer if the CPU buffer is dirty.
   * \deprecated Use UpdateCPUBuffer() instead. The time stamps of the image
   * and of the data manager are not compared anymore. */
#  if defined(ITK_LEGACY_REMOVE) && !defined(ITK_LEGACY_SILENT) && !defined(ITK_WRAPPING)
  [[deprecated("Use UpdateCPUBuffer() instead.")]]
#  endif
  virtual void
  MakeCPUBufferUpToDate()
  {
    this->UpdateCPUBuffer();
  }

  /** Copy the CPU buffer to the GPU buffer if the GPU buffer is dirty.
   * \deprecated Use UpdateGPUBuffer() instead. The time stamps of the image
   * and of the data manager are not compared anymore. */
#  if defined(ITK_LEGACY_REMOVE) && !defined(ITK_LEGACY_SILENT) && !defined(ITK_WRAPPING)
  [[deprecated("Use UpdateGPUBuffer() instead.")]]
#  endif
  virtual void
  MakeGPUBufferUpToDate()
  {
    this->UpdateGPUBuffer();
  }
#endif

  void
  UpdateCPUBuffer() override;

  /** Grafting Cuda Image Data */
  void
  Graft(const CudaDataManager * data) override;

protected:
  CudaImageDataManager() {}
  virtual ~CudaImageDataManager() {}

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  CudaImageDataManager(const Self &); // purposely not implemented
  void
  operator=(const Self &);

  ImageType *                       m_Image = nullptr;
  IndexType                         m_BufferedRegionIndex;
  SizeType                          m_BufferedRegionSize;
  typename CudaDataManager::Pointer m_GPUBufferedRegionIndex;
  typename CudaDataManager::Pointer m_GPUBufferedRegionSize;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCudaImageDataManager.hxx"
#endif

#endif

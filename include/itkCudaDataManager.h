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
#ifndef itkCudaDataManager_h
#define itkCudaDataManager_h

#include "itkObject.h"
#include "itkDataObject.h"
#include "itkObjectFactory.h"
#include "itkCudaUtil.h"
#include "CudaCommonExport.h"

#include <mutex>
#include <memory>

//#define VERBOSE

namespace itk
{
class CudaCommon_EXPORT GPUMemPointer : public Object
{
public:
  using Self = GPUMemPointer;
  using Superclass = Object;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
#ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(GPUMemPointer);
#else
  itkTypeMacro(GPUMemPointer, Object);
#endif

  void
  Allocate(size_t bufferSize)
  {
#ifdef VERBOSE
    if (m_GPUBuffer)
      std::cout << this << "::Freed GPU buffer of size " << m_BufferSize << " Bytes"
                << " : " << m_GPUBuffer << std::endl;
#endif
    m_BufferSize = bufferSize;
    CUDA_CHECK(cudaFree(m_GPUBuffer));
    CUDA_CHECK(cudaMalloc(&m_GPUBuffer, bufferSize));
#ifdef VERBOSE
    std::cout << this << "::Allocate Create GPU buffer of size " << bufferSize << " Bytes"
              << " : " << m_GPUBuffer << std::endl;
#endif
  }

  void
  Free()
  {
#ifdef VERBOSE
    if (m_GPUBuffer)
      std::cout << this << "::Freed GPU buffer of size " << m_BufferSize << " Bytes"
                << " : " << m_GPUBuffer << std::endl;
#endif
    CUDA_CHECK(cudaFree(m_GPUBuffer));
    m_GPUBuffer = nullptr;
    m_BufferSize = 0;
  }

  ~GPUMemPointer() override
  {
    if (m_GPUBuffer)
    {
      this->Free();
    }
  }

  void *
  GetPointer()
  {
    return m_GPUBuffer;
  }

  void *
  GetPointerPtr()
  {
    return &m_GPUBuffer;
  }

  size_t
  GetBufferSize() const
  {
    return m_BufferSize;
  }

protected:
  GPUMemPointer()
  {
    m_GPUBuffer = nullptr;
    m_BufferSize = 0;
  }

  void * m_GPUBuffer;
  size_t m_BufferSize;
};

/** \class CudaDataManager
 * \brief GPU memory manager implemented using Cuda. Required by CudaImage class.
 *
 * This class serves as a base class for Cuda data container for CudaImage class,
 * which is similar to ImageBase class for Image class. However, all the image-related
 * meta data will be already stored in image class (parent of CudaImage), therefore
 * we did not name it CudaImageBase. Rather, this class is a Cuda-specific data manager
 * that provides functionalities for RAM-GRAM data synchronization and grafting Cuda data.
 *
 * \ingroup ITKCudaCommon
 */
class CudaCommon_EXPORT CudaDataManager : public Object
{
public:
  using Self = CudaDataManager;
  using Superclass = Object;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;
  using ModifiedTimeType = unsigned long;

  itkNewMacro(Self);
#ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(CudaDataManager);
#else
  itkTypeMacro(CudaDataManager, Object);
#endif

  /** total buffer size in bytes */
  void
  SetBufferSize(size_t num);

  size_t
  GetBufferSize() const
  {
    return m_BufferSize;
  }

  void
  SetBufferFlag(int flags);

  void
  SetCPUBufferPointer(void * ptr);

  void
  SetCPUDirtyFlag(bool isDirty);

  void
  SetGPUDirtyFlag(bool isDirty);

  /** Make GPU up-to-date and mark CPU as dirty.
   * Call this function when you want to modify CPU data */
  void
  SetCPUBufferDirty();

  /** Make CPU up-to-date and mark GPU as dirty.
   * Call this function when you want to modify Cuda data */
  void
  SetGPUBufferDirty();

  bool
  IsCPUBufferDirty() const
  {
    return m_IsCPUBufferDirty;
  }

  bool
  IsGPUBufferDirty() const
  {
    return m_IsGPUBufferDirty;
  }

  /** actual Cuda->CPU memory copy takes place here */
  virtual void
  UpdateCPUBuffer();

  /** actual CPU->Cuda memory copy takes place here */
  virtual void
  UpdateGPUBuffer();

  void
  Allocate();
  void
  Free();

  /** Synchronize CPU and Cuda buffers (using dirty flags) */
  bool
  Update();

  /** Method for grafting the content of one CudaDataManager into another one */
  virtual void
  Graft(const CudaDataManager * data);

  /** Initialize CudaDataManager */
  virtual void
  Initialize();

  /** Get Cuda buffer pointer */
  void *
  GetGPUBufferPointer();

  /** Get CPU buffer pointer */
  void *
  GetCPUBufferPointer();

  /** Get Cuda buffer size without calling GetGPUBufferPointer, which
   * which would trigger an unwanted CPU -> GPU memory transfer */
  size_t
  GetGPUBufferSize()
  {
    return m_GPUBuffer->GetBufferSize();
  }

protected:
  CudaDataManager();
  ~CudaDataManager() override;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  int m_Device;

private:
  CudaDataManager(const Self &) = delete; // purposely not implemented
  void
  operator=(const Self &) = delete;

protected:
  size_t m_BufferSize; // # of bytes

  /** buffer type */
  int m_MemFlags;

  /** buffer pointers */
  GPUMemPointer::Pointer m_GPUBuffer;
  void *                 m_CPUBuffer;

  /** checks if buffer needs to be updated */
  bool m_IsGPUBufferDirty;
  bool m_IsCPUBufferDirty;

  /** whether gpu buffers from gpu memory should be released when dirty */
  bool m_ReleaseDirtyGPUBuffer;

  /** Mutex lock to prevent r/w hazard for multithreaded code */
  std::mutex m_Mutex;
};

} // namespace itk

#endif

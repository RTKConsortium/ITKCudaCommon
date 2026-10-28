%{
#include "itkCudaDataManager.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace
{
// Minimal subset of the legacy DLPack ABI. The layout must match
// DLManagedTensor from dlpack.h.
struct itk_cudacommon_dlpack_device
{
  int32_t device_type;
  int32_t device_id;
};

struct itk_cudacommon_dlpack_dtype
{
  uint8_t  code;
  uint8_t  bits;
  uint16_t lanes;
};

struct itk_cudacommon_dlpack_tensor
{
  void *                      data;
  itk_cudacommon_dlpack_device device;
  int32_t                     ndim;
  itk_cudacommon_dlpack_dtype  dtype;
  int64_t *                   shape;
  int64_t *                   strides;
  uint64_t                    byte_offset;
};

struct itk_cudacommon_dlpack_managed_tensor
{
  itk_cudacommon_dlpack_tensor tensor;
  void *                       manager_ctx;
  void (*deleter)(itk_cudacommon_dlpack_managed_tensor *);
};

struct itk_cudacommon_dlpack_view_registry
{
  std::mutex                                                      mutex;
  std::map<itk::CudaDataManager *, std::pair<unsigned int, bool>> views;
};

itk_cudacommon_dlpack_view_registry &
itk_cudacommon_dlpack_registry()
{
  static auto * registry = new itk_cudacommon_dlpack_view_registry;
  return *registry;
}

struct itk_cudacommon_cuda_device_guard
{
  int  previous_device = 0;
  bool restore = false;

  explicit itk_cudacommon_cuda_device_guard(int device)
  {
    if (cudaGetDevice(&previous_device) == cudaSuccess && cudaSetDevice(device) == cudaSuccess)
    {
      restore = true;
    }
  }

  ~itk_cudacommon_cuda_device_guard()
  {
    if (restore)
    {
      cudaSetDevice(previous_device);
    }
  }
};

struct itk_cudacommon_dlpack_manager
{
  itk_cudacommon_dlpack_managed_tensor managed_tensor;
  itk::CudaDataManager::Pointer       data_manager;
  std::vector<int64_t>                shape;
  std::vector<int64_t>                strides;
  bool                                registered = false;

  ~itk_cudacommon_dlpack_manager()
  {
    if (!registered || !data_manager)
    {
      return;
    }

    auto &                      registry = itk_cudacommon_dlpack_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto                        it = registry.views.find(data_manager.GetPointer());
    if (it == registry.views.end())
    {
      return;
    }
    if (it->second.first > 0 && --it->second.first == 0)
    {
      data_manager->SetReleaseDirtyGPUBuffer(it->second.second);
      registry.views.erase(it);
    }
  }
};

void
itk_cudacommon_dlpack_register_view(itk::CudaDataManager * data_manager)
{
  auto &                      registry = itk_cudacommon_dlpack_registry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  auto                        it = registry.views.find(data_manager);
  if (it == registry.views.end())
  {
    registry.views.emplace(data_manager, std::make_pair(1u, data_manager->GetReleaseDirtyGPUBuffer()));
    data_manager->SetReleaseDirtyGPUBuffer(false);
  }
  else
  {
    ++it->second.first;
  }
}

void
itk_cudacommon_dlpack_deleter(itk_cudacommon_dlpack_managed_tensor * tensor)
{
  delete static_cast<itk_cudacommon_dlpack_manager *>(tensor->manager_ctx);
}

void
itk_cudacommon_dlpack_capsule_destructor(PyObject * capsule)
{
  if (PyCapsule_IsValid(capsule, "used_dltensor"))
  {
    return;
  }

  auto * tensor =
    static_cast<itk_cudacommon_dlpack_managed_tensor *>(PyCapsule_GetPointer(capsule, "dltensor"));
  if (tensor == nullptr)
  {
    PyErr_WriteUnraisable(capsule);
    return;
  }
  tensor->deleter(tensor);
}
} // namespace
%}

%typemap(in) const std::vector<int64_t> & (std::vector<int64_t> values)
{
  PyObject * sequence = PySequence_Fast($input, "Expected a sequence of integers");
  if (sequence == nullptr)
  {
    SWIG_fail;
  }

  const Py_ssize_t size = PySequence_Size(sequence);
  if (size < 0)
  {
    Py_DECREF(sequence);
    SWIG_fail;
  }
  values.reserve(static_cast<size_t>(size));
  for (Py_ssize_t i = 0; i < size; ++i)
  {
    const long long value = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(sequence, i));
    if (value == -1 && PyErr_Occurred())
    {
      Py_DECREF(sequence);
      SWIG_fail;
    }
    values.push_back(static_cast<int64_t>(value));
  }
  Py_DECREF(sequence);
  $1 = &values;
}

%extend itkCudaDataManager
{
  PyObject *
  _create_dlpack_capsule(const std::vector<int64_t> & shape,
                         const std::vector<int64_t> & strides,
                         int                           dtype_code,
                         int                           dtype_bits,
                         int                           dtype_lanes,
                         PyObject *                    stream)
  {
    itk_cudacommon_cuda_device_guard device_guard(self->GetDevice());

    if (stream != Py_None)
    {
      const long long stream_value = PyLong_AsLongLong(stream);
      if (stream_value == -1 && PyErr_Occurred())
      {
        return nullptr;
      }
      if (stream_value != 1 && stream_value != -1)
      {
        cudaDeviceSynchronize();
      }
    }

    auto manager = std::make_unique<itk_cudacommon_dlpack_manager>();
    manager->data_manager = self;
    manager->shape = shape;
    manager->strides = strides;

    manager->managed_tensor.tensor.data = self->GetGPUBufferPointer();
    manager->managed_tensor.tensor.device = { 2, self->GetDevice() };

    itk_cudacommon_dlpack_register_view(self);
    manager->registered = true;

    manager->managed_tensor.tensor.ndim = static_cast<int32_t>(manager->shape.size());
    manager->managed_tensor.tensor.dtype = { static_cast<uint8_t>(dtype_code),
                                             static_cast<uint8_t>(dtype_bits),
                                             static_cast<uint16_t>(dtype_lanes) };
    manager->managed_tensor.tensor.shape = manager->shape.data();
    manager->managed_tensor.tensor.strides =
      manager->strides.empty() ? nullptr : manager->strides.data();
    manager->managed_tensor.tensor.byte_offset = 0;
    manager->managed_tensor.manager_ctx = manager.get();
    manager->managed_tensor.deleter = itk_cudacommon_dlpack_deleter;

    PyObject * capsule =
      PyCapsule_New(&manager->managed_tensor, "dltensor", itk_cudacommon_dlpack_capsule_destructor);
    if (capsule == nullptr)
    {
      return nullptr;
    }

    manager.release();
    return capsule;
  }
};

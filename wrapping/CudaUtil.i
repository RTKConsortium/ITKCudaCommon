%{
#include "itkCudaUtil.h"
%}

%rename(set_default_cuda_device) itk::SetDefaultCudaDevice;
%rename(get_default_cuda_device) itk::GetDefaultCudaDevice;

%inline %{
namespace itk {
void SetDefaultCudaDevice(int device);
int GetDefaultCudaDevice();
}
%}

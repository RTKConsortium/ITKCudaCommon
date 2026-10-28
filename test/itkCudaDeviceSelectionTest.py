import itk


def test_set_default_cuda_device_api():
    # Available at top-level itk.
    assert hasattr(itk, "set_default_cuda_device")
    assert hasattr(itk, "get_default_cuda_device")

    # Setting to an integer device index should succeed.
    itk.set_default_cuda_device(0)
    assert itk.get_default_cuda_device() == 0

    # Setting to -1 resets to automatic selection (max FLOPS device).
    itk.set_default_cuda_device(-1)
    assert itk.get_default_cuda_device() == 0

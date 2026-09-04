import itk
import os


def test_set_default_cuda_device_api():
    # Available at top-level itk.
    assert hasattr(itk, "set_default_cuda_device")

    # Setting to an integer device index should set the environment variable.
    itk.set_default_cuda_device(0)
    assert os.environ["ITK_CUDA_DEFAULT_DEVICE"] == "0"

    # Setting to -1 resets to automatic selection.
    itk.set_default_cuda_device(-1)
    assert os.environ["ITK_CUDA_DEFAULT_DEVICE"] == "-1"

    # Restore a clean state for other tests.
    os.environ.pop("ITK_CUDA_DEFAULT_DEVICE", None)

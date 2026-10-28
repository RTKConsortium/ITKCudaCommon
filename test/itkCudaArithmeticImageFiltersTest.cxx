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

#include "itkCudaAddImageFilter.h"
#include "itkCudaDivideImageFilter.h"
#include "itkCudaDivideOrZeroOutImageFilter.h"
#include "itkCudaImage.h"
#include "itkCudaMultiplyImageFilter.h"
#include "itkCudaSubtractImageFilter.h"
#include "itkCudaThresholdImageFilter.h"
#include "itkNumericTraits.h"
#include "itkCudaSquareImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageSource.h"
#include "itkStreamingImageFilter.h"

#include <array>
#include <type_traits>
#include <iostream>
#include <string>

namespace
{

template <typename TIndex>
float
ValueAtIndex(const TIndex & index)
{
  float value = 0;
  for (unsigned int d = 0; d < TIndex::Dimension; ++d)
  {
    value = value * 10 + index[d] + 4;
  }
  return value;
}

// Produce only the requested piece so the arithmetic chain can stream.
template <typename TImage>
class RegionImageSource : public itk::ImageSource<TImage>
{
public:
  using Self = RegionImageSource;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro(Self);

  unsigned int
  GetNumberOfGeneratedPieces() const
  {
    return m_NumberOfGeneratedPieces;
  }

protected:
  RegionImageSource() = default;

  void
  GenerateData() override
  {
    this->AllocateOutputs();
    auto *                                    output = this->GetOutput();
    itk::ImageRegionIteratorWithIndex<TImage> it(output, output->GetRequestedRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      it.Set(ValueAtIndex(it.GetIndex()));
    }
    ++m_NumberOfGeneratedPieces;
  }

private:
  unsigned int m_NumberOfGeneratedPieces{};
};

template <typename TImage, typename TPixel, std::size_t VNumberOfPixels>
typename TImage::Pointer
MakeImage(const std::array<TPixel, VNumberOfPixels> & pixels)
{
  auto image = TImage::New();
  image->SetRegions(itk::MakeSize(static_cast<itk::SizeValueType>(VNumberOfPixels), 1));
  image->Allocate();
  for (itk::SizeValueType i = 0; i < VNumberOfPixels; ++i)
  {
    image->SetPixel(itk::MakeIndex(static_cast<itk::IndexValueType>(i), 0),
                    static_cast<typename TImage::PixelType>(pixels[i]));
  }
  return image;
}

template <typename TImage, typename TExpected, std::size_t VNumberOfPixels>
bool
CheckPixels(const TImage * image, const std::array<TExpected, VNumberOfPixels> & expected, const char * filterName)
{
  for (itk::SizeValueType i = 0; i < VNumberOfPixels; ++i)
  {
    const auto actual = image->GetPixel(itk::MakeIndex(static_cast<itk::IndexValueType>(i), 0));
    if (actual != static_cast<typename TImage::PixelType>(expected[i]))
    {
      std::cerr << filterName << " pixel " << i << ": got " << actual << ", expected " << expected[i] << std::endl;
      return false;
    }
  }
  return true;
}

template <unsigned int VDimension>
bool
CheckRequestedRegions()
{
  using Image = itk::CudaImage<float, VDimension>;
  typename Image::RegionType full;
  typename Image::RegionType crop;
  typename Image::IndexType  start;
  typename Image::SizeType   size;
  start.Fill(-3);
  size.Fill(6);
  full.SetIndex(start);
  full.SetSize(size);
  start.Fill(-1);
  size.Fill(2);
  crop.SetIndex(start);
  crop.SetSize(size);

  auto makeInput = [&](const typename Image::RegionType & bufferedRegion) {
    auto image = Image::New();
    image->SetRegions(full);
    image->SetBufferedRegion(bufferedRegion);
    image->Allocate();
    itk::ImageRegionIteratorWithIndex<Image> it(image, image->GetBufferedRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      it.Set(ValueAtIndex(it.GetIndex()));
    }
    return image;
  };
  auto check = [&](auto * filter, auto expected) {
    filter->UpdateOutputInformation();
    filter->GetOutput()->SetRequestedRegion(crop);
    filter->Update();
    if (filter->GetOutput()->GetRequestedRegion() != crop || filter->GetOutput()->GetBufferedRegion() != crop)
    {
      std::cerr << "Filter enlarged a partial request" << std::endl;
      return false;
    }
    itk::ImageRegionConstIteratorWithIndex<Image> it(filter->GetOutput(), crop);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      if (it.Get() != expected(ValueAtIndex(it.GetIndex())))
      {
        std::cerr << filter->GetNameOfClass() << " cropped pixel mismatch at " << it.GetIndex() << std::endl;
        return false;
      }
    }
    return true;
  };

  for (bool inPlace : { false, true })
  {
    auto add = itk::CudaAddImageFilter<Image>::New();
    add->SetInPlace(inPlace);
    add->SetInput1(makeInput(crop));
    add->SetInput2(makeInput(crop));
    if (!check(add.GetPointer(), [](float v) { return v + v; }))
    {
      return false;
    }
    for (bool constantFirst : { false, true })
    {
      auto subtract = itk::CudaSubtractImageFilter<Image>::New();
      subtract->SetInPlace(inPlace);
      if (constantFirst)
      {
        subtract->SetConstant1(5.0f);
        subtract->SetInput2(makeInput(crop));
      }
      else
      {
        subtract->SetInput1(makeInput(crop));
        subtract->SetConstant2(5.0f);
      }
      if (!check(subtract.GetPointer(), [=](float v) { return constantFirst ? 5.0f - v : v - 5.0f; }))
      {
        return false;
      }
    }
    auto square = itk::CudaSquareImageFilter<Image>::New();
    square->SetInPlace(inPlace);
    square->SetInput(makeInput(crop));
    if (!check(square.GetPointer(), [](float v) { return v * v; }))
    {
      return false;
    }
    auto threshold = itk::CudaThresholdImageFilter<Image>::New();
    threshold->SetInPlace(inPlace);
    threshold->SetInput(makeInput(crop));
    threshold->ThresholdBelow(35.0f);
    threshold->SetOutsideValue(-1.0f);
    if (!check(threshold.GetPointer(), [](float v) { return v < 35.0f ? -1.0f : v; }))
    {
      return false;
    }
  }
  // ITK runs in place only when input 0's buffered region equals the
  // output request. Retain both containers to verify CPU and GPU buffer reuse.
  auto checkInPlace = [&](auto * filter, const Image * input, auto expected) {
    const auto pixels = typename Image::PixelContainer::ConstPointer(input->GetPixelContainer());
    const auto manager = input->GetCudaDataManager();
    filter->InPlaceOn();
    if (!check(filter, expected))
    {
      return false;
    }
    if (filter->GetOutput()->GetPixelContainer() != pixels.GetPointer() ||
        filter->GetOutput()->GetCudaDataManager() != manager)
    {
      std::cerr << filter->GetNameOfClass() << " did not reuse the input buffers" << std::endl;
      return false;
    }
    return true;
  };
  {
    auto input = makeInput(crop);
    auto add = itk::CudaAddImageFilter<Image>::New();
    add->SetInput1(input);
    add->SetInput2(makeInput(crop));
    if (!checkInPlace(add.GetPointer(), input.GetPointer(), [](float v) { return v + v; }))
    {
      return false;
    }
  }
  {
    auto input = makeInput(crop);
    auto subtract = itk::CudaSubtractImageFilter<Image>::New();
    subtract->SetInput1(input);
    subtract->SetConstant2(5.0f);
    if (!checkInPlace(subtract.GetPointer(), input.GetPointer(), [](float v) { return v - 5.0f; }))
    {
      return false;
    }
  }
  {
    auto input = makeInput(crop);
    auto square = itk::CudaSquareImageFilter<Image>::New();
    square->SetInput(input);
    if (!checkInPlace(square.GetPointer(), input.GetPointer(), [](float v) { return v * v; }))
    {
      return false;
    }
  }
  {
    auto input = makeInput(crop);
    auto threshold = itk::CudaThresholdImageFilter<Image>::New();
    threshold->SetInput(input);
    threshold->ThresholdBelow(35.0f);
    threshold->SetOutsideValue(-1.0f);
    if (!checkInPlace(threshold.GetPointer(), input.GetPointer(), [](float v) { return v < 35.0f ? -1.0f : v; }))
    {
      return false;
    }
  }
  auto checkRejected = [&](auto * filter) {
    filter->InPlaceOff();
    filter->UpdateOutputInformation();
    filter->GetOutput()->SetRequestedRegion(crop);
    try
    {
      filter->Update();
    }
    catch (const itk::InvalidRequestedRegionError &)
    {
      // ITK rejects requests outside a detached image's buffer before execution.
      return true;
    }
    catch (const itk::ExceptionObject & error)
    {
      if (std::string(error.GetDescription()).find("each buffered region to match") != std::string::npos)
      {
        return true;
      }
      std::cerr << "Unexpected exception: " << error << std::endl;
      return false;
    }
    std::cerr << filter->GetNameOfClass() << " accepted a mismatched buffered region" << std::endl;
    return false;
  };
  auto shifted = crop;
  shifted.SetIndex(0, crop.GetIndex(0) + 1);
  auto reshaped = crop;
  reshaped.SetSize(0, crop.GetSize(0) * 2);
  reshaped.SetSize(1, crop.GetSize(1) / 2);
  for (const auto & mismatched : { full, shifted, reshaped })
  {
    auto square = itk::CudaSquareImageFilter<Image>::New();
    square->SetInput(makeInput(mismatched));
    if (!checkRejected(square.GetPointer()))
    {
      return false;
    }
    auto threshold = itk::CudaThresholdImageFilter<Image>::New();
    threshold->SetInput(makeInput(mismatched));
    if (!checkRejected(threshold.GetPointer()))
    {
      return false;
    }
    for (bool firstInput : { false, true })
    {
      auto add = itk::CudaAddImageFilter<Image>::New();
      add->SetInput1(makeInput(firstInput ? mismatched : crop));
      add->SetInput2(makeInput(firstInput ? crop : mismatched));
      if (!checkRejected(add.GetPointer()))
      {
        return false;
      }
      auto subtract = itk::CudaSubtractImageFilter<Image>::New();
      if (firstInput)
      {
        subtract->SetInput1(makeInput(mismatched));
        subtract->SetConstant2(5.0f);
      }
      else
      {
        subtract->SetConstant1(5.0f);
        subtract->SetInput2(makeInput(mismatched));
      }
      if (!checkRejected(subtract.GetPointer()))
      {
        return false;
      }
    }
  }

  // Exercise successive requests from a real streaming consumer.
  auto source1 = RegionImageSource<Image>::New();
  auto source2 = RegionImageSource<Image>::New();
  source1->GetOutput()->SetLargestPossibleRegion(full);
  source2->GetOutput()->SetLargestPossibleRegion(full);
  auto add = itk::CudaAddImageFilter<Image>::New();
  add->InPlaceOff();
  add->SetInput1(source1->GetOutput());
  add->SetConstant2(5.0f);
  auto square = itk::CudaSquareImageFilter<Image>::New();
  square->InPlaceOff();
  square->SetInput(add->GetOutput());
  auto subtract = itk::CudaSubtractImageFilter<Image>::New();
  subtract->InPlaceOff();
  subtract->SetInput1(square->GetOutput());
  subtract->SetInput2(source2->GetOutput());
  using CPUImage = itk::Image<float, VDimension>;
  auto streamer = itk::StreamingImageFilter<Image, CPUImage>::New();
  streamer->SetInput(subtract->GetOutput());
  streamer->SetNumberOfStreamDivisions(3);
  streamer->Update();
  if (source1->GetNumberOfGeneratedPieces() != 3 || source2->GetNumberOfGeneratedPieces() != 3)
  {
    std::cerr << "Arithmetic pipeline did not stream matching pieces" << std::endl;
    return false;
  }
  itk::ImageRegionConstIteratorWithIndex<CPUImage> it(streamer->GetOutput(), full);
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const float value = ValueAtIndex(it.GetIndex());
    if (it.Get() != (value + 5.0f) * (value + 5.0f) - value)
    {
      std::cerr << "Streamed arithmetic pipeline mismatch" << std::endl;
      return false;
    }
  }
  return true;
}

} // namespace

int
itkCudaArithmeticImageFiltersTest(int, char *[])
{
  using ImageType = itk::CudaImage<float, 2>;
  using BinaryBase = typename itk::CudaAddImageFilter<ImageType>::Superclass;
  using UnaryBase = typename itk::CudaSquareImageFilter<ImageType>::Superclass;
  static_assert(std::is_same_v<typename BinaryBase::Pointer, itk::SmartPointer<BinaryBase>>);
  static_assert(std::is_same_v<typename BinaryBase::ConstPointer, itk::SmartPointer<const BinaryBase>>);
  static_assert(std::is_same_v<typename UnaryBase::Pointer, itk::SmartPointer<UnaryBase>>);
  static_assert(std::is_same_v<typename UnaryBase::ConstPointer, itk::SmartPointer<const UnaryBase>>);

  // Empty pieces must not launch a kernel or access a null device buffer.
  itk::LaunchUnaryKernel<itk::CudaUnaryImageOperation::Square, float, float>(0, nullptr, nullptr, 0, 0, 0);
  itk::LaunchBinaryKernel<itk::CudaBinaryImageOperation::Add, float, float, float>(
    0, nullptr, nullptr, 0, 0, nullptr, 0, 0, 0, 0);


  const std::array<float, 4> values1{ 6.0f, 4.0f, -4.0f, 12.0f };
  const std::array<float, 4> values2{ 2.0f, 0.0f, 2.0f, 3.0f };
  auto                       input1 = MakeImage<ImageType>(values1);
  auto                       input2 = MakeImage<ImageType>(values2);

  auto add = itk::CudaAddImageFilter<ImageType>::New();
  add->SetInput1(input1);
  add->SetInput2(input2);
  add->Update();
  if (!CheckPixels(add->GetOutput(), std::array<float, 4>{ 8.0f, 4.0f, -2.0f, 15.0f }, "CudaAddImageFilter"))
  {
    return EXIT_FAILURE;
  }

  auto addConstant = itk::CudaAddImageFilter<ImageType>::New();
  addConstant->SetInput1(input1);
  addConstant->SetConstant2(2.5f);
  addConstant->Update();
  if (!CheckPixels(
        addConstant->GetOutput(), std::array<float, 4>{ 8.5f, 6.5f, -1.5f, 14.5f }, "CudaAddImageFilter constant"))
  {
    return EXIT_FAILURE;
  }

  auto addConstant1 = itk::CudaAddImageFilter<ImageType>::New();
  addConstant1->SetInput1(3.0f);
  addConstant1->SetInput2(input2);
  addConstant1->Update();
  if (!CheckPixels(addConstant1->GetOutput(),
                   std::array<float, 4>{ 5.0f, 3.0f, 5.0f, 6.0f },
                   "CudaAddImageFilter first operand constant"))
  {
    return EXIT_FAILURE;
  }

  auto subtract = itk::CudaSubtractImageFilter<ImageType>::New();
  subtract->SetInput1(input1);
  subtract->SetInput2(input2);
  subtract->Update();
  if (!CheckPixels(subtract->GetOutput(), std::array<float, 4>{ 4.0f, 4.0f, -6.0f, 9.0f }, "CudaSubtractImageFilter"))
  {
    return EXIT_FAILURE;
  }

  auto multiply = itk::CudaMultiplyImageFilter<ImageType>::New();
  multiply->SetInput1(input1);
  multiply->SetInput2(input2);
  multiply->Update();
  if (!CheckPixels(multiply->GetOutput(), std::array<float, 4>{ 12.0f, 0.0f, -8.0f, 36.0f }, "CudaMultiplyImageFilter"))
  {
    return EXIT_FAILURE;
  }

  auto divide = itk::CudaDivideImageFilter<ImageType, ImageType, ImageType>::New();
  divide->SetInput1(input1);
  divide->SetInput2(input2);
  divide->Update();
  if (!CheckPixels(divide->GetOutput(),
                   std::array<float, 4>{ 3.0f, itk::NumericTraits<float>::max(), -2.0f, 4.0f },
                   "CudaDivideImageFilter"))
  {
    return EXIT_FAILURE;
  }

  const std::array<float, 4> denominatorValues{ 2.0f, 0.5f, 0.0f, -1.0f };
  auto                       denominator = MakeImage<ImageType>(denominatorValues);
  auto                       divideOrZeroOut = itk::CudaDivideOrZeroOutImageFilter<ImageType>::New();
  divideOrZeroOut->SetInput1(input1);
  divideOrZeroOut->SetInput2(denominator);
  divideOrZeroOut->SetThreshold(0.5f);
  divideOrZeroOut->SetConstant(-7.0f);
  divideOrZeroOut->Update();
  if (!CheckPixels(divideOrZeroOut->GetOutput(),
                   std::array<float, 4>{ 3.0f, 8.0f, -7.0f, -7.0f },
                   "CudaDivideOrZeroOutImageFilter"))
  {
    return EXIT_FAILURE;
  }

  auto threshold = itk::CudaThresholdImageFilter<ImageType>::New();
  threshold->SetInput(input1);
  threshold->SetOutsideValue(-100.0f);
  threshold->ThresholdOutside(0.0f, 6.0f);
  threshold->Update();
  if (!CheckPixels(
        threshold->GetOutput(), std::array<float, 4>{ 6.0f, 4.0f, -100.0f, -100.0f }, "CudaThresholdImageFilter"))
  {
    return EXIT_FAILURE;
  }

  using ShortImageType = itk::CudaImage<unsigned short, 2>;
  auto shortInput = MakeImage<ShortImageType>(std::array<unsigned short, 4>{ 1, 2, 3, 4 });
  auto shortAdd = itk::CudaAddImageFilter<ShortImageType>::New();
  shortAdd->SetInput1(shortInput);
  shortAdd->SetInput2(shortInput);
  shortAdd->Update();
  if (!CheckPixels(
        shortAdd->GetOutput(), std::array<unsigned short, 4>{ 2, 4, 6, 8 }, "CudaAddImageFilter unsigned short"))
  {
    return EXIT_FAILURE;
  }

  auto cpuFallback = itk::CudaAddImageFilter<ImageType>::New();
  cpuFallback->SetInput1(input1);
  cpuFallback->SetInput2(input2);
  cpuFallback->GPUEnabledOff();
  cpuFallback->Update();
  if (!CheckPixels(
        cpuFallback->GetOutput(), std::array<float, 4>{ 8.0f, 4.0f, -2.0f, 15.0f }, "CudaAddImageFilter CPU fallback"))
  {
    return EXIT_FAILURE;
  }

  return CheckRequestedRegions<2>() && CheckRequestedRegions<3>() ? EXIT_SUCCESS : EXIT_FAILURE;
}

#include "../src/core/vision/image_ops.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace {

using neuriplo_tasks::BoundingBox;
using neuriplo_tasks::Image;
using neuriplo_tasks::PixelType;
using neuriplo_tasks::vision::ops::DetectionBox;
using neuriplo_tasks::vision::ops::Interpolation;

TEST(ImageOpsTest, SplitsMergesAndSwapsChannels) {
    Image image(2, 1, 3, PixelType::UInt8);
    auto* data = image.data<uint8_t>();
    const std::vector<uint8_t> values{1, 2, 3, 4, 5, 6};
    std::copy(values.begin(), values.end(), data);

    const auto channels = neuriplo_tasks::vision::ops::splitChannels(image.view());
    ASSERT_EQ(channels.size(), 3U);
    EXPECT_EQ(channels[1].data<uint8_t>()[1], 5);

    Image merged = neuriplo_tasks::vision::ops::mergeChannels(channels);
    neuriplo_tasks::vision::ops::swapBgrRgb(merged);
    EXPECT_EQ(merged.data<uint8_t>()[0], 3);
    EXPECT_EQ(merged.data<uint8_t>()[2], 1);
    EXPECT_EQ(merged.data<uint8_t>()[3], 6);
}

TEST(ImageOpsTest, ThresholdsAndFindsRange) {
    Image image(4, 1, 1, PixelType::Float32);
    auto* data = image.data<float>();
    data[0] = -2.0F;
    data[1] = 0.5F;
    data[2] = 2.0F;
    data[3] = 8.0F;

    double minimum = 0.0;
    double maximum = 0.0;
    neuriplo_tasks::vision::ops::minMax(image.view(), minimum, maximum);
    const Image binary = neuriplo_tasks::vision::ops::thresholdBinary(image.view(), 1.0, 4.0);

    EXPECT_DOUBLE_EQ(minimum, -2.0);
    EXPECT_DOUBLE_EQ(maximum, 8.0);
    EXPECT_FLOAT_EQ(binary.data<float>()[1], 0.0F);
    EXPECT_FLOAT_EQ(binary.data<float>()[2], 4.0F);
}

TEST(ImageOpsTest, ResizesAndCopiesRegions) {
    Image source(2, 2, 1, PixelType::UInt8);
    auto* data = source.data<uint8_t>();
    data[0] = 10;
    data[1] = 20;
    data[2] = 30;
    data[3] = 40;

    const Image resized = neuriplo_tasks::vision::ops::resize(source, 1, 1, Interpolation::Area);
    EXPECT_EQ(resized.data<uint8_t>()[0], 25);

    Image destination = Image::zeros(3, 3, 1, PixelType::UInt8);
    neuriplo_tasks::vision::ops::copyRegion(source.view(), BoundingBox(1, 0, 1, 2), destination,
                                            BoundingBox(2, 1, 1, 2));
    EXPECT_EQ(destination.ptr<uint8_t>(1)[2], 20);
    EXPECT_EQ(destination.ptr<uint8_t>(2)[2], 40);
}

TEST(ImageOpsTest, SuppressesOnlyOverlappingLowerScores) {
    const std::vector<DetectionBox> detections{
        {BoundingBox(0, 0, 10, 10), 0.9F, 0},
        {BoundingBox(1, 1, 10, 10), 0.8F, 0},
        {BoundingBox(30, 30, 5, 5), 0.7F, 0},
    };

    const auto kept = neuriplo_tasks::vision::ops::nms(detections, 0.5F);

    ASSERT_EQ(kept.size(), 2U);
    EXPECT_EQ(kept[0], 0);
    EXPECT_EQ(kept[1], 2);
}

} // namespace

#include "neuriplo/tasks/core/vision/stb_io.hpp"

#include <cstdio>
#include <gtest/gtest.h>
#include <string>

namespace {

TEST(StbIoTest, SavesAndLoadsRgbPng) {
    neuriplo_tasks::vision::Image image(2, 1, 3, neuriplo_tasks::vision::PixelType::UInt8);
    auto* pixels = image.data<uint8_t>();
    pixels[0] = 1;
    pixels[1] = 2;
    pixels[2] = 3;
    pixels[3] = 200;
    pixels[4] = 150;
    pixels[5] = 100;
    const std::string path = "stb_io_round_trip.png";

    ASSERT_TRUE(neuriplo_tasks::vision::saveImage(path, image));
    const auto loaded = neuriplo_tasks::vision::loadImage(path, 3);
    std::remove(path.c_str());

    ASSERT_EQ(loaded.sizeBytes(), image.sizeBytes());
    EXPECT_EQ(loaded.width(), 2);
    EXPECT_EQ(loaded.height(), 1);
    EXPECT_EQ(loaded.data<uint8_t>()[4], 150);
}

} // namespace

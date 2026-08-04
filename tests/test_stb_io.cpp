#include "neuriplo/tasks/core/vision/stb_io.hpp"

#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

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

TEST(StbIoTest, DecodesEncodedBytesFromMemory) {
    neuriplo_tasks::vision::Image image(2, 1, 3, neuriplo_tasks::vision::PixelType::UInt8);
    auto* pixels = image.data<uint8_t>();
    pixels[0] = 1;
    pixels[1] = 2;
    pixels[2] = 3;
    pixels[3] = 200;
    pixels[4] = 150;
    pixels[5] = 100;
    const std::string path = "stb_io_decode_source.png";
    ASSERT_TRUE(neuriplo_tasks::vision::saveImage(path, image));

    std::ifstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream.is_open());
    const std::vector<uint8_t> encoded(std::istreambuf_iterator<char>(stream), {});
    stream.close();
    std::remove(path.c_str());
    ASSERT_FALSE(encoded.empty());

    // Decoding the bytes must match decoding the same bytes from disk.
    const auto decoded = neuriplo_tasks::vision::decodeImage(encoded.data(), encoded.size(), 3);
    ASSERT_EQ(decoded.sizeBytes(), image.sizeBytes());
    EXPECT_EQ(decoded.width(), 2);
    EXPECT_EQ(decoded.height(), 1);
    EXPECT_EQ(decoded.data<uint8_t>()[4], 150);
}

TEST(StbIoTest, DecodeRejectsEmptyAndInvalidBuffers) {
    EXPECT_THROW(neuriplo_tasks::vision::decodeImage(nullptr, 0), std::runtime_error);

    const std::vector<uint8_t> garbage(64, 0x7f);
    EXPECT_THROW(neuriplo_tasks::vision::decodeImage(garbage.data(), garbage.size()), std::runtime_error);
}

} // namespace

#pragma once

#include "neuriplo/tasks/core/image.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace neuriplo_tasks::vision_test {

[[nodiscard]] inline Image makeImage(int width, int height, int channels = 3, std::uint8_t value = 0) {
    Image image(width, height, channels, PixelType::UInt8);
    std::fill(image.raw(), image.raw() + image.sizeBytes(), value);
    return image;
}

[[nodiscard]] inline Image makeFloatImage(int width, int height, int channels = 1, float value = 0.0F) {
    Image image(width, height, channels, PixelType::Float32);
    float* data = image.data<float>();
    std::fill(data, data + image.totalPixels() * static_cast<std::size_t>(channels), value);
    return image;
}

inline void fillRect(Image& image, int x0, int y0, int x1, int y1, std::uint8_t value) {
    const int min_x = std::max(0, std::min(x0, x1));
    const int max_x = std::min(image.width(), std::max(x0, x1));
    const int min_y = std::max(0, std::min(y0, y1));
    const int max_y = std::min(image.height(), std::max(y0, y1));
    for (int y = min_y; y < max_y; ++y) {
        std::uint8_t* row = image.ptr<std::uint8_t>(y);
        for (int x = min_x; x < max_x; ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                row[static_cast<std::size_t>(x) * static_cast<std::size_t>(image.channels()) +
                    static_cast<std::size_t>(c)] = value;
            }
        }
    }
}

} // namespace neuriplo_tasks::vision_test

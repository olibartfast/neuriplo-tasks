#pragma once

#include "neuriplo/tasks/core/vision/image.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace neuriplo_tasks::vision {

[[nodiscard]] Image loadImage(const std::string& path, int desired_channels = 3);

/**
 * @brief Decode an encoded image (JPEG, PNG, BMP, TGA) already held in memory.
 *
 * Same result as loadImage, without a filesystem round trip. Serving runtimes
 * receive encoded bytes on the wire and have no file to point at.
 *
 * @throws std::runtime_error if the buffer is empty or not a decodable image.
 */
[[nodiscard]] Image decodeImage(const std::uint8_t* bytes, std::size_t size, int desired_channels = 3);

bool saveImage(const std::string& path, const Image& image);

} // namespace neuriplo_tasks::vision

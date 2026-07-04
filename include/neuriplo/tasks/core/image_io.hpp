#pragma once

#include "neuriplo/tasks/core/image.hpp"

#include <string>

namespace neuriplo_tasks {

/**
 * @brief Load an image from disk (PNG / JPEG / BMP / TGA / etc.).
 *
 * Backed by stb_image. The returned Image is always interleaved HxWxC (NHWC
 * pixel layout) and has PixelType::UInt8. Pass desired_channels = 3 for RGB
 * or 4 for RGBA; 0 forces the file's native channel count.
 *
 * @param path Filesystem path to the image.
 * @param desired_channels 1 (grey), 2 (grey+alpha), 3 (RGB), 4 (RGBA), 0 (native).
 * @return Owning Image. Throws std::runtime_error on failure.
 */
[[nodiscard]] Image loadImage(const std::string& path, int desired_channels = 3);

/**
 * @brief Save an image to disk. Format inferred from path extension.
 *
 * Backed by stb_image_write. Supports .png, .jpg, .bmp, .tga. The Image must
 * have PixelType::UInt8 and 1/2/3/4 channels.
 *
 * @param path Destination path (extension selects the encoder).
 * @param image Image to write.
 * @return true on success, false on failure.
 */
bool saveImage(const std::string& path, const Image& image);

} // namespace neuriplo_tasks

#include "neuriplo/tasks/core/vision/stb_io.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

// stb is vendored third-party code that does not meet the project's -Werror
// warning standards (old-style casts, missing-field-initializers, shadowing).
// Suppress warnings only across the stb includes / implementation macros.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wunused"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wnull-dereference"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wformat=2"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace neuriplo_tasks::vision {

namespace {

[[nodiscard]] const char* stbFailure() { return stbi_failure_reason(); }

} // namespace

vision::Image loadImage(const std::string& path, int desired_channels) {
    int width = 0;
    int height = 0;
    int channels = 0;
    const int req = (desired_channels > 0) ? desired_channels : STBI_default;
    stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels, req);
    if (pixels == nullptr) {
        throw std::runtime_error("loadImage failed for '" + path + "': " + stbFailure());
    }
    const int out_channels = (req != STBI_default) ? req : channels;

    vision::Image image(width, height, out_channels, vision::PixelType::UInt8);
    const std::size_t bytes = image.sizeBytes();
    std::memcpy(image.data<std::uint8_t>(), pixels, bytes);
    stbi_image_free(pixels);
    return image;
}

bool saveImage(const std::string& path, const vision::Image& image) {
    if (image.empty() || image.pixelType() != vision::PixelType::UInt8) {
        return false;
    }
    const int w = image.width();
    const int h = image.height();
    const int c = image.channels();

    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }
    const std::string ext = path.substr(dot + 1);

    int success = 0;
    if (ext == "png") {
        success = stbi_write_png(path.c_str(), w, h, c, image.raw(), static_cast<int>(image.stride()));
    } else if (ext == "jpg" || ext == "jpeg") {
        success = stbi_write_jpg(path.c_str(), w, h, c, image.raw(), 90);
    } else if (ext == "bmp") {
        success = stbi_write_bmp(path.c_str(), w, h, c, image.raw());
    } else if (ext == "tga") {
        success = stbi_write_tga(path.c_str(), w, h, c, image.raw());
    } else {
        return false;
    }
    return success != 0;
}

} // namespace neuriplo_tasks::vision

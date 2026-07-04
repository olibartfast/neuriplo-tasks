#pragma once

#include "neuriplo/tasks/core/bounding_box.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Pixel data type tag, replacing OpenCV CV_8U / CV_32F / CV_32S macros.
 */
enum class PixelType : uint8_t {
    UInt8,
    Float32,
    Int32,
};

/**
 * @brief Size of the bytes occupied by one element of the given pixel type.
 */
[[nodiscard]] inline std::size_t pixelTypeSize(PixelType t) noexcept {
    switch (t) {
    case PixelType::UInt8:
        return sizeof(uint8_t);
    case PixelType::Float32:
        return sizeof(float);
    case PixelType::Int32:
        return sizeof(int32_t);
    }
    return 0;
}

/**
 * @brief Two-dimensional size in pixels, replacing cv::Size in public headers.
 */
struct Size {
    int width{0};
    int height{0};

    Size() = default;
    Size(int w, int h) : width(w), height(h) {}

    [[nodiscard]] int area() const noexcept { return width * height; }
    [[nodiscard]] bool isEmpty() const noexcept { return width <= 0 || height <= 0; }

    friend bool operator==(const Size& lhs, const Size& rhs) noexcept {
        return lhs.width == rhs.width && lhs.height == rhs.height;
    }
    friend bool operator!=(const Size& lhs, const Size& rhs) noexcept { return !(lhs == rhs); }
};

/**
 * @brief Non-owning view over a contiguous row-major image buffer.
 *
 * Wraps external memory (e.g. a cv::Mat::data pointer in consumers) without
 * copying. Does not manage lifetime of the pointed-to buffer. Use Image for
 * owning storage.
 */
class ImageView {
  public:
    ImageView() noexcept = default;

    ImageView(const void* data, int width, int height, int channels, PixelType pixel_type) noexcept
        : data_(const_cast<void*>(data)), width_(width), height_(height), channels_(channels), pixel_type_(pixel_type) {
    }

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] int cols() const noexcept { return width_; }
    [[nodiscard]] int rows() const noexcept { return height_; }
    [[nodiscard]] int channels() const noexcept { return channels_; }
    [[nodiscard]] PixelType pixelType() const noexcept { return pixel_type_; }
    [[nodiscard]] bool empty() const noexcept { return data_ == nullptr || width_ <= 0 || height_ <= 0; }

    [[nodiscard]] std::size_t stride() const noexcept {
        return static_cast<std::size_t>(width_) * static_cast<std::size_t>(channels_) * pixelTypeSize(pixel_type_);
    }

    [[nodiscard]] std::size_t totalPixels() const noexcept {
        return static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
    }

    [[nodiscard]] std::size_t sizeBytes() const noexcept {
        return totalPixels() * static_cast<std::size_t>(channels_) * pixelTypeSize(pixel_type_);
    }

    template <typename T> [[nodiscard]] const T* data() const noexcept { return reinterpret_cast<const T*>(data_); }

    template <typename T> [[nodiscard]] const T* ptr(int row) const noexcept {
        return reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(data_) +
                                          static_cast<std::size_t>(row) * stride());
    }

    [[nodiscard]] const void* raw() const noexcept { return data_; }

  protected:
    void* data_{nullptr};
    int width_{0};
    int height_{0};
    int channels_{0};
    PixelType pixel_type_{PixelType::UInt8};
};

/**
 * @brief Owning contiguous row-major image buffer, replacing cv::Mat.
 *
 * Owns its storage as a std::vector<std::uint8_t>. Layout is HxWxC row-major,
 * interleaved channels (NHWC pixel layout). For NCHW planar layout, use the
 * per-channel split/merge helpers in image_ops.
 */
class Image {
  public:
    Image() = default;

    Image(int width, int height, int channels, PixelType pixel_type)
        : width_(width), height_(height), channels_(channels), pixel_type_(pixel_type),
          buffer_(computeSizeBytes(width, height, channels, pixel_type), 0) {}

    static Image zeros(int width, int height, int channels, PixelType pixel_type) {
        return Image(width, height, channels, pixel_type);
    }

    static Image uninit(int width, int height, int channels, PixelType pixel_type) {
        Image img;
        img.width_ = width;
        img.height_ = height;
        img.channels_ = channels;
        img.pixel_type_ = pixel_type;
        img.buffer_.resize(computeSizeBytes(width, height, channels, pixel_type));
        return img;
    }

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] int cols() const noexcept { return width_; }
    [[nodiscard]] int rows() const noexcept { return height_; }
    [[nodiscard]] int channels() const noexcept { return channels_; }
    [[nodiscard]] PixelType pixelType() const noexcept { return pixel_type_; }
    [[nodiscard]] bool empty() const noexcept { return buffer_.empty() || width_ <= 0 || height_ <= 0; }
    [[nodiscard]] Size size() const noexcept { return Size(width_, height_); }

    [[nodiscard]] std::size_t stride() const noexcept {
        return static_cast<std::size_t>(width_) * static_cast<std::size_t>(channels_) * pixelTypeSize(pixel_type_);
    }

    [[nodiscard]] std::size_t totalPixels() const noexcept {
        return static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
    }

    [[nodiscard]] std::size_t sizeBytes() const noexcept { return buffer_.size(); }

    template <typename T> [[nodiscard]] T* data() noexcept { return reinterpret_cast<T*>(buffer_.data()); }

    template <typename T> [[nodiscard]] const T* data() const noexcept {
        return reinterpret_cast<const T*>(buffer_.data());
    }

    template <typename T> [[nodiscard]] T* ptr(int row) noexcept {
        return reinterpret_cast<T*>(buffer_.data() + static_cast<std::size_t>(row) * stride());
    }

    template <typename T> [[nodiscard]] const T* ptr(int row) const noexcept {
        return reinterpret_cast<const T*>(buffer_.data() + static_cast<std::size_t>(row) * stride());
    }

    [[nodiscard]] std::uint8_t* raw() noexcept { return buffer_.data(); }
    [[nodiscard]] const std::uint8_t* raw() const noexcept { return buffer_.data(); }

    [[nodiscard]] ImageView view() const noexcept {
        return ImageView(buffer_.data(), width_, height_, channels_, pixel_type_);
    }

    [[nodiscard]] Image clone() const { return *this; }

    [[nodiscard]] ImageView subregion(const BoundingBox& roi) const noexcept {
        return ImageView(ptr<std::uint8_t>(roi.y) + static_cast<std::size_t>(roi.x) *
                                                        static_cast<std::size_t>(channels_) *
                                                        pixelTypeSize(pixel_type_),
                         roi.width, roi.height, channels_, pixel_type_);
    }

    /**
     * @brief In-place affine element transform: dst = src * alpha + beta.
     * Replaces cv::Mat::convertTo(dst, type, alpha, beta). Changes pixel_type
     * to target_type and resizes the buffer.
     */
    void convertTo(PixelType target_type, double alpha = 1.0, double beta = 0.0) {
        if (target_type == pixel_type_ && alpha == 1.0 && beta == 0.0) {
            return;
        }
        const std::size_t count = totalPixels() * static_cast<std::size_t>(channels_);
        std::vector<std::uint8_t> out(count * pixelTypeSize(target_type));
        for (std::size_t i = 0; i < count; ++i) {
            double v = readElementAsDouble(i) * alpha + beta;
            writeElementAs(out.data(), i, target_type, v);
        }
        pixel_type_ = target_type;
        buffer_ = std::move(out);
    }

    /**
     * @brief Returns a new Image whose pixel type is target_type, with the
     * same geometry. Convenience wrapper around convertTo for immutable use.
     */
    [[nodiscard]] Image convertedTo(PixelType target_type, double alpha = 1.0, double beta = 0.0) const {
        Image out = *this;
        out.convertTo(target_type, alpha, beta);
        return out;
    }

  private:
    int width_{0};
    int height_{0};
    int channels_{0};
    PixelType pixel_type_{PixelType::UInt8};
    std::vector<std::uint8_t> buffer_;

    static std::size_t computeSizeBytes(int width, int height, int channels, PixelType pixel_type) noexcept {
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(channels) *
               pixelTypeSize(pixel_type);
    }

    [[nodiscard]] double readElementAsDouble(std::size_t flat_index) const noexcept {
        const std::uint8_t* p = buffer_.data() + flat_index * pixelTypeSize(pixel_type_);
        switch (pixel_type_) {
        case PixelType::UInt8:
            return static_cast<double>(*p);
        case PixelType::Float32:
            return static_cast<double>(*reinterpret_cast<const float*>(p));
        case PixelType::Int32:
            return static_cast<double>(*reinterpret_cast<const int32_t*>(p));
        }
        return 0.0;
    }

    static void writeElementAs(std::uint8_t* out, std::size_t flat_index, PixelType type, double value) noexcept {
        std::uint8_t* p = out + flat_index * pixelTypeSize(type);
        switch (type) {
        case PixelType::UInt8:
            *p = static_cast<uint8_t>(std::clamp(value, 0.0, 255.0));
            break;
        case PixelType::Float32:
            *reinterpret_cast<float*>(p) = static_cast<float>(value);
            break;
        case PixelType::Int32:
            *reinterpret_cast<int32_t*>(p) = static_cast<int32_t>(value);
            break;
        }
    }
};

} // namespace neuriplo_tasks

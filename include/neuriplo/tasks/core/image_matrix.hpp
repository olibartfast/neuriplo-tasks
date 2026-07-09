#pragma once

#include "neuriplo/tasks/core/image.hpp"

#include <memory>

namespace neuriplo_tasks {

/**
 * @brief Opaque image buffer handle without OpenCV in public headers.
 *
 * Internally backed by neuriplo_tasks::Image (owning contiguous buffer).
 * Exposes geometry, pixel type, and clone. Use the data accessors below
 * to read raw bytes; for typed access, copy into an Image of known layout.
 */
class ImageMatrix {
  public:
    ImageMatrix();
    ImageMatrix(const ImageMatrix& other);
    ImageMatrix& operator=(const ImageMatrix& other);
    ImageMatrix(ImageMatrix&& other) noexcept;
    ImageMatrix& operator=(ImageMatrix&& other) noexcept;
    ~ImageMatrix();

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] int rows() const noexcept;
    [[nodiscard]] int cols() const noexcept;
    [[nodiscard]] int channels() const noexcept;
    [[nodiscard]] PixelType pixelType() const noexcept;
    [[nodiscard]] ImageMatrix clone() const;

    /** Raw byte access (read-only) into the underlying contiguous buffer. */
    [[nodiscard]] const std::uint8_t* data() const noexcept;
    [[nodiscard]] std::size_t sizeBytes() const noexcept;

  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;

    friend ImageMatrix fromImage(Image image);
    friend const Image& toImage(const ImageMatrix& matrix);
};

[[nodiscard]] ImageMatrix fromImage(Image image);
[[nodiscard]] const Image& toImage(const ImageMatrix& matrix);

} // namespace neuriplo_tasks

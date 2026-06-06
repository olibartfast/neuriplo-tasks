#pragma once

#include <memory>

namespace cv {
class Mat;
}

namespace vision_core {

/**
 * @brief Opaque image buffer handle without OpenCV in public headers.
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
    [[nodiscard]] int type() const noexcept;
    [[nodiscard]] ImageMatrix clone() const;

  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;

    friend const cv::Mat& toCvMat(const ImageMatrix& matrix);
    friend cv::Mat& mutableCvMat(ImageMatrix& matrix);
};

} // namespace vision_core

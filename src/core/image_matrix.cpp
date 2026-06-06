#include "neuriplo/tasks/core/image_matrix.hpp"

#include <opencv2/core/mat.hpp>
#include <utility>

namespace neuriplo_tasks {

struct ImageMatrix::Impl {
    cv::Mat data;
};

ImageMatrix::ImageMatrix() : impl_(std::make_shared<Impl>()) {}

ImageMatrix::ImageMatrix(const ImageMatrix& other)
    : impl_(other.impl_ ? std::make_shared<Impl>(*other.impl_) : std::make_shared<Impl>()) {}

ImageMatrix& ImageMatrix::operator=(const ImageMatrix& other) {
    if (this != &other) {
        impl_ = other.impl_ ? std::make_shared<Impl>(*other.impl_) : std::make_shared<Impl>();
    }
    return *this;
}

ImageMatrix::ImageMatrix(ImageMatrix&& other) noexcept : impl_(std::move(other.impl_)) {
    if (!impl_) {
        impl_ = std::make_shared<Impl>();
    }
}

ImageMatrix& ImageMatrix::operator=(ImageMatrix&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
        if (!impl_) {
            impl_ = std::make_shared<Impl>();
        }
    }
    return *this;
}

ImageMatrix::~ImageMatrix() = default;

bool ImageMatrix::empty() const noexcept { return !impl_ || impl_->data.empty(); }

int ImageMatrix::rows() const noexcept { return impl_ ? impl_->data.rows : 0; }

int ImageMatrix::cols() const noexcept { return impl_ ? impl_->data.cols : 0; }

int ImageMatrix::type() const noexcept { return impl_ ? impl_->data.type() : 0; }

ImageMatrix ImageMatrix::clone() const {
    ImageMatrix copy;
    if (impl_ && !impl_->data.empty()) {
        copy.impl_->data = impl_->data.clone();
    }
    return copy;
}

const cv::Mat& toCvMat(const ImageMatrix& matrix) {
    static const cv::Mat kEmpty;
    if (!matrix.impl_ || matrix.impl_->data.empty()) {
        return kEmpty;
    }
    return matrix.impl_->data;
}

cv::Mat& mutableCvMat(ImageMatrix& matrix) {
    if (!matrix.impl_) {
        matrix.impl_ = std::make_shared<ImageMatrix::Impl>();
    }
    return matrix.impl_->data;
}

ImageMatrix fromCvMat(const cv::Mat& mat) {
    ImageMatrix matrix;
    mutableCvMat(matrix) = mat;
    return matrix;
}

} // namespace neuriplo_tasks

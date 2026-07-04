#include "neuriplo/tasks/core/image_matrix.hpp"

#include <utility>

namespace neuriplo_tasks {

struct ImageMatrix::Impl {
    Image data;
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

int ImageMatrix::rows() const noexcept { return impl_ ? impl_->data.height() : 0; }

int ImageMatrix::cols() const noexcept { return impl_ ? impl_->data.width() : 0; }

int ImageMatrix::channels() const noexcept { return impl_ ? impl_->data.channels() : 0; }

PixelType ImageMatrix::pixelType() const noexcept { return impl_ ? impl_->data.pixelType() : PixelType::UInt8; }

ImageMatrix ImageMatrix::clone() const {
    ImageMatrix copy;
    if (impl_ && !impl_->data.empty()) {
        copy.impl_->data = impl_->data.clone();
    }
    return copy;
}

const std::uint8_t* ImageMatrix::data() const noexcept {
    return (impl_ && !impl_->data.empty()) ? impl_->data.raw() : nullptr;
}

std::size_t ImageMatrix::sizeBytes() const noexcept {
    return (impl_ && !impl_->data.empty()) ? impl_->data.sizeBytes() : 0;
}

ImageMatrix fromImage(Image image) {
    ImageMatrix matrix;
    matrix.impl_->data = std::move(image);
    return matrix;
}

const Image& toImage(const ImageMatrix& matrix) {
    static const Image kEmpty;
    if (!matrix.impl_ || matrix.impl_->data.empty()) {
        return kEmpty;
    }
    return matrix.impl_->data;
}

} // namespace neuriplo_tasks

#include "neuriplo/tasks/core/vision/opencv_adapter.hpp"

#include <cstring>
#include <stdexcept>

namespace neuriplo_tasks::vision::opencv {
namespace {

vision::PixelType pixelTypeFromCvDepth(int depth) {
    switch (depth) {
    case CV_8U:
        return vision::PixelType::UInt8;
    case CV_32F:
        return vision::PixelType::Float32;
    case CV_32S:
        return vision::PixelType::Int32;
    default:
        throw std::invalid_argument("OpenCV adapter supports only CV_8U, CV_32F, and CV_32S");
    }
}

int cvDepthFromPixelType(vision::PixelType pixel_type) {
    switch (pixel_type) {
    case vision::PixelType::UInt8:
        return CV_8U;
    case vision::PixelType::Float32:
        return CV_32F;
    case vision::PixelType::Int32:
        return CV_32S;
    }
    throw std::invalid_argument("Unsupported vision pixel type");
}

} // namespace

vision::ImageView toImageView(const cv::Mat& mat) {
    if (!mat.isContinuous()) {
        throw std::invalid_argument("Cannot wrap a non-contiguous cv::Mat as ImageView");
    }
    return vision::ImageView(mat.data, mat.cols, mat.rows, mat.channels(), pixelTypeFromCvDepth(mat.depth()));
}

vision::Image copyFromCvMat(const cv::Mat& mat) {
    vision::Image image(mat.cols, mat.rows, mat.channels(), pixelTypeFromCvDepth(mat.depth()));
    const std::size_t row_bytes = image.stride();
    for (int row = 0; row < mat.rows; ++row) {
        std::memcpy(image.raw() + static_cast<std::size_t>(row) * row_bytes, mat.ptr(row), row_bytes);
    }
    return image;
}

cv::Mat toCvMat(const vision::ImageView& image) {
    cv::Mat mat(image.height(), image.width(), CV_MAKETYPE(cvDepthFromPixelType(image.pixelType()), image.channels()));
    std::memcpy(mat.data, image.raw(), image.sizeBytes());
    return mat;
}

cv::Mat toCvMat(const ImageMatrix& matrix) { return toCvMat(toImage(matrix).view()); }

} // namespace neuriplo_tasks::vision::opencv

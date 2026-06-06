#pragma once

#include "vision-core/core/bounding_box.hpp"
#include "vision-core/core/image_matrix.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace vision_core {

[[nodiscard]] inline cv::Rect toCvRect(const BoundingBox& box) { return cv::Rect(box.x, box.y, box.width, box.height); }

[[nodiscard]] inline BoundingBox fromCvRect(const cv::Rect& rect) {
    return BoundingBox(rect.x, rect.y, rect.width, rect.height);
}

[[nodiscard]] const cv::Mat& toCvMat(const ImageMatrix& matrix);
[[nodiscard]] cv::Mat& mutableCvMat(ImageMatrix& matrix);
[[nodiscard]] ImageMatrix fromCvMat(const cv::Mat& mat);

} // namespace vision_core

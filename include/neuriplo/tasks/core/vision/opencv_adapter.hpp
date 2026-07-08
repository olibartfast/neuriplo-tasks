#pragma once

#include "neuriplo/tasks/core/image_matrix.hpp"
#include "neuriplo/tasks/core/vision/geometry.hpp"
#include "neuriplo/tasks/core/vision/image.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace neuriplo_tasks::vision::opencv {

[[nodiscard]] ImageView toImageView(const cv::Mat& mat);
[[nodiscard]] Image copyFromCvMat(const cv::Mat& mat);
[[nodiscard]] cv::Mat toCvMat(const ImageView& image);
[[nodiscard]] cv::Mat toCvMat(const ImageMatrix& matrix);

[[nodiscard]] inline cv::Rect toCvRect(const Rect& rect) { return cv::Rect(rect.x, rect.y, rect.width, rect.height); }

[[nodiscard]] inline Rect fromCvRect(const cv::Rect& rect) { return Rect(rect.x, rect.y, rect.width, rect.height); }

[[nodiscard]] inline cv::Size toCvSize(const Size& size) { return cv::Size(size.width, size.height); }
[[nodiscard]] inline Size fromCvSize(const cv::Size& size) { return Size(size.width, size.height); }

} // namespace neuriplo_tasks::vision::opencv

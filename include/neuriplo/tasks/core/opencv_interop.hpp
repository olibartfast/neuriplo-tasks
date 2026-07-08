#pragma once

#include "neuriplo/tasks/core/vision/opencv_adapter.hpp"

namespace neuriplo_tasks {

[[nodiscard]] inline ImageMatrix fromCvMat(const cv::Mat& mat) { return fromImage(vision::opencv::copyFromCvMat(mat)); }
using vision::opencv::fromCvRect;
using vision::opencv::fromCvSize;
using vision::opencv::toCvMat;
using vision::opencv::toCvRect;
using vision::opencv::toCvSize;

} // namespace neuriplo_tasks

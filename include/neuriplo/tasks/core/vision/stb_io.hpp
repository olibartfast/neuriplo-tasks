#pragma once

#include "neuriplo/tasks/core/vision/image.hpp"

#include <string>

namespace neuriplo_tasks::vision {

[[nodiscard]] Image loadImage(const std::string& path, int desired_channels = 3);

bool saveImage(const std::string& path, const Image& image);

} // namespace neuriplo_tasks::vision

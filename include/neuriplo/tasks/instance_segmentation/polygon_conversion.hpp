#pragma once

#include "neuriplo/tasks/core/image_matrix.hpp"
#include "neuriplo/tasks/core/segmentation_types.hpp"

#include <vector>

namespace neuriplo_tasks {

[[nodiscard]] std::vector<SegmentationPolygon> maskToPolygons(const ImageMatrix& mask);

} // namespace neuriplo_tasks

#pragma once

#include "neuriplo/tasks/core/vision/geometry.hpp"

#include <cstdint>
#include <vector>

namespace neuriplo_tasks {

enum class SegmentationOutput : uint8_t {
    Mask,
    Polygon,
};

struct SegmentationPolygon {
    std::vector<vision::Point2f> exterior;
    std::vector<std::vector<vision::Point2f>> holes;
};

} // namespace neuriplo_tasks

#pragma once

#include "neuriplo/tasks/core/bounding_box.hpp"
#include "neuriplo/tasks/core/image.hpp"

#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Utilities for bounding box calculations and transformations
 *
 * Provides static methods for common bounding box operations including
 * letterbox scaling, coordinate transformation, and clamping.
 */
class BBoxProcessor {
  public:
    /**
     * @brief Calculate bounding box with letterbox scaling
     *
     * Converts bounding box coordinates from network space to original image space,
     * accounting for letterbox padding used during preprocessing.
     */
    [[nodiscard]] static BoundingBox calculate_bounding_box(const Size& image_size, const std::vector<float>& bbox,
                                                           int network_width, int network_height);

    /**
     * @brief Calculate bounding box from XYXY coordinates with letterbox scaling
     */
    [[nodiscard]] static BoundingBox calculate_bounding_box_from_xyxy(const Size& image_size, const std::vector<float>& bbox,
                                                                     int network_width, int network_height);

    /**
     * @brief Scale bbox from network coordinates to image coordinates
     *
     * Simple scaling without letterbox consideration.
     */
    [[nodiscard]] static BoundingBox scale_to_original(const BoundingBox& bbox, const Size& original_size,
                                                       const Size& network_size) noexcept;

    /**
     * @brief Clamp bounding box to image boundaries
     */
    static void clamp_to_bounds(BoundingBox& box, const Size& image_size) noexcept;
};

} // namespace neuriplo_tasks

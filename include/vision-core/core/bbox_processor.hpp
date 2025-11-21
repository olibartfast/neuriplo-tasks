#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace vision_core {

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
     * 
     * @param image_size Original image size
     * @param bbox Raw bbox coordinates [x_center, y_center, width, height]
     * @param network_width Network input width
     * @param network_height Network input height
     * @return Scaled bounding box in original image coordinates
     * @throws std::invalid_argument if bbox has fewer than 4 elements
     */
    [[nodiscard]] static cv::Rect calculate_bounding_box(
        const cv::Size& image_size,
        const std::vector<float>& bbox,
        int network_width,
        int network_height
    );

    /**
     * @brief Calculate bounding box from XYXY coordinates with letterbox scaling
     * 
     * Converts bounding box coordinates from network space (x1, y1, x2, y2) to original image space,
     * accounting for letterbox padding used during preprocessing.
     * 
     * @param image_size Original image size
     * @param bbox Raw bbox coordinates [x1, y1, x2, y2]
     * @param network_width Network input width
     * @param network_height Network input height
     * @return Scaled bounding box in original image coordinates
     */
    [[nodiscard]] static cv::Rect calculate_bounding_box_from_xyxy(
        const cv::Size& image_size,
        const std::vector<float>& bbox,
        int network_width,
        int network_height
    );

    /**
     * @brief Scale bbox from network coordinates to image coordinates
     * 
     * Simple scaling without letterbox consideration. Use calculate_bounding_box()
     * if letterbox padding was applied during preprocessing.
     * 
     * @param bbox Bounding box in network coordinates
     * @param original_size Target image size
     * @param network_size Network input size
     * @return Scaled bounding box
     */
    [[nodiscard]] static cv::Rect scale_to_original(
        const cv::Rect& bbox,
        const cv::Size& original_size,
        const cv::Size& network_size
    ) noexcept;

    /**
     * @brief Clamp bounding box to image boundaries
     * 
     * @param box Bounding box to clamp (modified in place)
     * @param image_size Image dimensions for clamping
     */
    static void clamp_to_bounds(cv::Rect& box, const cv::Size& image_size) noexcept;
};

} // namespace vision_core

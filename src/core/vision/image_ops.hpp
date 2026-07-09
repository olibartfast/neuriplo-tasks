#pragma once

#include "neuriplo/tasks/core/bounding_box.hpp"
#include "neuriplo/tasks/core/image.hpp"

#include <cstdint>
#include <vector>

namespace neuriplo_tasks::vision::ops {

enum class Interpolation : std::uint8_t {
    Linear,
    Area,
    Cubic,
};

/**
 * @brief Resize an image to a new geometry.
 *
 * Replaces cv::resize. INTER_AREA is hand-rolled area-pixel downsampling
 * (box filter), INTER_CUBIC is hand-rolled bicubic, INTER_LINEAR is bilinear.
 */
[[nodiscard]] Image resize(const Image& src, int dst_width, int dst_height,
                           Interpolation interp = Interpolation::Linear);
[[nodiscard]] Image resize(const ImageView& src, int dst_width, int dst_height,
                           Interpolation interp = Interpolation::Linear);

inline Image resize(const Image& src, const Size& dst, Interpolation interp = Interpolation::Linear) {
    return resize(src, dst.width, dst.height, interp);
}
inline Image resize(const ImageView& src, const Size& dst, Interpolation interp = Interpolation::Linear) {
    return resize(src, dst.width, dst.height, interp);
}

/**
 * @brief In-place BGR <-> RGB channel swap for 3-channel images.
 * Replaces cv::cvtColor(BGR2RGB) / cv::cvtColor(RGB2BGR).
 */
void swapBgrRgb(Image& img);

/**
 * @brief Binarize a single-channel image: dst = src > threshold ? max_value : 0.
 * Replaces cv::threshold(THRESH_BINARY).
 */
[[nodiscard]] Image thresholdBinary(const ImageView& src, double threshold, double max_value);

/**
 * @brief Split an interleaved HxWxC image into C single-channel images.
 * Replaces cv::split.
 */
[[nodiscard]] std::vector<Image> splitChannels(const ImageView& src);

/**
 * @brief Merge C single-channel images of equal geometry into one interleaved
 * HxWxC image. Replaces cv::merge.
 */
[[nodiscard]] Image mergeChannels(const std::vector<Image>& channels);

/**
 * @brief Locate the min and max element values in a single-channel image.
 * Replaces cv::minMaxLoc (ignores the location outputs).
 */
void minMax(const ImageView& src, double& out_min, double& out_max);

/**
 * @brief Copy a rectangular region from src into the same rectangular region
 * of dst. Replaces cv::Mat::copyTo into an ROI (mat_full(bbox) = ...).
 */
void copyRegion(const ImageView& src, const BoundingBox& src_roi, Image& dst, const BoundingBox& dst_roi);

/**
 * @brief Per-image non-maximum suppression by IoU.
 *
 * Promoted from the existing hand-rolled implementation in
 * src/object_detection/yolo_postprocessor.cpp. Replaces cv::dnn::NMSBoxes.
 */
struct DetectionBox {
    BoundingBox bbox;
    float score{0.0f};
    int class_id{0};
};

[[nodiscard]] std::vector<int> nms(const std::vector<DetectionBox>& detections, float iou_threshold);

} // namespace neuriplo_tasks::vision::ops

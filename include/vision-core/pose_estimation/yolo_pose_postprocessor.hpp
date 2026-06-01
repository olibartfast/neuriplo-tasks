#pragma once

#include "vision-core/core/task_interface.hpp"
#include "vision-core/pose_estimation/pose_postprocessor.hpp"

#include <opencv2/core.hpp>

namespace vision_core {

/**
 * @brief YOLO pose postprocessor for Ultralytics YOLO pose models
 *
 * Supports YOLOv5, YOLOv8, YOLOv11, YOLO26 pose variants.
 *
 * Output tensor format (auto-detected by shape):
 *   - YOLOv8/v11+ (no objectness): [batch, 4+1+3*kpts, anchors]
 *   - YOLOv5      (objectness):    [batch, anchors, 4+1+3*kpts]
 *
 * Per detection: cx, cy, w, h, confidence, kpt0_x, kpt0_y, kpt0_conf, ...
 */
class YoloPosePostprocessor : public PosePostprocessor {
  public:
    YoloPosePostprocessor(const cv::Size& input_size, float confidence_threshold, float nms_threshold);

    std::vector<PoseEstimation> postprocess(const std::vector<Tensor>& tensors, const cv::Size& original_size,
                                            const cv::Size& input_size) override;

  private:
    cv::Size input_size_;
    float confidence_threshold_;
    float nms_threshold_;

    cv::Rect scaleBoxToOriginal(float cx, float cy, float w, float h, const cv::Size& frame_size) const;
    cv::Point2f scaleKptToOriginal(float kx, float ky, const cv::Size& frame_size) const;
    void applyNMS(std::vector<PoseEstimation>& poses) const;
};

} // namespace vision_core

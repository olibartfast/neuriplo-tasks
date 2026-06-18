#pragma once

#include "neuriplo/tasks/core/task_interface.hpp"
#include "neuriplo/tasks/pose_estimation/pose_postprocessor.hpp"

#include <opencv2/core.hpp>
#include <vector>

namespace neuriplo_tasks {

/**
 * @brief RF-DETR keypoint pose postprocessor
 *
 * Decodes 3-tensor output from RF-DETR keypoint models:
 *   - dets:      [1, N, 4] — bbox centers (cx, cy, w, h)
 *   - labels:    [1, N, C] — class logits
 *   - keypoints: [1, N, C*K_max, 8] — per-keypoint 8-channel tensor
 *
 * 8 channels per keypoint: x, y, findability_logit, visibility_logit,
 * log_l11, l21, log_l22, class_boost.
 *
 * Per-keypoint decoding includes Cholesky→precision→covariance and
 * uncertainty-weighted score fusion.
 */
class RfDetrPosePostprocessor : public PosePostprocessor {
  public:
    RfDetrPosePostprocessor(const cv::Size& input_size, float confidence_threshold, float keypoint_uncertainty_alpha,
                            const std::vector<int>& keypoint_counts = {0, 17});

    std::vector<PoseEstimation> postprocess(const std::vector<Tensor>& tensors, const cv::Size& original_size,
                                            const cv::Size& input_size) override;

  private:
    cv::Size input_size_;
    float confidence_threshold_;
    float keypoint_uncertainty_alpha_;
    std::vector<int> keypoint_counts_;
};

} // namespace neuriplo_tasks

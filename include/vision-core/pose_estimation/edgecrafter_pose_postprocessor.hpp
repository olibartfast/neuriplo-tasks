#pragma once

#include "vision-core/pose_estimation/pose_postprocessor.hpp"

#include <string>
#include <vector>

namespace vision_core {

class EdgeCrafterPosePostprocessor : public PosePostprocessor {
  public:
    EdgeCrafterPosePostprocessor(float confidence_threshold, float keypoint_threshold,
                                 const std::vector<std::string>& output_names = {});

    std::vector<PoseEstimation> postprocess(const std::vector<Tensor>& tensors, const cv::Size& original_size,
                                            const cv::Size& input_size) override;

  private:
    float confidence_threshold_;
    float keypoint_threshold_;
    int scores_idx_{1};
    int keypoints_idx_{2};
    int labels_idx_{0};

    void findOutputIndices(const std::vector<std::string>& output_names);
    static float getTensorFloat(const TensorElement& element);
    static int getTensorInt(const TensorElement& element);
    void deriveBboxFromKeypoints(PoseEstimation& pose) const;
};

} // namespace vision_core

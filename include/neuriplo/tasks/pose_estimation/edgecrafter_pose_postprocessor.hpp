#pragma once

#include "neuriplo/tasks/pose_estimation/pose_postprocessor.hpp"

#include <string>
#include <vector>

namespace neuriplo_tasks {

class EdgeCrafterPosePostprocessor : public PosePostprocessor {
  public:
    EdgeCrafterPosePostprocessor(float confidence_threshold, float keypoint_threshold,
                                 const std::vector<std::string>& output_names = {});

    std::vector<PoseEstimation> postprocess(const std::vector<Tensor>& tensors, const Size& original_size,
                                            const Size& input_size) override;

  private:
    float confidence_threshold_;
    float keypoint_threshold_;
    int scores_idx_{1};
    int keypoints_idx_{2};
    int labels_idx_{0};

    void findOutputIndices(const std::vector<std::string>& output_names);
    void deriveBboxFromKeypoints(PoseEstimation& pose) const;
};

} // namespace neuriplo_tasks

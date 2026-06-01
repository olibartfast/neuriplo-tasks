#pragma once

#include "vision-core/object_detection/postprocessor.hpp"

#include <string>
#include <vector>

namespace vision_core {

class EdgeCrafterPostprocessor : public Postprocessor {
  public:
    EdgeCrafterPostprocessor(float confidence_threshold, const std::vector<std::string>& output_names = {});

    std::vector<Detection> postprocess(const std::vector<Tensor>& tensors, const cv::Size& frame_size) override;

  private:
    float confidence_threshold_;
    int scores_idx_{2};
    int boxes_idx_{1};
    int labels_idx_{0};

    void findOutputIndices(const std::vector<std::string>& output_names);
};

} // namespace vision_core

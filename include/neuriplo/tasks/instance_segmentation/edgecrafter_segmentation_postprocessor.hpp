#pragma once

#include "neuriplo/tasks/instance_segmentation/segmentation_postprocessor.hpp"

#include <string>
#include <vector>

namespace neuriplo_tasks {

class EdgeCrafterSegmentationPostprocessor : public SegmentationPostprocessor {
  public:
    EdgeCrafterSegmentationPostprocessor(float confidence_threshold, float mask_threshold,
                                         const std::vector<std::string>& output_names = {});

    std::vector<InstanceSegmentation> postprocess(const std::vector<Tensor>& tensors, const Size& frame_size) override;

  private:
    float confidence_threshold_;
    float mask_threshold_;
    int scores_idx_{2};
    int boxes_idx_{1};
    int labels_idx_{0};
    int masks_idx_{3};

    void findOutputIndices(const std::vector<std::string>& output_names);
};

} // namespace neuriplo_tasks

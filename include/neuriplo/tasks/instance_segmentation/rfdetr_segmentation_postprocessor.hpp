#pragma once

#include "neuriplo/tasks/instance_segmentation/segmentation_postprocessor.hpp"

namespace neuriplo_tasks {

class RfDetrSegmentationPostprocessor : public SegmentationPostprocessor {
  public:
    RfDetrSegmentationPostprocessor(const cv::Size& input_size, float confidence_threshold, float mask_threshold,
                                    const std::vector<std::string>& output_names = {});

    std::vector<InstanceSegmentation> postprocess(const std::vector<Tensor>& tensors,
                                                  const cv::Size& frame_size) override;

  private:
    cv::Size input_size_;
    float confidence_threshold_;
    float mask_threshold_;
    int boxes_idx_{0};  // Index for boxes output
    int masks_idx_{1};  // Index for masks output
    int labels_idx_{2}; // Index for labels output

    void findOutputIndices(const std::vector<std::string>& output_names);
};

} // namespace neuriplo_tasks

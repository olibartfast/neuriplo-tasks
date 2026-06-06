#pragma once

#include "neuriplo/tasks/object_detection/postprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief RF-DETR postprocessor
 *
 * RF-DETR outputs:
 * - dets: [1, 300, 4] - boxes in normalized cx,cy,w,h format
 * - labels: [1, 300, num_classes] - class logits (need sigmoid)
 */
class RfDetrPostprocessor : public Postprocessor {
  public:
    RfDetrPostprocessor(const cv::Size& input_size, float confidence_threshold,
                        const std::vector<std::string>& output_names = {});

    std::vector<Detection> postprocess(const std::vector<Tensor>& tensors, const cv::Size& frame_size) override;

  private:
    cv::Size input_size_;
    float confidence_threshold_;
    int dets_idx_{0};   // Index for dets (boxes) output
    int labels_idx_{1}; // Index for labels (logits) output

    void findOutputIndices(const std::vector<std::string>& output_names);
};

} // namespace neuriplo_tasks

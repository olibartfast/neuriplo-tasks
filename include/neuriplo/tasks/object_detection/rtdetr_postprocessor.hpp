#pragma once

#include "neuriplo/tasks/object_detection/object_detection_task.hpp"
#include "neuriplo/tasks/object_detection/postprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief RT-DETR/DEIM/DFINE postprocessor
 *
 * These models output 3 tensors:
 * - scores: [1, 300] - confidence per detection
 * - boxes: [1, 300, 4] - x1, y1, x2, y2 coordinates
 * - labels: [1, 300] - class ID per detection
 */
class RtDetrPostprocessor : public Postprocessor {
  public:
    RtDetrPostprocessor(ObjectDetectionTask::ModelType model_type, const cv::Size& input_size,
                        float confidence_threshold, const std::vector<std::string>& output_names = {});

    std::vector<Detection> postprocess(const std::vector<Tensor>& tensors, const cv::Size& frame_size) override;

  private:
    ObjectDetectionTask::ModelType model_type_;
    cv::Size input_size_;
    float confidence_threshold_;
    int scores_idx_{0};
    int boxes_idx_{1};
    int labels_idx_{2};

    void findOutputIndices(const std::vector<std::string>& output_names);

    // RT-DETR/DEIM/DFINE: 3 separate outputs
    std::vector<Detection> postprocessRTDETR(const Tensor& scores, const Tensor& boxes, const Tensor& labels,
                                             const cv::Size& frame_size);

    // RT-DETR Ultralytics: single combined output
    std::vector<Detection> postprocessRTDETRUL(const Tensor& output, const cv::Size& frame_size);
};

} // namespace neuriplo_tasks

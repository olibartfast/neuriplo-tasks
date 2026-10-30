#pragma once

#include "neuriplo/tasks/object_detection/object_detection_task.hpp"
#include "neuriplo/tasks/object_detection/postprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief RT-DETR/DEIM/DFINE postprocessor
 *
 * These models output 3 tensors, but the ordering depends on the export:
 * - ONNX exports keep the upstream (labels, boxes, scores) names, resolved
 *   by name.
 * - TorchScript keeps the traced tuple order with no names at all, so the
 *   order is recovered from shapes and dtypes.
 */
class RtDetrPostprocessor : public Postprocessor {
  public:
    RtDetrPostprocessor(ObjectDetectionTask::ModelType model_type, const Size& input_size, float confidence_threshold,
                        const std::vector<std::string>& output_names = {});

    std::vector<Detection> postprocess(const std::vector<Tensor>& tensors, const Size& frame_size) override;

  private:
    ObjectDetectionTask::ModelType model_type_;
    Size input_size_;
    float confidence_threshold_;
    int scores_idx_{0};
    int boxes_idx_{1};
    int labels_idx_{2};

    void findOutputIndices(const std::vector<std::string>& output_names);
    void resolveOutputIndices(const std::vector<Tensor>& tensors);

    // RT-DETR/DEIM/DFINE: 3 separate outputs
    std::vector<Detection> postprocessRTDETR(const Tensor& scores, const Tensor& boxes, const Tensor& labels,
                                             const Size& frame_size);

    // RT-DETR Ultralytics: single combined output
    std::vector<Detection> postprocessRTDETRUL(const Tensor& output, const Size& frame_size);
};

} // namespace neuriplo_tasks

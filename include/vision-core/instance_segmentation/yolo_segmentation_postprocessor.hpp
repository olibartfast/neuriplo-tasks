#pragma once

#include "vision-core/instance_segmentation/segmentation_postprocessor.hpp"

namespace vision_core {

class YoloSegmentationPostprocessor : public SegmentationPostprocessor {
public:
    YoloSegmentationPostprocessor(float confidence_threshold, float nms_threshold, float mask_threshold);

    std::vector<InstanceSegmentation> postprocess(
        const std::vector<Tensor>& tensors,
        const cv::Size& frame_size) override;

private:
    float confidence_threshold_;
    float nms_threshold_;
    float mask_threshold_;

    float getTensorFloat(const TensorElement& element);
};

} // namespace vision_core

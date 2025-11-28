#pragma once

#include "vision-core/instance_segmentation/segmentation_postprocessor.hpp"

namespace vision_core {

class RfDetrSegmentationPostprocessor : public SegmentationPostprocessor {
public:
    RfDetrSegmentationPostprocessor(float confidence_threshold, float mask_threshold);

    std::vector<InstanceSegmentation> postprocess(
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes,
        const cv::Size& frame_size) override;

private:
    float confidence_threshold_;
    float mask_threshold_;

    float getTensorFloat(const TensorElement& element);
};

} // namespace vision_core

#pragma once

#include "neuriplo/tasks/instance_segmentation/instance_segmentation_task.hpp"
#include "neuriplo/tasks/instance_segmentation/segmentation_postprocessor.hpp"

#include <vector>

namespace neuriplo_tasks {

class YoloSegmentationPostprocessor : public SegmentationPostprocessor {
  public:
    YoloSegmentationPostprocessor(InstanceSegmentationTask::ModelType model_type, const cv::Size& input_size,
                                  float confidence_threshold, float nms_threshold, float mask_threshold);

    std::vector<InstanceSegmentation> postprocess(const std::vector<Tensor>& tensors,
                                                  const cv::Size& frame_size) override;

  private:
    // Internal detection structure for NMS processing
    struct Detection {
        int class_id{0};
        float confidence{0.0f};
        float x1{0.0f};
        float y1{0.0f};
        float x2{0.0f};
        float y2{0.0f};
        std::vector<float> mask_coeffs;
    };

    // Member variables
    InstanceSegmentationTask::ModelType model_type_;
    cv::Size input_size_;
    float confidence_threshold_;
    float nms_threshold_;
    float mask_threshold_;

    // Model-specific postprocessing methods

    /// Standard YOLO segmentation (with NMS) - YOLOv8-seg, YOLOv11-seg
    /// Input tensors:
    ///   [0]: Detections [1, 4+cls+32, 8400] - boxes, class scores, mask coefficients
    ///   [1]: Mask Prototypes [1, 32, 160, 160]
    std::vector<InstanceSegmentation> postprocessYoloSeg(const std::vector<Tensor>& tensors,
                                                         const cv::Size& frame_size);

    /// NMS-free YOLO segmentation - YOLOv10-seg, YOLOv26-seg
    /// Input tensors:
    ///   [0]: Detections [1, 300, 38] - x1,y1,x2,y2, score, class, 32 coeffs
    ///   [1]: Mask Prototypes [1, 32, 160, 160]
    std::vector<InstanceSegmentation> postprocessYoloNmsFreeSeg(const std::vector<Tensor>& tensors,
                                                                const cv::Size& frame_size);

    // Helper methods

    /// Applies letterbox inverse transformation to convert coordinates
    /// from model input space to original image space
    [[nodiscard]] cv::Rect scaleToOriginal(float x1, float y1, float x2, float y2, const cv::Size& frame_size) const;

    /// Identifies indices for detection and prototype tensors based on dimensions
    /// @param tensors Input tensors
    /// @return Pair of indices {detection_idx, proto_idx}
    std::pair<int, int> findOutputIndices(const std::vector<Tensor>& tensors);

    /// Extracts float value from TensorElement variant

    /// Generates segmentation mask from coefficients and prototypes
    /// @param coeffs Mask coefficients (typically 32 values)
    /// @param protos_data Pointer to prototype masks data
    /// @param proto_h Height of prototype masks
    /// @param proto_w Width of prototype masks
    /// @return Generated mask at prototype resolution with sigmoid applied
    cv::Mat generateMask(const std::vector<float>& coeffs, const float* protos_data, int proto_h, int proto_w);

    /// Crops mask to bounding box region and resizes to bbox size
    /// @param mask Full prototype mask
    /// @param x1,y1,x2,y2 Bounding box coordinates in input space
    /// @param bbox Bounding box in original image space
    /// @param frame_size Original image size
    /// @return Full-frame binary mask with bbox region filled
    cv::Mat cropAndResizeMask(const cv::Mat& mask, float x1, float y1, float x2, float y2, const cv::Rect& bbox,
                              const cv::Size& frame_size);

    /// Applies Non-Maximum Suppression to filter overlapping detections
    /// @param detections Input detections to filter
    /// @return Filtered detections after NMS
    std::vector<Detection> applyNMS(const std::vector<Detection>& detections);

    /// Calculates Intersection over Union between two detections
    /// @param a First detection
    /// @param b Second detection
    /// @return IoU value [0.0, 1.0]
    float calculateIoU(const Detection& a, const Detection& b);
};

} // namespace neuriplo_tasks

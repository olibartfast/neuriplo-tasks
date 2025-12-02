#include "vision-core/instance_segmentation/rfdetr_segmentation_postprocessor.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace vision_core {

RfDetrSegmentationPostprocessor::RfDetrSegmentationPostprocessor(const cv::Size& input_size, 
                                                                 float confidence_threshold, 
                                                                 float mask_threshold,
                                                                 const std::vector<std::string>& output_names)
    : input_size_(input_size)
    , confidence_threshold_(confidence_threshold)
    , mask_threshold_(mask_threshold) 
{
    findOutputIndices(output_names);
}

void RfDetrSegmentationPostprocessor::findOutputIndices(const std::vector<std::string>& output_names) {
    // RF-DETR segmentation default: boxes at 0, masks at 1, labels at 2
    boxes_idx_ = 0;
    masks_idx_ = 1;
    labels_idx_ = 2;
    
    if (output_names.empty()) {
        return;
    }
    
    for (size_t i = 0; i < output_names.size(); ++i) {
        const auto& name = output_names[i];
        if (name == "boxes" || name == "dets") {
            boxes_idx_ = static_cast<int>(i);
        } else if (name == "masks" || name.find_first_of("0123456789") != std::string::npos) {
            // Masks can be identified by numeric name like "4245" or explicit "masks"
            masks_idx_ = static_cast<int>(i);
        } else if (name == "labels") {
            labels_idx_ = static_cast<int>(i);
        }
    }
}

std::vector<InstanceSegmentation> RfDetrSegmentationPostprocessor::postprocess(
    const std::vector<std::vector<TensorElement>>& infer_results,
    const std::vector<std::vector<int64_t>>& infer_shapes,
    const cv::Size& frame_size) {
    
    if (infer_results.size() < 3 || infer_shapes.size() < 3) {
        throw std::runtime_error("RF-DETR segmentation requires at least 3 output tensors");
    }

    const auto& boxes = infer_results[boxes_idx_];
    const auto& labels = infer_results[labels_idx_];
    const auto& masks = infer_results[masks_idx_];
    const auto& boxes_shape = infer_shapes[boxes_idx_];
    const auto& labels_shape = infer_shapes[labels_idx_];
    const auto& masks_shape = infer_shapes[masks_idx_];
    
    std::vector<InstanceSegmentation> segmentations;
    
    if (boxes_shape.size() < 3 || labels_shape.size() < 3 || masks_shape.size() < 4) {
        return {};
    }
    
    int num_dets = static_cast<int>(boxes_shape[1]);
    int num_classes = static_cast<int>(labels_shape[2]);
    int mask_h = static_cast<int>(masks_shape[2]);
    int mask_w = static_cast<int>(masks_shape[3]);
    
    
    // Scale factors from input to frame

    
    for (int i = 0; i < num_dets; ++i) {
        // Find max class score (apply sigmoid to logits)
        float max_score = 0.0f;
        int class_id = -1;
        
        for (int c = 0; c < num_classes; ++c) {
            float logit = getTensorFloat(labels[i * num_classes + c]);
            float score = 1.0f / (1.0f + std::exp(-logit)); // sigmoid
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }
        
        if (max_score < confidence_threshold_) {
            continue;
        }
        
        // RF-DETR uses 1-based class ids, adjust to 0-based
        if (class_id > 0) class_id -= 1;
        
        // Extract box (cx, cy, w, h) in normalized coordinates
        float cx = getTensorFloat(boxes[i * 4 + 0]);
        float cy = getTensorFloat(boxes[i * 4 + 1]);
        float w = getTensorFloat(boxes[i * 4 + 2]);
        float h = getTensorFloat(boxes[i * 4 + 3]);

        // Convert to frame coordinates
        float x_center = cx * frame_size.width;
        float y_center = cy * frame_size.height;
        float width = w * frame_size.width;
        float height = h * frame_size.height;

        float x_min = x_center - width / 2.0f;
        float y_min = y_center - height / 2.0f;
        
        InstanceSegmentation seg;
        seg.class_id = class_id;
        seg.class_confidence = max_score;
        seg.bbox = cv::Rect(static_cast<int>(x_min), static_cast<int>(y_min),
                           static_cast<int>(width), static_cast<int>(height));
        
        // Extract mask for this detection
        cv::Mat mask_logits(mask_h, mask_w, CV_32F);
        int mask_offset = i * mask_h * mask_w;
        float max_val = 0.0f, min_val = 1.0f;
        for (int y = 0; y < mask_h; ++y) {
            for (int x = 0; x < mask_w; ++x) {
                float logit = getTensorFloat(masks[mask_offset + y * mask_w + x]);
                float val = 1.0f / (1.0f + std::exp(-logit)); // sigmoid
                mask_logits.at<float>(y, x) = val;
                max_val = std::max(max_val, val);
                min_val = std::min(min_val, val);
            }
        }
        
        // Resize mask to frame size
        cv::Mat mask_resized;
        cv::resize(mask_logits, mask_resized, frame_size, 0, 0, cv::INTER_LINEAR);
        
        // Threshold to binary
        cv::Mat mask_binary;
        cv::threshold(mask_resized, mask_binary, mask_threshold_, 1.0, cv::THRESH_BINARY);
        mask_binary.convertTo(seg.mask, CV_8UC1, 255.0);
        
        // Check mask content
        
        // Crop mask to bbox
        cv::Mat mask_cropped = cv::Mat::zeros(frame_size, CV_8UC1);
        cv::Rect roi = seg.bbox & cv::Rect(0, 0, frame_size.width, frame_size.height);
        if (roi.width > 0 && roi.height > 0) {
            seg.mask(roi).copyTo(mask_cropped(roi));
        }
        seg.mask = mask_cropped;
        
        // Populate mask_data for visualization
        seg.mask_height = frame_size.height;
        seg.mask_width = frame_size.width;
        seg.mask_data.assign(seg.mask.datastart, seg.mask.dataend);
        
        
        segmentations.push_back(seg);
    }
    
    return segmentations;
}

float RfDetrSegmentationPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

int RfDetrSegmentationPostprocessor::getTensorInt(const TensorElement& element) {
    return std::visit([](auto&& value) -> int {
        return static_cast<int>(value);
    }, element);
}

} // namespace vision_core

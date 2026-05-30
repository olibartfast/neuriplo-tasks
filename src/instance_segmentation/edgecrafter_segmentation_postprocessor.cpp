#include "vision-core/instance_segmentation/edgecrafter_segmentation_postprocessor.hpp"

#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace vision_core {

EdgeCrafterSegmentationPostprocessor::EdgeCrafterSegmentationPostprocessor(float confidence_threshold,
                                                                           float mask_threshold,
                                                                           const std::vector<std::string>& output_names)
    : confidence_threshold_(confidence_threshold), mask_threshold_(mask_threshold) {
    findOutputIndices(output_names);
}

void EdgeCrafterSegmentationPostprocessor::findOutputIndices(const std::vector<std::string>& output_names) {
    if (output_names.empty()) {
        return;
    }

    for (size_t i = 0; i < output_names.size(); ++i) {
        const auto& name = output_names[i];
        if (name == "scores") {
            scores_idx_ = static_cast<int>(i);
        } else if (name == "boxes") {
            boxes_idx_ = static_cast<int>(i);
        } else if (name == "labels") {
            labels_idx_ = static_cast<int>(i);
        } else if (name == "masks") {
            masks_idx_ = static_cast<int>(i);
        }
    }
}

std::vector<InstanceSegmentation> EdgeCrafterSegmentationPostprocessor::postprocess(const std::vector<Tensor>& tensors,
                                                                                    const cv::Size& frame_size) {

    if (tensors.size() < 4) {
        throw std::runtime_error("EdgeCrafter segmentation requires 4 output tensors (labels, boxes, scores, masks)");
    }

    const auto& scores_tensor = tensors[static_cast<size_t>(scores_idx_)];
    const auto& boxes_tensor = tensors[static_cast<size_t>(boxes_idx_)];
    const auto& labels_tensor = tensors[static_cast<size_t>(labels_idx_)];
    const auto& masks_tensor = tensors[static_cast<size_t>(masks_idx_)];

    if (scores_tensor.shape.size() < 2 || boxes_tensor.shape.size() < 3 || labels_tensor.shape.size() < 2 ||
        masks_tensor.shape.size() < 4) {
        return {};
    }

    int num_dets = static_cast<int>(scores_tensor.shape[1]);
    int mask_h = static_cast<int>(masks_tensor.shape[2]);
    int mask_w = static_cast<int>(masks_tensor.shape[3]);

    std::vector<InstanceSegmentation> segmentations;
    segmentations.reserve(static_cast<size_t>(num_dets));

    for (int i = 0; i < num_dets; ++i) {
        float score = getTensorFloat(scores_tensor.data[static_cast<size_t>(i)]);

        if (score < confidence_threshold_) {
            continue;
        }

        int class_id = getTensorInt(labels_tensor.data[static_cast<size_t>(i)]);
        if (class_id < 0) {
            continue;
        }

        float x1 = getTensorFloat(boxes_tensor.data[static_cast<size_t>(i * 4 + 0)]);
        float y1 = getTensorFloat(boxes_tensor.data[static_cast<size_t>(i * 4 + 1)]);
        float x2 = getTensorFloat(boxes_tensor.data[static_cast<size_t>(i * 4 + 2)]);
        float y2 = getTensorFloat(boxes_tensor.data[static_cast<size_t>(i * 4 + 3)]);

        int ix1 = std::max(0, static_cast<int>(x1));
        int iy1 = std::max(0, static_cast<int>(y1));
        int ix2 = std::min(frame_size.width, static_cast<int>(x2));
        int iy2 = std::min(frame_size.height, static_cast<int>(y2));

        cv::Rect bbox(ix1, iy1, std::max(1, ix2 - ix1), std::max(1, iy2 - iy1));

        const size_t mask_offset = static_cast<size_t>(i) * static_cast<size_t>(mask_h) * static_cast<size_t>(mask_w);

        cv::Mat mask_small(mask_h, mask_w, CV_32F);
        for (int h = 0; h < mask_h; ++h) {
            for (int w = 0; w < mask_w; ++w) {
                mask_small.at<float>(h, w) =
                    getTensorFloat(masks_tensor.data[mask_offset + static_cast<size_t>(h * mask_w + w)]);
            }
        }

        cv::Mat mask_resized;
        cv::resize(mask_small, mask_resized, frame_size, 0, 0, cv::INTER_LINEAR);

        cv::Mat mask_binary;
        cv::threshold(mask_resized, mask_binary, mask_threshold_, 255, cv::THRESH_BINARY);
        mask_binary.convertTo(mask_binary, CV_8UC1);

        cv::Mat mask_full = cv::Mat::zeros(frame_size, CV_8UC1);
        cv::Rect clamped_bbox = bbox;
        clamped_bbox.x = std::max(0, std::min(bbox.x, frame_size.width - 1));
        clamped_bbox.y = std::max(0, std::min(bbox.y, frame_size.height - 1));
        clamped_bbox.width = std::max(1, std::min(bbox.width, frame_size.width - clamped_bbox.x));
        clamped_bbox.height = std::max(1, std::min(bbox.height, frame_size.height - clamped_bbox.y));

        cv::Mat mask_cropped = mask_binary(clamped_bbox);
        mask_cropped.copyTo(mask_full(clamped_bbox));

        InstanceSegmentation seg;
        seg.class_id = static_cast<float>(class_id);
        seg.class_confidence = score;
        seg.bbox = clamped_bbox;
        seg.mask = mask_full;
        seg.mask_height = mask_full.rows;
        seg.mask_width = mask_full.cols;
        segmentations.push_back(std::move(seg));
    }

    return segmentations;
}

float EdgeCrafterSegmentationPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float { return static_cast<float>(value); }, element);
}

int EdgeCrafterSegmentationPostprocessor::getTensorInt(const TensorElement& element) {
    return std::visit([](auto&& value) -> int { return static_cast<int>(value); }, element);
}

} // namespace vision_core

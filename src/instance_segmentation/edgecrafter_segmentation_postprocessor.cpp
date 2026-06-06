#include "vision-core/instance_segmentation/edgecrafter_segmentation_postprocessor.hpp"

#include "vision-core/core/opencv_interop.hpp"
#include "vision-core/core/output_name_utils.hpp"
#include "vision-core/core/tensor_utils.hpp"

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
    scores_idx_ = findOutputIndexByName(output_names, "scores", scores_idx_);
    boxes_idx_ = findOutputIndexByName(output_names, "boxes", boxes_idx_);
    labels_idx_ = findOutputIndexByName(output_names, "labels", labels_idx_);
    masks_idx_ = findOutputIndexByName(output_names, "masks", masks_idx_);
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
        float score = tensorElementToFloat(scores_tensor.data[static_cast<size_t>(i)]);

        if (score < confidence_threshold_) {
            continue;
        }

        int class_id = tensorElementToInt(labels_tensor.data[static_cast<size_t>(i)]);
        if (class_id < 0) {
            continue;
        }

        const size_t box_offset = static_cast<size_t>(i) * 4U;
        float x1 = tensorElementToFloat(boxes_tensor.data[box_offset + 0U]);
        float y1 = tensorElementToFloat(boxes_tensor.data[box_offset + 1U]);
        float x2 = tensorElementToFloat(boxes_tensor.data[box_offset + 2U]);
        float y2 = tensorElementToFloat(boxes_tensor.data[box_offset + 3U]);

        int ix1 = std::max(0, static_cast<int>(x1));
        int iy1 = std::max(0, static_cast<int>(y1));
        int ix2 = std::min(frame_size.width, static_cast<int>(x2));
        int iy2 = std::min(frame_size.height, static_cast<int>(y2));

        cv::Rect bbox(ix1, iy1, std::max(1, ix2 - ix1), std::max(1, iy2 - iy1));

        const size_t mask_offset = static_cast<size_t>(i) * static_cast<size_t>(mask_h) * static_cast<size_t>(mask_w);

        cv::Mat mask_small(mask_h, mask_w, CV_32F);
        for (int h = 0; h < mask_h; ++h) {
            for (int w = 0; w < mask_w; ++w) {
                const size_t mask_index =
                    mask_offset + static_cast<size_t>(h) * static_cast<size_t>(mask_w) + static_cast<size_t>(w);
                mask_small.at<float>(h, w) = tensorElementToFloat(masks_tensor.data[mask_index]);
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
        seg.bbox = fromCvRect(clamped_bbox);
        seg.mask = fromCvMat(mask_full);
        seg.mask_height = mask_full.rows;
        seg.mask_width = mask_full.cols;
        segmentations.push_back(std::move(seg));
    }

    return segmentations;
}

} // namespace vision_core

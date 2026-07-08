#include "neuriplo/tasks/instance_segmentation/edgecrafter_segmentation_postprocessor.hpp"

#include "image_ops.hpp"
#include "neuriplo/tasks/core/output_name_utils.hpp"
#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <stdexcept>

namespace neuriplo_tasks {

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
                                                                                    const vision::Size& frame_size) {

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

    const int batch = static_cast<int>(scores_tensor.shape[0]);
    const int num_dets = static_cast<int>(scores_tensor.shape[1]);
    const int mask_h = static_cast<int>(masks_tensor.shape[2]);
    const int mask_w = static_cast<int>(masks_tensor.shape[3]);

    const size_t batch_scores_stride = static_cast<size_t>(num_dets);
    const size_t batch_boxes_stride = static_cast<size_t>(num_dets) * 4;
    const size_t batch_labels_stride = static_cast<size_t>(num_dets);
    const size_t batch_masks_stride =
        static_cast<size_t>(num_dets) * static_cast<size_t>(mask_h) * static_cast<size_t>(mask_w);

    std::vector<InstanceSegmentation> segmentations;
    segmentations.reserve(static_cast<size_t>(batch * num_dets));

    for (int b = 0; b < batch; ++b) {
        const size_t scores_batch_offset = static_cast<size_t>(b) * batch_scores_stride;
        const size_t boxes_batch_offset = static_cast<size_t>(b) * batch_boxes_stride;
        const size_t labels_batch_offset = static_cast<size_t>(b) * batch_labels_stride;
        const size_t masks_batch_offset = static_cast<size_t>(b) * batch_masks_stride;

        for (int i = 0; i < num_dets; ++i) {
            float score = tensorElementToFloat(scores_tensor.data[scores_batch_offset + static_cast<size_t>(i)]);

            if (score < confidence_threshold_) {
                continue;
            }

            int class_id = tensorElementToInt(labels_tensor.data[labels_batch_offset + static_cast<size_t>(i)]);
            if (class_id < 0) {
                continue;
            }

            const size_t box_offset = boxes_batch_offset + static_cast<size_t>(i) * 4U;
            float x1 = tensorElementToFloat(boxes_tensor.data[box_offset + 0U]);
            float y1 = tensorElementToFloat(boxes_tensor.data[box_offset + 1U]);
            float x2 = tensorElementToFloat(boxes_tensor.data[box_offset + 2U]);
            float y2 = tensorElementToFloat(boxes_tensor.data[box_offset + 3U]);

            int ix1 = std::max(0, static_cast<int>(x1));
            int iy1 = std::max(0, static_cast<int>(y1));
            int ix2 = std::min(frame_size.width, static_cast<int>(x2));
            int iy2 = std::min(frame_size.height, static_cast<int>(y2));

            vision::Rect bbox(ix1, iy1, std::max(1, ix2 - ix1), std::max(1, iy2 - iy1));

            const size_t mask_offset =
                masks_batch_offset + static_cast<size_t>(i) * static_cast<size_t>(mask_h) * static_cast<size_t>(mask_w);

            vision::Image mask_small = vision::Image::uninit(mask_w, mask_h, 1, vision::PixelType::Float32);
            for (int h = 0; h < mask_h; ++h) {
                for (int w = 0; w < mask_w; ++w) {
                    const size_t mask_index =
                        mask_offset + static_cast<size_t>(h) * static_cast<size_t>(mask_w) + static_cast<size_t>(w);
                    mask_small.ptr<float>(h)[w] = tensorElementToFloat(masks_tensor.data[mask_index]);
                }
            }

            vision::Image mask_resized =
                image_ops::resize(mask_small, frame_size.width, frame_size.height, image_ops::Interpolation::Linear);

            vision::Image mask_binary = image_ops::thresholdBinary(mask_resized.view(), mask_threshold_, 255.0);
            mask_binary.convertTo(vision::PixelType::UInt8);

            vision::Rect clamped_bbox = bbox;
            clamped_bbox.x = std::max(0, std::min(bbox.x, frame_size.width - 1));
            clamped_bbox.y = std::max(0, std::min(bbox.y, frame_size.height - 1));
            clamped_bbox.width = std::max(1, std::min(bbox.width, frame_size.width - clamped_bbox.x));
            clamped_bbox.height = std::max(1, std::min(bbox.height, frame_size.height - clamped_bbox.y));

            vision::Image mask_full =
                vision::Image::zeros(frame_size.width, frame_size.height, 1, vision::PixelType::UInt8);
            image_ops::copyRegion(mask_binary.view(), clamped_bbox, mask_full, clamped_bbox);

            InstanceSegmentation seg;
            seg.class_id = static_cast<float>(class_id);
            seg.class_confidence = score;
            seg.bbox = clamped_bbox;
            seg.mask_height = mask_full.height();
            seg.mask_width = mask_full.width();
            seg.mask = fromImage(std::move(mask_full));
            segmentations.push_back(std::move(seg));
        }
    }

    return segmentations;
}

} // namespace neuriplo_tasks

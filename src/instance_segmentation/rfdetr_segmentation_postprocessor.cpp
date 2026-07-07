#include "neuriplo/tasks/instance_segmentation/rfdetr_segmentation_postprocessor.hpp"

#include "image_ops.hpp"
#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace neuriplo_tasks {

RfDetrSegmentationPostprocessor::RfDetrSegmentationPostprocessor(const Size& input_size, float confidence_threshold,
                                                                 float mask_threshold,
                                                                 const std::vector<std::string>& output_names)
    : input_size_(input_size), confidence_threshold_(confidence_threshold), mask_threshold_(mask_threshold) {
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

std::vector<InstanceSegmentation> RfDetrSegmentationPostprocessor::postprocess(const std::vector<Tensor>& tensors,
                                                                               const Size& frame_size) {

    if (tensors.size() < 3) {
        throw std::runtime_error("RF-DETR segmentation requires at least 3 output tensors");
    }

    const auto& boxes = tensors[static_cast<size_t>(boxes_idx_)].data;
    const auto& labels = tensors[static_cast<size_t>(labels_idx_)].data;
    const auto& masks = tensors[static_cast<size_t>(masks_idx_)].data;
    const auto& boxes_shape = tensors[static_cast<size_t>(boxes_idx_)].shape;
    const auto& labels_shape = tensors[static_cast<size_t>(labels_idx_)].shape;
    const auto& masks_shape = tensors[static_cast<size_t>(masks_idx_)].shape;

    std::vector<InstanceSegmentation> segmentations;

    if (boxes_shape.size() < 3 || labels_shape.size() < 3 || masks_shape.size() < 4) {
        return {};
    }

    const int batch = static_cast<int>(boxes_shape[0]);
    const int num_dets = static_cast<int>(boxes_shape[1]);
    const int num_classes = static_cast<int>(labels_shape[2]);
    const int mask_h = static_cast<int>(masks_shape[2]);
    const int mask_w = static_cast<int>(masks_shape[3]);

    const size_t box_batch_stride = static_cast<size_t>(num_dets) * 4;
    const size_t labels_batch_stride = static_cast<size_t>(num_dets) * static_cast<size_t>(num_classes);
    const size_t masks_batch_stride =
        static_cast<size_t>(num_dets) * static_cast<size_t>(mask_h) * static_cast<size_t>(mask_w);

    for (int b = 0; b < batch; ++b) {
        const size_t box_batch_offset = static_cast<size_t>(b) * box_batch_stride;
        const size_t labels_batch_offset = static_cast<size_t>(b) * labels_batch_stride;
        const size_t masks_batch_offset = static_cast<size_t>(b) * masks_batch_stride;

        for (int i = 0; i < num_dets; ++i) {
            float max_score = 0.0f;
            int class_id = -1;

            for (int c = 0; c < num_classes; ++c) {
                float logit =
                    tensorElementToFloat(labels[labels_batch_offset + static_cast<size_t>(i * num_classes + c)]);
                float score = 1.0f / (1.0f + std::exp(-logit));
                if (score > max_score) {
                    max_score = score;
                    class_id = c;
                }
            }

            if (max_score < confidence_threshold_) {
                continue;
            }

            if (class_id > 0) {
                class_id -= 1;
            }

            float cx = tensorElementToFloat(boxes[box_batch_offset + static_cast<size_t>(i * 4 + 0)]);
            float cy = tensorElementToFloat(boxes[box_batch_offset + static_cast<size_t>(i * 4 + 1)]);
            float w = tensorElementToFloat(boxes[box_batch_offset + static_cast<size_t>(i * 4 + 2)]);
            float h = tensorElementToFloat(boxes[box_batch_offset + static_cast<size_t>(i * 4 + 3)]);

            float x_center = cx * static_cast<float>(frame_size.width);
            float y_center = cy * static_cast<float>(frame_size.height);
            float width = w * static_cast<float>(frame_size.width);
            float height = h * static_cast<float>(frame_size.height);

            float x_min = x_center - width / 2.0f;
            float y_min = y_center - height / 2.0f;

            InstanceSegmentation seg;
            seg.class_id = static_cast<float>(class_id);
            seg.class_confidence = max_score;
            seg.bbox = BoundingBox(static_cast<int>(x_min), static_cast<int>(y_min), static_cast<int>(width),
                                   static_cast<int>(height));

            Image mask_logits = Image::uninit(mask_w, mask_h, 1, PixelType::Float32);
            size_t mask_offset =
                masks_batch_offset + static_cast<size_t>(i) * static_cast<size_t>(mask_h) * static_cast<size_t>(mask_w);
            float max_val = 0.0f;
            float min_val = 1.0f;
            for (int y = 0; y < mask_h; ++y) {
                for (int x = 0; x < mask_w; ++x) {
                    float logit = tensorElementToFloat(masks[mask_offset + static_cast<size_t>(y * mask_w + x)]);
                    float val = 1.0f / (1.0f + std::exp(-logit));
                    mask_logits.ptr<float>(y)[x] = val;
                    max_val = std::max(max_val, val);
                    min_val = std::min(min_val, val);
                }
            }

            Image mask_resized =
                image_ops::resize(mask_logits, frame_size.width, frame_size.height, image_ops::Interpolation::Linear);

            Image mask_binary = image_ops::thresholdBinary(mask_resized.view(), mask_threshold_, 1.0);
            Image mask_uint8 = mask_binary.convertedTo(PixelType::UInt8, 255.0);

            Image mask_cropped = Image::zeros(frame_size.width, frame_size.height, 1, PixelType::UInt8);
            const BoundingBox frame_bounds(0, 0, frame_size.width, frame_size.height);
            const BoundingBox roi_box = seg.bbox.intersect(frame_bounds);
            if (roi_box.width > 0 && roi_box.height > 0) {
                image_ops::copyRegion(mask_uint8.view(), roi_box, mask_cropped, roi_box);
            }

            seg.mask_height = frame_size.height;
            seg.mask_width = frame_size.width;
            seg.mask_data.assign(mask_uint8.raw(), mask_uint8.raw() + mask_uint8.sizeBytes());
            seg.mask = fromImage(std::move(mask_cropped));

            segmentations.push_back(seg);
        }
    }

    return segmentations;
}

} // namespace neuriplo_tasks

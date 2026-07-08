#include "neuriplo/tasks/instance_segmentation/yolo_segmentation_postprocessor.hpp"

#include "image_ops.hpp"
#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace neuriplo_tasks {

YoloSegmentationPostprocessor::YoloSegmentationPostprocessor(InstanceSegmentationTask::ModelType model_type,
                                                             const vision::Size& input_size, float confidence_threshold,
                                                             float nms_threshold, float mask_threshold)
    : model_type_(model_type), input_size_(input_size), confidence_threshold_(confidence_threshold),
      nms_threshold_(nms_threshold), mask_threshold_(mask_threshold) {}

std::vector<InstanceSegmentation> YoloSegmentationPostprocessor::postprocess(const std::vector<Tensor>& tensors,
                                                                             const vision::Size& frame_size) {
    switch (model_type_) {
    case InstanceSegmentationTask::ModelType::YOLO_V10_SEG:
    case InstanceSegmentationTask::ModelType::YOLO_26_SEG:
        return postprocessYoloNmsFreeSeg(tensors, frame_size);

    case InstanceSegmentationTask::ModelType::YOLO_SEG:
    default:
        return postprocessYoloSeg(tensors, frame_size);
    }
}

std::vector<InstanceSegmentation> YoloSegmentationPostprocessor::postprocessYoloSeg(const std::vector<Tensor>& tensors,
                                                                                    const vision::Size& frame_size) {

    if (tensors.size() < 2) {
        throw std::runtime_error("YOLO segmentation requires at least 2 output tensors");
    }

    auto indices = findOutputIndices(tensors);
    int det_idx = indices.first;
    int proto_idx = indices.second;

    const auto& dets_data = tensors[static_cast<size_t>(det_idx)].data;
    const auto& protos_data_raw = tensors[static_cast<size_t>(proto_idx)].data;
    const auto& dets_shape = tensors[static_cast<size_t>(det_idx)].shape;
    const auto& protos_shape = tensors[static_cast<size_t>(proto_idx)].shape;

    if (dets_shape.size() < 3 || protos_shape.size() < 4) {
        return {};
    }

    const int batch = static_cast<int>(dets_shape[0]);
    const int channels = static_cast<int>(dets_shape[1]);
    const int anchors = static_cast<int>(dets_shape[2]);
    const size_t batch_stride = static_cast<size_t>(channels) * static_cast<size_t>(anchors);
    int num_mask_coeffs = 32;
    int num_classes = channels - 4 - num_mask_coeffs;

    if (num_classes <= 0) {
        return {};
    }

    int proto_h = static_cast<int>(protos_shape[2]);
    int proto_w = static_cast<int>(protos_shape[3]);

    std::vector<float> protos_data(protos_data_raw.size());
    for (size_t i = 0; i < protos_data.size(); ++i) {
        protos_data[i] = tensorElementToFloat(protos_data_raw[i]);
    }

    const int num_protos = static_cast<int>(protos_shape[1]);
    const size_t proto_batch_stride =
        static_cast<size_t>(num_protos) * static_cast<size_t>(proto_h) * static_cast<size_t>(proto_w);

    std::vector<InstanceSegmentation> segmentations;

    for (int b = 0; b < batch; ++b) {
        const size_t batch_offset = static_cast<size_t>(b) * batch_stride;
        const size_t proto_offset = static_cast<size_t>(b) * proto_batch_stride;

        std::vector<Detection> detections;

        for (int i = 0; i < anchors; ++i) {
            float max_score = 0.0f;
            int class_id = -1;

            for (int c = 0; c < num_classes; ++c) {
                float score =
                    tensorElementToFloat(dets_data[batch_offset + static_cast<size_t>((c + 4) * anchors + i)]);
                if (score > max_score) {
                    max_score = score;
                    class_id = c;
                }
            }

            if (max_score < confidence_threshold_) {
                continue;
            }

            float cx = tensorElementToFloat(dets_data[batch_offset + static_cast<size_t>(0 * anchors + i)]);
            float cy = tensorElementToFloat(dets_data[batch_offset + static_cast<size_t>(1 * anchors + i)]);
            float w = tensorElementToFloat(dets_data[batch_offset + static_cast<size_t>(2 * anchors + i)]);
            float h = tensorElementToFloat(dets_data[batch_offset + static_cast<size_t>(3 * anchors + i)]);

            float x1 = cx - w / 2.0f;
            float y1 = cy - h / 2.0f;
            float x2 = cx + w / 2.0f;
            float y2 = cy + h / 2.0f;

            std::vector<float> mask_coeffs;
            mask_coeffs.reserve(static_cast<size_t>(num_mask_coeffs));
            for (int m = 0; m < num_mask_coeffs; ++m) {
                mask_coeffs.push_back(tensorElementToFloat(
                    dets_data[batch_offset + static_cast<size_t>((4 + num_classes + m) * anchors + i)]));
            }

            Detection det;
            det.class_id = class_id;
            det.confidence = max_score;
            det.x1 = x1;
            det.y1 = y1;
            det.x2 = x2;
            det.y2 = y2;
            det.mask_coeffs = mask_coeffs;

            detections.push_back(det);
        }

        std::vector<Detection> nms_detections = applyNMS(detections);

        for (const auto& det : nms_detections) {
            vision::Image mask = generateMask(det.mask_coeffs, protos_data.data() + proto_offset, proto_h, proto_w);

            vision::Rect bbox = scaleToOriginal(det.x1, det.y1, det.x2, det.y2, frame_size);

            vision::Image final_mask = cropAndResizeMask(mask.view(), det.x1, det.y1, det.x2, det.y2, bbox, frame_size);

            double max_val = 0.0;
            double dummy_min = 0.0;
            image_ops::minMax(final_mask.view(), dummy_min, max_val);
            if (max_val <= 0.0) {
                continue;
            }

            InstanceSegmentation seg;
            seg.class_id = static_cast<float>(det.class_id);
            seg.class_confidence = det.confidence;
            seg.bbox = bbox;
            seg.mask_height = final_mask.height();
            seg.mask_width = final_mask.width();
            seg.mask = fromImage(std::move(final_mask));

            segmentations.push_back(seg);
        }
    }

    return segmentations;
}

std::vector<InstanceSegmentation>
YoloSegmentationPostprocessor::postprocessYoloNmsFreeSeg(const std::vector<Tensor>& tensors,
                                                         const vision::Size& frame_size) {

    if (tensors.size() < 2) {
        throw std::runtime_error("YOLO NMS-free segmentation requires 2 output tensors");
    }

    auto indices = findOutputIndices(tensors);
    int det_idx = indices.first;
    int proto_idx = indices.second;

    const auto& dets_tensor = tensors[static_cast<size_t>(det_idx)];
    const auto& protos_tensor = tensors[static_cast<size_t>(proto_idx)];
    const auto& dets_shape = dets_tensor.shape;
    const auto& protos_shape = protos_tensor.shape;

    if (dets_shape.size() < 3 || protos_shape.size() < 4) {
        return {};
    }

    const int batch = static_cast<int>(dets_shape[0]);
    const int num_dets = static_cast<int>(dets_shape[1]);
    const int det_dims = static_cast<int>(dets_shape[2]);
    const int num_protos = static_cast<int>(protos_shape[1]);
    const int proto_h = static_cast<int>(protos_shape[2]);
    const int proto_w = static_cast<int>(protos_shape[3]);

    if (det_dims < 38 || num_protos != 32) {
        return {};
    }

    const size_t dets_batch_stride = static_cast<size_t>(num_dets) * static_cast<size_t>(det_dims);
    const size_t protos_batch_stride =
        static_cast<size_t>(num_protos) * static_cast<size_t>(proto_h) * static_cast<size_t>(proto_w);

    std::vector<InstanceSegmentation> segmentations;

    std::vector<float> protos_data(static_cast<size_t>(batch * num_protos * proto_h * proto_w));
    for (size_t i = 0; i < protos_data.size(); ++i) {
        protos_data[i] = tensorElementToFloat(protos_tensor.data[i]);
    }

    for (int b = 0; b < batch; ++b) {
        const size_t dets_batch_offset = static_cast<size_t>(b) * dets_batch_stride;
        const size_t protos_batch_offset = static_cast<size_t>(b) * protos_batch_stride;

        for (int i = 0; i < num_dets; ++i) {
            const size_t base = dets_batch_offset + static_cast<size_t>(i * det_dims);
            float x1 = tensorElementToFloat(dets_tensor.data[base + 0]);
            float y1 = tensorElementToFloat(dets_tensor.data[base + 1]);
            float x2 = tensorElementToFloat(dets_tensor.data[base + 2]);
            float y2 = tensorElementToFloat(dets_tensor.data[base + 3]);
            float score = tensorElementToFloat(dets_tensor.data[base + 4]);
            int class_id = static_cast<int>(tensorElementToFloat(dets_tensor.data[base + 5]));

            if (score < confidence_threshold_) {
                continue;
            }

            std::vector<float> mask_coeffs;
            mask_coeffs.reserve(32);
            for (int c = 0; c < 32; ++c) {
                mask_coeffs.push_back(tensorElementToFloat(dets_tensor.data[base + 6 + static_cast<size_t>(c)]));
            }

            vision::Image mask_proto = vision::Image::zeros(proto_w, proto_h, 1, vision::PixelType::Float32);

            for (int h = 0; h < proto_h; ++h) {
                for (int w = 0; w < proto_w; ++w) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_protos; ++c) {
                        size_t pidx =
                            protos_batch_offset + static_cast<size_t>(c * proto_h * proto_w + h * proto_w + w);
                        sum += mask_coeffs[static_cast<size_t>(c)] * protos_data[pidx];
                    }
                    mask_proto.ptr<float>(h)[w] = 1.0f / (1.0f + std::exp(-sum));
                }
            }

            float scale_x = static_cast<float>(proto_w) / static_cast<float>(input_size_.width);
            float scale_y = static_cast<float>(proto_h) / static_cast<float>(input_size_.height);

            int proto_x1 = static_cast<int>(x1 * scale_x);
            int proto_y1 = static_cast<int>(y1 * scale_y);
            int proto_x2 = static_cast<int>(x2 * scale_x);
            int proto_y2 = static_cast<int>(y2 * scale_y);

            proto_x1 = std::max(0, std::min(proto_x1, proto_w - 1));
            proto_y1 = std::max(0, std::min(proto_y1, proto_h - 1));
            proto_x2 = std::max(proto_x1 + 1, std::min(proto_x2, proto_w));
            proto_y2 = std::max(proto_y1 + 1, std::min(proto_y2, proto_h));

            vision::Rect proto_bbox(proto_x1, proto_y1, proto_x2 - proto_x1, proto_y2 - proto_y1);

            if (proto_bbox.width <= 0 || proto_bbox.height <= 0) {
                continue;
            }

            vision::Image mask_cropped =
                vision::Image::uninit(proto_bbox.width, proto_bbox.height, 1, vision::PixelType::Float32);
            image_ops::copyRegion(mask_proto.view(), proto_bbox, mask_cropped,
                                  vision::Rect(0, 0, proto_bbox.width, proto_bbox.height));

            double max_val = 0.0;
            double dummy_min = 0.0;
            image_ops::minMax(mask_cropped.view(), dummy_min, max_val);
            if (max_val <= 0.0) {
                continue;
            }

            vision::Rect bbox = scaleToOriginal(x1, y1, x2, y2, frame_size);

            bbox.x = std::max(0, std::min(bbox.x, frame_size.width - 1));
            bbox.y = std::max(0, std::min(bbox.y, frame_size.height - 1));
            bbox.width = std::max(1, std::min(bbox.width, frame_size.width - bbox.x));
            bbox.height = std::max(1, std::min(bbox.height, frame_size.height - bbox.y));

            vision::Image mask_resized =
                image_ops::resize(mask_cropped, bbox.width, bbox.height, image_ops::Interpolation::Linear);

            vision::Image mask_binary = image_ops::thresholdBinary(mask_resized.view(), mask_threshold_, 255.0);
            mask_binary.convertTo(vision::PixelType::UInt8);

            vision::Image mask_full =
                vision::Image::zeros(frame_size.width, frame_size.height, 1, vision::PixelType::UInt8);
            image_ops::copyRegion(mask_binary.view(), vision::Rect(0, 0, bbox.width, bbox.height), mask_full, bbox);

            InstanceSegmentation seg;
            seg.class_id = static_cast<float>(class_id);
            seg.class_confidence = score;
            seg.bbox = bbox;
            seg.mask_height = mask_full.height();
            seg.mask_width = mask_full.width();
            seg.mask = fromImage(std::move(mask_full));

            segmentations.push_back(seg);
        }
    }

    return segmentations;
}

std::pair<int, int> YoloSegmentationPostprocessor::findOutputIndices(const std::vector<Tensor>& tensors) {
    int det_idx = -1;
    int proto_idx = -1;

    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].shape.size() == 4) {
            proto_idx = static_cast<int>(i);
        } else if (tensors[i].shape.size() == 3) {
            det_idx = static_cast<int>(i);
        }
    }

    if (det_idx == -1) {
        det_idx = 0;
    }
    if (proto_idx == -1) {
        proto_idx = 1;
    }

    return {det_idx, proto_idx};
}

vision::Rect YoloSegmentationPostprocessor::scaleToOriginal(float x1, float y1, float x2, float y2,
                                                            const vision::Size& frame_size) const {

    float r_w = static_cast<float>(input_size_.width) / static_cast<float>(frame_size.width);
    float r_h = static_cast<float>(input_size_.height) / static_cast<float>(frame_size.height);
    float r = std::min(r_w, r_h);

    float pad_w = (static_cast<float>(input_size_.width) - r * static_cast<float>(frame_size.width)) / 2.0f;
    float pad_h = (static_cast<float>(input_size_.height) - r * static_cast<float>(frame_size.height)) / 2.0f;

    float orig_x1 = (x1 - pad_w) / r;
    float orig_y1 = (y1 - pad_h) / r;
    float orig_x2 = (x2 - pad_w) / r;
    float orig_y2 = (y2 - pad_h) / r;

    orig_x1 = std::max(0.0f, std::min(orig_x1, static_cast<float>(frame_size.width)));
    orig_y1 = std::max(0.0f, std::min(orig_y1, static_cast<float>(frame_size.height)));
    orig_x2 = std::max(0.0f, std::min(orig_x2, static_cast<float>(frame_size.width)));
    orig_y2 = std::max(0.0f, std::min(orig_y2, static_cast<float>(frame_size.height)));

    return vision::Rect(static_cast<int>(orig_x1), static_cast<int>(orig_y1), static_cast<int>(orig_x2 - orig_x1),
                        static_cast<int>(orig_y2 - orig_y1));
}

vision::Image YoloSegmentationPostprocessor::generateMask(const std::vector<float>& coeffs, const float* protos_data,
                                                          int proto_h, int proto_w) {

    vision::Image mask = vision::Image::zeros(proto_w, proto_h, 1, vision::PixelType::Float32);

    for (int h = 0; h < proto_h; ++h) {
        for (int w = 0; w < proto_w; ++w) {
            float sum = 0.0f;
            for (size_t c = 0; c < coeffs.size(); ++c) {
                int pidx = static_cast<int>(c) * proto_h * proto_w + h * proto_w + w;
                sum += coeffs[c] * protos_data[pidx];
            }
            mask.ptr<float>(h)[w] = 1.0f / (1.0f + std::exp(-sum));
        }
    }

    return mask;
}

vision::Image YoloSegmentationPostprocessor::cropAndResizeMask(const vision::ImageView& mask, float x1, float y1,
                                                               float x2, float y2, const vision::Rect& bbox,
                                                               const vision::Size& frame_size) {

    int proto_h = mask.height();
    int proto_w = mask.width();

    float scale_x = static_cast<float>(proto_w) / static_cast<float>(input_size_.width);
    float scale_y = static_cast<float>(proto_h) / static_cast<float>(input_size_.height);

    int proto_x1 = static_cast<int>(x1 * scale_x);
    int proto_y1 = static_cast<int>(y1 * scale_y);
    int proto_x2 = static_cast<int>(x2 * scale_x);
    int proto_y2 = static_cast<int>(y2 * scale_y);

    proto_x1 = std::max(0, std::min(proto_x1, proto_w - 1));
    proto_y1 = std::max(0, std::min(proto_y1, proto_h - 1));
    proto_x2 = std::max(proto_x1 + 1, std::min(proto_x2, proto_w));
    proto_y2 = std::max(proto_y1 + 1, std::min(proto_y2, proto_h));

    vision::Rect proto_bbox(proto_x1, proto_y1, proto_x2 - proto_x1, proto_y2 - proto_y1);

    if (proto_bbox.width <= 0 || proto_bbox.height <= 0) {
        return vision::Image::zeros(frame_size.width, frame_size.height, 1, vision::PixelType::UInt8);
    }

    vision::Image mask_cropped =
        vision::Image::uninit(proto_bbox.width, proto_bbox.height, 1, vision::PixelType::Float32);
    image_ops::copyRegion(mask, proto_bbox, mask_cropped, vision::Rect(0, 0, proto_bbox.width, proto_bbox.height));

    if (bbox.width <= 0 || bbox.height <= 0) {
        return vision::Image::zeros(frame_size.width, frame_size.height, 1, vision::PixelType::UInt8);
    }

    vision::Image mask_resized =
        image_ops::resize(mask_cropped, bbox.width, bbox.height, image_ops::Interpolation::Linear);

    vision::Image mask_binary = image_ops::thresholdBinary(mask_resized.view(), mask_threshold_, 255.0);
    mask_binary.convertTo(vision::PixelType::UInt8);

    vision::Image mask_full = vision::Image::zeros(frame_size.width, frame_size.height, 1, vision::PixelType::UInt8);
    if (bbox.x >= 0 && bbox.y >= 0 && bbox.x + bbox.width <= frame_size.width &&
        bbox.y + bbox.height <= frame_size.height) {
        image_ops::copyRegion(mask_binary.view(), vision::Rect(0, 0, bbox.width, bbox.height), mask_full, bbox);
    }

    return mask_full;
}

std::vector<YoloSegmentationPostprocessor::Detection>
YoloSegmentationPostprocessor::applyNMS(const std::vector<Detection>& detections) {

    if (detections.empty()) {
        return {};
    }

    std::vector<Detection> sorted_dets = detections;
    std::sort(sorted_dets.begin(), sorted_dets.end(),
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(sorted_dets.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < sorted_dets.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        result.push_back(sorted_dets[i]);

        for (size_t j = i + 1; j < sorted_dets.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }

            if (sorted_dets[i].class_id != sorted_dets[j].class_id) {
                continue;
            }

            float iou = calculateIoU(sorted_dets[i], sorted_dets[j]);
            if (iou > nms_threshold_) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

float YoloSegmentationPostprocessor::calculateIoU(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection;

    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

} // namespace neuriplo_tasks

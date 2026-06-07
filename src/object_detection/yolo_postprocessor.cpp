#include "neuriplo/tasks/object_detection/yolo_postprocessor.hpp"

#include "neuriplo/tasks/core/opencv_interop.hpp"
#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace neuriplo_tasks {

YoloPostprocessor::YoloPostprocessor(ObjectDetectionTask::ModelType model_type, const cv::Size& input_size,
                                     float confidence_threshold, float nms_threshold)
    : model_type_(model_type), input_size_(input_size), confidence_threshold_(confidence_threshold),
      nms_threshold_(nms_threshold) {}

cv::Rect YoloPostprocessor::scaleToOriginal(float cx, float cy, float w, float h, const cv::Size& frame_size) const {
    // Apply letterbox inverse transformation
    // This converts coordinates from letterboxed model space to original frame space
    float r_w = static_cast<float>(input_size_.width) / static_cast<float>(frame_size.width);
    float r_h = static_cast<float>(input_size_.height) / static_cast<float>(frame_size.height);

    int x, y, width, height;

    if (r_h > r_w) {
        // Width is the limiting factor - padding is on top/bottom
        float pad_h = (static_cast<float>(input_size_.height) - r_w * static_cast<float>(frame_size.height)) / 2.0f;
        float x_min = cx - w / 2.0f;
        float x_max = cx + w / 2.0f;
        float y_min = cy - h / 2.0f - pad_h;
        float y_max = cy + h / 2.0f - pad_h;

        x = static_cast<int>(x_min / r_w);
        y = static_cast<int>(y_min / r_w);
        width = static_cast<int>((x_max - x_min) / r_w);
        height = static_cast<int>((y_max - y_min) / r_w);
    } else {
        // Height is the limiting factor - padding is on left/right
        float pad_w = (static_cast<float>(input_size_.width) - r_h * static_cast<float>(frame_size.width)) / 2.0f;
        float x_min = cx - w / 2.0f - pad_w;
        float x_max = cx + w / 2.0f - pad_w;
        float y_min = cy - h / 2.0f;
        float y_max = cy + h / 2.0f;

        x = static_cast<int>(x_min / r_h);
        y = static_cast<int>(y_min / r_h);
        width = static_cast<int>((x_max - x_min) / r_h);
        height = static_cast<int>((y_max - y_min) / r_h);
    }

    return cv::Rect(x, y, width, height);
}

cv::Rect YoloPostprocessor::scaleXyxyToOriginal(float x1, float y1, float x2, float y2,
                                                const cv::Size& frame_size) const {
    const float r_w = static_cast<float>(input_size_.width) / static_cast<float>(frame_size.width);
    const float r_h = static_cast<float>(input_size_.height) / static_cast<float>(frame_size.height);

    if (r_h > r_w) {
        const float pad_h =
            (static_cast<float>(input_size_.height) - r_w * static_cast<float>(frame_size.height)) / 2.0f;
        y1 = (y1 - pad_h) / r_w;
        y2 = (y2 - pad_h) / r_w;
        x1 = x1 / r_w;
        x2 = x2 / r_w;
    } else {
        const float pad_w = (static_cast<float>(input_size_.width) - r_h * static_cast<float>(frame_size.width)) / 2.0f;
        x1 = (x1 - pad_w) / r_h;
        x2 = (x2 - pad_w) / r_h;
        y1 = y1 / r_h;
        y2 = y2 / r_h;
    }

    return cv::Rect(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
}

std::vector<Detection> YoloPostprocessor::postprocess(const std::vector<Tensor>& tensors, const cv::Size& frame_size) {

    std::vector<Detection> detections;

    switch (model_type_) {
    case ObjectDetectionTask::ModelType::YOLO_STANDARD: {
        if (tensors.empty())
            return {};
        detections = postprocessYoloStandard(tensors[0].data, tensors[0].shape, frame_size);
        break;
    }

    case ObjectDetectionTask::ModelType::YOLO_V4: {
        if (tensors.empty())
            return {};
        detections = postprocessYoloV4(tensors, frame_size);
        break;
    }

    case ObjectDetectionTask::ModelType::YOLO_NMS_FREE: {
        if (tensors.empty())
            return {};
        detections = postprocessYoloNmsFree(tensors[0], frame_size);
        break;
    }

    case ObjectDetectionTask::ModelType::YOLO_NAS: {
        if (tensors.size() < 2) {
            throw std::runtime_error("YOLO-NAS requires 2 output tensors");
        }
        detections = postprocessYoloNAS(tensors[0], tensors[1], frame_size);
        break;
    }

    case ObjectDetectionTask::ModelType::YOLO_V7_E2E: {
        if (tensors.size() < 4) {
            throw std::runtime_error("YOLOv7 end-to-end requires 4 output tensors");
        }
        // Outputs: num_dets, det_boxes, det_scores, det_classes
        detections = postprocessYoloV7E2E(tensors[0], tensors[1], tensors[2], tensors[3], frame_size);
        break;
    }

    default:
        throw std::runtime_error("Unsupported YOLO model type");
    }

    return detections;
}

namespace {

std::vector<Detection> decodeYoloStandardBatchSlice(
    const std::vector<TensorElement>& output, const cv::Size& frame_size, int batch_index, float confidence_threshold,
    bool has_objectness, int channels, int anchors, int num_classes, int class_offset,
    const std::function<cv::Rect(float, float, float, float, const cv::Size&)>& scale_to_original) {
    std::vector<Detection> detections;

    const size_t slice_stride = static_cast<size_t>(channels) * static_cast<size_t>(anchors);
    const size_t base_offset = static_cast<size_t>(batch_index) * slice_stride;

    for (int i = 0; i < anchors; ++i) {
        float objectness = 1.0f;
        if (has_objectness) {
            objectness = tensorElementToFloat(output[base_offset + static_cast<size_t>(i * channels + 4)]);
            if (objectness < confidence_threshold) {
                continue;
            }
        }

        float max_class_score = 0.0f;
        int class_id = -1;

        for (int c = 0; c < num_classes; ++c) {
            float score;
            if (has_objectness) {
                score =
                    tensorElementToFloat(output[base_offset + static_cast<size_t>(i * channels + (c + class_offset))]);
            } else {
                score =
                    tensorElementToFloat(output[base_offset + static_cast<size_t>((c + class_offset) * anchors + i)]);
            }

            if (score > max_class_score) {
                max_class_score = score;
                class_id = c;
            }
        }

        const float final_score = has_objectness ? (objectness * max_class_score) : max_class_score;
        if (final_score < confidence_threshold) {
            continue;
        }

        float cx = 0.0f;
        float cy = 0.0f;
        float w = 0.0f;
        float h = 0.0f;
        if (has_objectness) {
            cx = tensorElementToFloat(output[base_offset + static_cast<size_t>(i * channels + 0)]);
            cy = tensorElementToFloat(output[base_offset + static_cast<size_t>(i * channels + 1)]);
            w = tensorElementToFloat(output[base_offset + static_cast<size_t>(i * channels + 2)]);
            h = tensorElementToFloat(output[base_offset + static_cast<size_t>(i * channels + 3)]);
        } else {
            cx = tensorElementToFloat(output[base_offset + static_cast<size_t>(0 * anchors + i)]);
            cy = tensorElementToFloat(output[base_offset + static_cast<size_t>(1 * anchors + i)]);
            w = tensorElementToFloat(output[base_offset + static_cast<size_t>(2 * anchors + i)]);
            h = tensorElementToFloat(output[base_offset + static_cast<size_t>(3 * anchors + i)]);
        }

        Detection det;
        det.class_id = static_cast<float>(class_id);
        det.class_confidence = final_score;
        det.bbox = fromCvRect(scale_to_original(cx, cy, w, h, frame_size));
        detections.push_back(det);
    }

    return detections;
}

} // namespace

std::vector<Detection> YoloPostprocessor::postprocessYoloStandard(const std::vector<TensorElement>& output,
                                                                  const std::vector<int64_t>& shape,
                                                                  const cv::Size& frame_size) {

    std::vector<Detection> detections;

    // Check shape dimensions
    if (shape.size() < 3) {
        return {};
    }

    const int batch = static_cast<int>(shape[0]);
    const int dim1 = static_cast<int>(shape[1]);
    const int dim2 = static_cast<int>(shape[2]);

    // Detect format by shape:
    // YOLOv5/v6/v7: [batch, anchors, 4+1+classes] where dim2 < dim1, has objectness at index 4
    // YOLOv8+:      [batch, 4+classes, anchors] where dim1 < dim2, no objectness
    bool has_objectness = (dim2 < dim1); // YOLOv5/v6/v7 format

    int channels, anchors;
    if (has_objectness) {
        // YOLOv5/v6/v7: [batch, anchors, channels]
        anchors = dim1;
        channels = dim2;
    } else {
        // YOLOv8+: [batch, channels, anchors]
        channels = dim1;
        anchors = dim2;
    }

    int num_classes = has_objectness ? (channels - 5) : (channels - 4);
    int class_offset = has_objectness ? 5 : 4;

    if (num_classes <= 0) {
        return {};
    }

    const auto scale_fn = [this](float cx, float cy, float w, float h, const cv::Size& fs) {
        return scaleToOriginal(cx, cy, w, h, fs);
    };

    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        std::vector<Detection> slice_detections =
            decodeYoloStandardBatchSlice(output, frame_size, batch_index, confidence_threshold_, has_objectness,
                                         channels, anchors, num_classes, class_offset, scale_fn);
        applyNMS(slice_detections);
        detections.insert(detections.end(), slice_detections.begin(), slice_detections.end());
    }

    return detections;
}

std::vector<Detection> YoloPostprocessor::postprocessYoloV4(const std::vector<Tensor>& tensors,
                                                            const cv::Size& frame_size) {
    std::vector<Detection> detections;

    // Iterate over the output tensors
    for (size_t i = 0; i < tensors.size(); ++i) {
        const TensorElement* output = tensors[i].data.data();

        // Iterate over the detections in the output tensor
        for (int j = 0; j < tensors[i].shape[0]; ++j, output += tensors[i].shape[1]) {
            // Find the class with the highest confidence
            // Find the class with the highest confidence
            auto maxSPtr = std::max_element(output + 5, output + tensors[i].shape[1],
                                            [](const TensorElement& a, const TensorElement& b) {
                                                return tensorElementToFloat(a) < tensorElementToFloat(b);
                                            });

            float score = tensorElementToFloat(*maxSPtr);

            // Check if the confidence is above the threshold
            if (score > confidence_threshold_) {
                float cx = tensorElementToFloat(output[0]);
                float cy = tensorElementToFloat(output[1]);
                float w = tensorElementToFloat(output[2]);
                float h = tensorElementToFloat(output[3]);

                int label = static_cast<int>(maxSPtr - (output + 5));

                // YOLOv4 outputs normalized coordinates [0,1] relative to the model
                // input Convert to model input pixel coordinates
                float cx_model = cx * static_cast<float>(input_size_.width);
                float cy_model = cy * static_cast<float>(input_size_.height);
                float w_model = w * static_cast<float>(input_size_.width);
                float h_model = h * static_cast<float>(input_size_.height);

                // Apply letterbox inverse transformation using model-space coordinates
                cv::Rect bbox = scaleToOriginal(cx_model, cy_model, w_model, h_model, frame_size);

                // Create a detection object
                Detection detection;
                detection.bbox = fromCvRect(bbox);
                detection.class_confidence = score;
                detection.class_id = static_cast<float>(label);

                // Add the detection to the vector
                detections.push_back(detection);
            }
        }
    }

    // Apply non-maximum suppression to the detections
    std::vector<Detection> filtered_detections;
    std::map<int, std::vector<size_t>> class2indices;
    for (size_t i = 0; i < detections.size(); i++) {
        if (detections[i].class_confidence >= confidence_threshold_) {
            class2indices[static_cast<int>(detections[i].class_id)].push_back(i);
        }
    }

    for (std::map<int, std::vector<size_t>>::iterator it = class2indices.begin(); it != class2indices.end(); ++it) {
        std::vector<cv::Rect> localBoxes;
        std::vector<float> localConfidences;
        std::vector<size_t> classIndices = it->second;
        for (size_t i = 0; i < classIndices.size(); i++) {
            localBoxes.push_back(toCvRect(detections[classIndices[i]].bbox));
            localConfidences.push_back(detections[classIndices[i]].class_confidence);
        }
        std::vector<int> nmsIndices;
        cv::dnn::NMSBoxes(localBoxes, localConfidences, confidence_threshold_, nms_threshold_, nmsIndices);
        for (size_t i = 0; i < nmsIndices.size(); i++) {
            filtered_detections.push_back(detections[classIndices[static_cast<size_t>(nmsIndices[i])]]);
        }
    }

    return filtered_detections;
}

std::vector<Detection> YoloPostprocessor::postprocessYoloNmsFree(const Tensor& output, const cv::Size& frame_size) {

    std::vector<Detection> detections;

    // YOLOv10/YOLO26 output: [1, 300, 6] (x1, y1, x2, y2, score, class)
    if (output.shape.size() < 3 || output.shape[2] < 6)
        return {};

    int num_dets = static_cast<int>(output.shape[1]);
    int dims = static_cast<int>(output.shape[2]);

    for (int i = 0; i < num_dets; ++i) {
        float score = tensorElementToFloat(output.data[static_cast<size_t>(i * dims + 4)]);
        if (score < confidence_threshold_)
            continue;

        float x1 = tensorElementToFloat(output.data[static_cast<size_t>(i * dims + 0)]);
        float y1 = tensorElementToFloat(output.data[static_cast<size_t>(i * dims + 1)]);
        float x2 = tensorElementToFloat(output.data[static_cast<size_t>(i * dims + 2)]);
        float y2 = tensorElementToFloat(output.data[static_cast<size_t>(i * dims + 3)]);
        int class_id = static_cast<int>(tensorElementToFloat(output.data[static_cast<size_t>(i * dims + 5)]));

        x1 *= static_cast<float>(input_size_.width);
        y1 *= static_cast<float>(input_size_.height);
        x2 *= static_cast<float>(input_size_.width);
        y2 *= static_cast<float>(input_size_.height);

        Detection det;
        det.class_id = static_cast<float>(class_id);
        det.class_confidence = score;
        det.bbox = fromCvRect(scaleXyxyToOriginal(x1, y1, x2, y2, frame_size));
        detections.push_back(det);
    }

    // End-to-end models (YOLOv10/YOLO26) typically don't need NMS, but we can apply it if needed
    applyNMS(detections);
    return detections;
}

std::vector<Detection> YoloPostprocessor::postprocessYoloNAS(const Tensor& boxes, const Tensor& scores,
                                                             const cv::Size& frame_size) {

    std::vector<Detection> detections;

    // Boxes: [1, N, 4], Scores: [1, N, C]
    if (boxes.shape.size() < 3 || scores.shape.size() < 3)
        return {};

    int num_dets = static_cast<int>(boxes.shape[1]);
    int num_classes = static_cast<int>(scores.shape[2]);

    for (int i = 0; i < num_dets; ++i) {
        // Find max score for this detection
        float max_score = 0.0f;
        int class_id = -1;

        for (int c = 0; c < num_classes; ++c) {
            float score = tensorElementToFloat(scores.data[static_cast<size_t>(i * num_classes + c)]);
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score < confidence_threshold_)
            continue;

        float x1 = tensorElementToFloat(boxes.data[static_cast<size_t>(i * 4 + 0)]);
        float y1 = tensorElementToFloat(boxes.data[static_cast<size_t>(i * 4 + 1)]);
        float x2 = tensorElementToFloat(boxes.data[static_cast<size_t>(i * 4 + 2)]);
        float y2 = tensorElementToFloat(boxes.data[static_cast<size_t>(i * 4 + 3)]);

        Detection det;
        det.class_id = static_cast<float>(class_id);
        det.class_confidence = max_score;
        det.bbox = fromCvRect(scaleXyxyToOriginal(x1, y1, x2, y2, frame_size));
        detections.push_back(det);
    }

    applyNMS(detections);
    return detections;
}

void YoloPostprocessor::applyNMS(std::vector<Detection>& detections) {
    if (detections.empty())
        return;

    std::vector<bool> suppress(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppress[i])
            continue;

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppress[j])
                continue;
            if (static_cast<int>(detections[i].class_id) != static_cast<int>(detections[j].class_id))
                continue;

            const BoundingBox intersection = detections[i].bbox.intersect(detections[j].bbox);
            float intersection_area = static_cast<float>(intersection.area());
            float union_area = static_cast<float>(detections[i].bbox.area()) +
                               static_cast<float>(detections[j].bbox.area()) - intersection_area;

            if (union_area > 0) {
                float iou = intersection_area / union_area;
                if (iou > nms_threshold_) {
                    if (detections[i].class_confidence > detections[j].class_confidence) {
                        suppress[j] = true;
                    } else {
                        suppress[i] = true;
                        break;
                    }
                }
            }
        }
    }

    detections.erase(std::remove_if(detections.begin(), detections.end(),
                                    [&](const Detection& det) {
                                        size_t idx = static_cast<size_t>(&det - &detections[0]);
                                        return suppress[idx];
                                    }),
                     detections.end());
}

std::vector<Detection> YoloPostprocessor::postprocessYoloV7E2E(const Tensor& num_dets_tensor, const Tensor& boxes,
                                                               const Tensor& scores, const Tensor& classes,
                                                               const cv::Size& frame_size) {

    std::vector<Detection> detections;

    // YOLOv7 end-to-end format:
    // num_dets: [1,1] - number of valid detections
    // boxes: [1, max_dets, 4] - bounding boxes in [x1, y1, x2, y2] format (model space)
    // scores: [1, max_dets] - confidence scores
    // classes: [1, max_dets] - class IDs

    if (num_dets_tensor.data.empty() || boxes.data.empty() || scores.data.empty() || classes.data.empty()) {
        return detections;
    }

    // Get actual number of detections
    int num_dets = static_cast<int>(tensorElementToFloat(num_dets_tensor.data[0]));

    // boxes.shape is [1, max_dets, 4]
    int max_dets = static_cast<int>(boxes.shape.size() >= 2 ? boxes.shape[1] : 100);
    num_dets = std::min(num_dets, max_dets); // Clamp to max_dets

    for (int i = 0; i < num_dets; ++i) {
        float score = tensorElementToFloat(scores.data[static_cast<size_t>(i)]);
        if (score < confidence_threshold_)
            continue;

        // Extract box coordinates in model space
        float x1 = tensorElementToFloat(boxes.data[static_cast<size_t>(i * 4 + 0)]);
        float y1 = tensorElementToFloat(boxes.data[static_cast<size_t>(i * 4 + 1)]);
        float x2 = tensorElementToFloat(boxes.data[static_cast<size_t>(i * 4 + 2)]);
        float y2 = tensorElementToFloat(boxes.data[static_cast<size_t>(i * 4 + 3)]);

        int class_id = static_cast<int>(tensorElementToFloat(classes.data[static_cast<size_t>(i)]));

        Detection det;
        det.class_id = static_cast<float>(class_id);
        det.class_confidence = score;
        det.bbox = fromCvRect(scaleXyxyToOriginal(x1, y1, x2, y2, frame_size));
        detections.push_back(det);
    }

    // YOLOv7 end-to-end already has NMS applied, so no need to apply it again
    return detections;
}

} // namespace neuriplo_tasks

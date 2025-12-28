#include "vision-core/object_detection/yolo_postprocessor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vision_core {

YoloPostprocessor::YoloPostprocessor(ObjectDetectionTask::ModelType model_type, 
                                     const cv::Size& input_size,
                                     float confidence_threshold, 
                                     float nms_threshold)
    : model_type_(model_type)
    , input_size_(input_size)
    , confidence_threshold_(confidence_threshold)
    , nms_threshold_(nms_threshold) {}

cv::Rect YoloPostprocessor::scaleToOriginal(float cx, float cy, float w, float h, const cv::Size& frame_size) const {
    // Apply letterbox inverse transformation
    // This converts coordinates from letterboxed model space to original frame space
    float r_w = static_cast<float>(input_size_.width) / frame_size.width;
    float r_h = static_cast<float>(input_size_.height) / frame_size.height;
    
    int x, y, width, height;
    
    if (r_h > r_w) {
        // Width is the limiting factor - padding is on top/bottom
        float pad_h = (input_size_.height - r_w * frame_size.height) / 2.0f;
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
        float pad_w = (input_size_.width - r_h * frame_size.width) / 2.0f;
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

std::vector<Detection> YoloPostprocessor::postprocess(
    const std::vector<std::vector<TensorElement>>& infer_results,
    const std::vector<std::vector<int64_t>>& infer_shapes,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;

    switch (model_type_) {
        case ObjectDetectionTask::ModelType::YOLO_STANDARD: {
            if (infer_results.empty() || infer_shapes.empty()) return {};
            detections = postprocessYoloStandard(infer_results[0], infer_shapes[0], frame_size);
            break;
        }

        case ObjectDetectionTask::ModelType::YOLO_V4: {
            if (infer_results.empty() || infer_shapes.empty()) return {};
            detections = postprocessYoloV4(infer_results, infer_shapes, frame_size);
            break;
        }

        case ObjectDetectionTask::ModelType::YOLO_V10: {
            if (infer_results.empty() || infer_shapes.empty()) return {};
            detections = postprocessYoloV10(infer_results[0], infer_shapes[0], frame_size);
            break;
        }
        
        case ObjectDetectionTask::ModelType::YOLO_NAS: {
            if (infer_results.size() < 2) {
                throw std::runtime_error("YOLO-NAS requires 2 output tensors");
            }
            detections = postprocessYoloNAS(infer_results[0], infer_results[1], 
                                          infer_shapes[0], infer_shapes[1], frame_size);
            break;
        }
        
        case ObjectDetectionTask::ModelType::YOLO_V7_E2E: {
            if (infer_results.size() < 4) {
                throw std::runtime_error("YOLOv7 end-to-end requires 4 output tensors");
            }
            // Outputs: num_dets, det_boxes, det_scores, det_classes
            detections = postprocessYoloV7E2E(infer_results[0], infer_results[1],
                                            infer_results[2], infer_results[3],
                                            infer_shapes[1], frame_size);
            break;
        }
        
        default:
            throw std::runtime_error("Unsupported YOLO model type");
    }

    return detections;
}

std::vector<Detection> YoloPostprocessor::postprocessYoloStandard(
    const std::vector<TensorElement>& output,
    const std::vector<int64_t>& shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    
    // Check shape dimensions
    if (shape.size() < 3) return {};
    
    // int batch = shape[0]; // Unused
    int dim1 = shape[1];
    int dim2 = shape[2];
    
    // Detect format by shape:
    // YOLOv5/v6/v7: [batch, anchors, 4+1+classes] where dim2 < dim1, has objectness at index 4
    // YOLOv8+:      [batch, 4+classes, anchors] where dim1 < dim2, no objectness
    bool has_objectness = (dim2 < dim1);  // YOLOv5/v6/v7 format
    
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
    
    if (num_classes <= 0) return {};

    for (int i = 0; i < anchors; ++i) {
        // For YOLOv5/v6/v7, check objectness score first (index 4)
        float objectness = 1.0f;
        if (has_objectness) {
            objectness = getTensorFloat(output[i * channels + 4]);
            if (objectness < confidence_threshold_) continue;
        }
        
        // Extract class scores
        float max_class_score = 0.0f;
        int class_id = -1;
        
        for (int c = 0; c < num_classes; ++c) {
            float score;
            if (has_objectness) {
                // YOLOv5/v6/v7: [batch, anchors, channels] -> data[i * channels + (c + 5)]
                score = getTensorFloat(output[i * channels + (c + class_offset)]);
            } else {
                // YOLOv8+: [batch, channels, anchors] -> data[(c + 4) * anchors + i]
                score = getTensorFloat(output[(c + class_offset) * anchors + i]);
            }
            
            if (score > max_class_score) {
                max_class_score = score;
                class_id = c;
            }
        }
        
        // For YOLOv5/v6/v7, multiply objectness with class score
        float final_score = has_objectness ? (objectness * max_class_score) : max_class_score;
        
        if (final_score < confidence_threshold_) continue;
        
        // Extract box coordinates
        float cx, cy, w, h;
        if (has_objectness) {
            // YOLOv5/v6/v7: [batch, anchors, channels]
            cx = getTensorFloat(output[i * channels + 0]);
            cy = getTensorFloat(output[i * channels + 1]);
            w  = getTensorFloat(output[i * channels + 2]);
            h  = getTensorFloat(output[i * channels + 3]);
        } else {
            // YOLOv8+: [batch, channels, anchors]
            cx = getTensorFloat(output[0 * anchors + i]);
            cy = getTensorFloat(output[1 * anchors + i]);
            w  = getTensorFloat(output[2 * anchors + i]);
            h  = getTensorFloat(output[3 * anchors + i]);
        }
        
        Detection det;
        det.class_id = class_id;
        det.class_confidence = final_score;
        det.bbox = scaleToOriginal(cx, cy, w, h, frame_size);
        detections.push_back(det);
    }
    
    applyNMS(detections);
    return detections;
}

std::vector<Detection> YoloPostprocessor::postprocessYoloV4(
    const std::vector<std::vector<TensorElement>> &outputs,
    const std::vector<std::vector<int64_t>> &shapes,
    const cv::Size &frame_size) {
  std::vector<Detection> detections;

  // Iterate over the output tensors
  for (size_t i = 0; i < outputs.size(); ++i) {
    const TensorElement *output = outputs[i].data();

    // Iterate over the detections in the output tensor
    for (int j = 0; j < shapes[i][0]; ++j, output += shapes[i][1]) {
      // Find the class with the highest confidence
      // Find the class with the highest confidence
      auto maxSPtr = std::max_element(
          output + 5, output + shapes[i][1],
          [this](const TensorElement &a, const TensorElement &b) {
            return this->getTensorFloat(a) < this->getTensorFloat(b);
          });

      float score = getTensorFloat(*maxSPtr);

      // Check if the confidence is above the threshold
      if (score > confidence_threshold_) {
        float cx = getTensorFloat(output[0]);
        float cy = getTensorFloat(output[1]);
        float w = getTensorFloat(output[2]);
        float h = getTensorFloat(output[3]);

        int label = maxSPtr - (output + 5);

        // YOLOv4 outputs normalized coordinates [0,1] relative to the model
        // input Convert to model input pixel coordinates
        float cx_model = cx * input_size_.width;
        float cy_model = cy * input_size_.height;
        float w_model = w * input_size_.width;
        float h_model = h * input_size_.height;

        // Apply letterbox inverse transformation using model-space coordinates
        cv::Rect bbox =
            scaleToOriginal(cx_model, cy_model, w_model, h_model, frame_size);

        // Create a detection object
        Detection detection;
        detection.bbox = bbox;
        detection.class_confidence = score;
        detection.class_id = label;

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
      class2indices[detections[i].class_id].push_back(i);
    }
  }

  for (std::map<int, std::vector<size_t>>::iterator it = class2indices.begin();
       it != class2indices.end(); ++it) {
    std::vector<cv::Rect> localBoxes;
    std::vector<float> localConfidences;
    std::vector<size_t> classIndices = it->second;
    for (size_t i = 0; i < classIndices.size(); i++) {
      localBoxes.push_back(detections[classIndices[i]].bbox);
      localConfidences.push_back(detections[classIndices[i]].class_confidence);
    }
    std::vector<int> nmsIndices;
    cv::dnn::NMSBoxes(localBoxes, localConfidences, confidence_threshold_,
                      nms_threshold_, nmsIndices);
    for (size_t i = 0; i < nmsIndices.size(); i++) {
      filtered_detections.push_back(detections[classIndices[nmsIndices[i]]]);
    }
  }

  return filtered_detections;
}

std::vector<Detection> YoloPostprocessor::postprocessYoloV10(
    const std::vector<TensorElement>& output,
    const std::vector<int64_t>& shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    
    // YOLOv10 output: [1, 300, 6] (x1, y1, x2, y2, score, class)
    if (shape.size() < 3 || shape[2] < 6) return {};
    
    int num_dets = shape[1];
    int dims = shape[2];
    
    // Calculate scale ratios for letterbox inverse
    float r_w = static_cast<float>(input_size_.width) / frame_size.width;
    float r_h = static_cast<float>(input_size_.height) / frame_size.height;
    
    for (int i = 0; i < num_dets; ++i) {
        float score = getTensorFloat(output[i * dims + 4]);
        if (score < confidence_threshold_) continue;
        
        float x1 = getTensorFloat(output[i * dims + 0]);
        float y1 = getTensorFloat(output[i * dims + 1]);
        float x2 = getTensorFloat(output[i * dims + 2]);
        float y2 = getTensorFloat(output[i * dims + 3]);
        int class_id = static_cast<int>(getTensorFloat(output[i * dims + 5]));
        
        // Apply letterbox inverse transformation
        if (r_h > r_w) {
            float pad_h = (input_size_.height - r_w * frame_size.height) / 2.0f;
            y1 = (y1 - pad_h) / r_w;
            y2 = (y2 - pad_h) / r_w;
            x1 = x1 / r_w;
            x2 = x2 / r_w;
        } else {
            float pad_w = (input_size_.width - r_h * frame_size.width) / 2.0f;
            x1 = (x1 - pad_w) / r_h;
            x2 = (x2 - pad_w) / r_h;
            y1 = y1 / r_h;
            y2 = y2 / r_h;
        }
        
        Detection det;
        det.class_id = class_id;
        det.class_confidence = score;
        det.bbox = cv::Rect(static_cast<int>(x1), static_cast<int>(y1), 
                           static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
        detections.push_back(det);
    }
    
    // YOLOv10 typically doesn't need NMS, but we can apply it if needed
    // applyNMS(detections); 
    return detections;
}

std::vector<Detection> YoloPostprocessor::postprocessYoloNAS(
    const std::vector<TensorElement>& boxes,
    const std::vector<TensorElement>& scores,
    const std::vector<int64_t>& box_shape,
    const std::vector<int64_t>& score_shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    
    // Boxes: [1, N, 4], Scores: [1, N, C]
    if (box_shape.size() < 3 || score_shape.size() < 3) return {};
    
    int num_dets = box_shape[1];
    int num_classes = score_shape[2];
    
    // Calculate scale ratios for letterbox inverse
    float r_w = static_cast<float>(input_size_.width) / frame_size.width;
    float r_h = static_cast<float>(input_size_.height) / frame_size.height;
    
    for (int i = 0; i < num_dets; ++i) {
        // Find max score for this detection
        float max_score = 0.0f;
        int class_id = -1;
        
        for (int c = 0; c < num_classes; ++c) {
            float score = getTensorFloat(scores[i * num_classes + c]);
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }
        
        if (max_score < confidence_threshold_) continue;
        
        float x1 = getTensorFloat(boxes[i * 4 + 0]);
        float y1 = getTensorFloat(boxes[i * 4 + 1]);
        float x2 = getTensorFloat(boxes[i * 4 + 2]);
        float y2 = getTensorFloat(boxes[i * 4 + 3]);
        
        // Apply letterbox inverse transformation
        if (r_h > r_w) {
            float pad_h = (input_size_.height - r_w * frame_size.height) / 2.0f;
            y1 = (y1 - pad_h) / r_w;
            y2 = (y2 - pad_h) / r_w;
            x1 = x1 / r_w;
            x2 = x2 / r_w;
        } else {
            float pad_w = (input_size_.width - r_h * frame_size.width) / 2.0f;
            x1 = (x1 - pad_w) / r_h;
            x2 = (x2 - pad_w) / r_h;
            y1 = y1 / r_h;
            y2 = y2 / r_h;
        }
        
        Detection det;
        det.class_id = class_id;
        det.class_confidence = max_score;
        det.bbox = cv::Rect(static_cast<int>(x1), static_cast<int>(y1), 
                           static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
        detections.push_back(det);
    }
    
    applyNMS(detections);
    return detections;
}

void YoloPostprocessor::applyNMS(std::vector<Detection>& detections) {
    if (detections.empty()) return;
    
    std::vector<bool> suppress(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppress[i]) continue;
        
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppress[j]) continue;
            
            cv::Rect intersection = detections[i].bbox & detections[j].bbox;
            float intersection_area = intersection.area();
            float union_area = detections[i].bbox.area() + detections[j].bbox.area() - intersection_area;
            
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
    
    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
                      [&](const Detection& det) {
                          size_t idx = &det - &detections[0];
                          return suppress[idx];
                      }),
        detections.end()
    );
}

float YoloPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

std::vector<Detection> YoloPostprocessor::postprocessYoloV7E2E(
    const std::vector<TensorElement>& num_dets_tensor,
    const std::vector<TensorElement>& boxes,
    const std::vector<TensorElement>& scores,
    const std::vector<TensorElement>& classes,
    const std::vector<int64_t>& boxes_shape,
    const cv::Size& frame_size) {
    
    std::vector<Detection> detections;
    
    // YOLOv7 end-to-end format:
    // num_dets: [1,1] - number of valid detections
    // boxes: [1, max_dets, 4] - bounding boxes in [x1, y1, x2, y2] format (model space)
    // scores: [1, max_dets] - confidence scores
    // classes: [1, max_dets] - class IDs
    
    if (num_dets_tensor.empty() || boxes.empty() || scores.empty() || classes.empty()) {
        return detections;
    }
    
    // Get actual number of detections
    int num_dets = static_cast<int>(getTensorFloat(num_dets_tensor[0]));
    
    // boxes_shape is [1, max_dets, 4]
    int max_dets = boxes_shape.size() >= 2 ? boxes_shape[1] : 100;
    num_dets = std::min(num_dets, max_dets);  // Clamp to max_dets
    
    // Calculate scale ratios for letterbox inverse
    float r_w = static_cast<float>(input_size_.width) / frame_size.width;
    float r_h = static_cast<float>(input_size_.height) / frame_size.height;
    
    for (int i = 0; i < num_dets; ++i) {
        float score = getTensorFloat(scores[i]);
        if (score < confidence_threshold_) continue;
        
        // Extract box coordinates in model space
        float x1 = getTensorFloat(boxes[i * 4 + 0]);
        float y1 = getTensorFloat(boxes[i * 4 + 1]);
        float x2 = getTensorFloat(boxes[i * 4 + 2]);
        float y2 = getTensorFloat(boxes[i * 4 + 3]);
        
        // Apply letterbox inverse transformation
        if (r_h > r_w) {
            float pad_h = (input_size_.height - r_w * frame_size.height) / 2.0f;
            y1 = (y1 - pad_h) / r_w;
            y2 = (y2 - pad_h) / r_w;
            x1 = x1 / r_w;
            x2 = x2 / r_w;
        } else {
            float pad_w = (input_size_.width - r_h * frame_size.width) / 2.0f;
            x1 = (x1 - pad_w) / r_h;
            x2 = (x2 - pad_w) / r_h;
            y1 = y1 / r_h;
            y2 = y2 / r_h;
        }
        
        int class_id = static_cast<int>(getTensorFloat(classes[i]));
        
        Detection det;
        det.class_id = class_id;
        det.class_confidence = score;
        det.bbox = cv::Rect(static_cast<int>(x1), static_cast<int>(y1), 
                           static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
        detections.push_back(det);
    }
    
    // YOLOv7 end-to-end already has NMS applied, so no need to apply it again
    return detections;
}

} // namespace vision_core

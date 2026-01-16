#include "vision-core/instance_segmentation/yolo_segmentation_postprocessor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vision_core {

YoloSegmentationPostprocessor::YoloSegmentationPostprocessor(
    InstanceSegmentationTask::ModelType model_type, const cv::Size& input_size,
    float confidence_threshold, float nms_threshold, float mask_threshold)
    : model_type_(model_type), input_size_(input_size),
      confidence_threshold_(confidence_threshold),
      nms_threshold_(nms_threshold), mask_threshold_(mask_threshold) {}

std::vector<InstanceSegmentation>
YoloSegmentationPostprocessor::postprocess(const std::vector<Tensor>& tensors,
                                           const cv::Size& frame_size) {
  switch (model_type_) {
  case InstanceSegmentationTask::ModelType::YOLO_V10_SEG:
  case InstanceSegmentationTask::ModelType::YOLO_26_SEG:
    return postprocessYoloNmsFreeSeg(tensors, frame_size);

  case InstanceSegmentationTask::ModelType::YOLO_SEG:
  default:
    return postprocessYoloSeg(tensors, frame_size);
  }
}

std::vector<InstanceSegmentation>
YoloSegmentationPostprocessor::postprocessYoloSeg(
    const std::vector<Tensor>& tensors, const cv::Size& frame_size) {

  if (tensors.size() < 2) {
    throw std::runtime_error(
        "YOLO segmentation requires at least 2 output tensors");
  }

  // Output 0: Detections [1, 4+cls+32, 8400]
  // Output 1: Mask Protos [1, 32, 160, 160]

  const auto& dets_tensor = tensors[0].data;
  const auto& protos_tensor = tensors[1].data;
  const auto& dets_shape = tensors[0].shape;
  const auto& protos_shape = tensors[1].shape;

  if (dets_shape.size() < 3 || protos_shape.size() < 4) {
    return {};
  }

  int channels = dets_shape[1];
  int anchors = dets_shape[2];
  int num_mask_coeffs = 32;
  int num_classes = channels - 4 - num_mask_coeffs;

  if (num_classes <= 0) {
    return {};
  }

  int proto_h = protos_shape[2];
  int proto_w = protos_shape[3];

  const float* dets_data = std::get_if<float>(&dets_tensor[0]);
  const float* protos_data = std::get_if<float>(&protos_tensor[0]);

  if (!dets_data || !protos_data) {
    return {};
  }

  // Collect all detections before NMS
  std::vector<Detection> detections;

  for (int i = 0; i < anchors; ++i) {
    // Find max class score
    float max_score = 0.0f;
    int class_id = -1;

    for (int c = 0; c < num_classes; ++c) {
      float score = dets_data[(c + 4) * anchors + i];
      if (score > max_score) {
        max_score = score;
        class_id = c;
      }
    }

    if (max_score < confidence_threshold_) {
      continue;
    }

    // Extract box (cx, cy, w, h)
    float cx = dets_data[0 * anchors + i];
    float cy = dets_data[1 * anchors + i];
    float w = dets_data[2 * anchors + i];
    float h = dets_data[3 * anchors + i];

    float x1 = cx - w / 2.0f;
    float y1 = cy - h / 2.0f;
    float x2 = cx + w / 2.0f;
    float y2 = cy + h / 2.0f;

    // Extract mask coefficients
    std::vector<float> mask_coeffs;
    mask_coeffs.reserve(num_mask_coeffs);
    for (int m = 0; m < num_mask_coeffs; ++m) {
      mask_coeffs.push_back(dets_data[(4 + num_classes + m) * anchors + i]);
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

  // Apply NMS
  std::vector<Detection> nms_detections = applyNMS(detections);

  // Process masks for NMS survivors
  std::vector<InstanceSegmentation> segmentations;

  for (const auto& det : nms_detections) {
    // Generate mask from coefficients and prototypes
    cv::Mat mask = generateMask(det.mask_coeffs, protos_data, proto_h, proto_w);

    // Scale bbox to original image coordinates
    cv::Rect bbox = scaleToOriginal(det.x1, det.y1, det.x2, det.y2, frame_size);

    // Crop and resize mask to bbox
    cv::Mat final_mask = cropAndResizeMask(mask, det.x1, det.y1, det.x2, det.y2,
                                           bbox, frame_size);

    // Check if mask has any content
    double max_val;
    cv::minMaxLoc(final_mask, nullptr, &max_val);
    if (max_val <= 0.0f) {
      continue;
    }

    InstanceSegmentation seg;
    seg.class_id = det.class_id;
    seg.class_confidence = det.confidence;
    seg.bbox = bbox;
    seg.mask = final_mask;
    seg.mask_height = final_mask.rows;
    seg.mask_width = final_mask.cols;

    segmentations.push_back(seg);
  }

  return segmentations;
}

std::vector<InstanceSegmentation>
YoloSegmentationPostprocessor::postprocessYoloNmsFreeSeg(
    const std::vector<Tensor>& tensors, const cv::Size& frame_size) {

  if (tensors.size() < 2) {
    throw std::runtime_error(
        "YOLO NMS-free segmentation requires 2 output tensors");
  }

  // Output 0: Detections [1, 300, 38] (x1, y1, x2, y2, score, class, 32 coeffs)
  // Output 1: Mask Prototypes [1, 32, 160, 160]

  const auto& dets_tensor = tensors[0];
  const auto& protos_tensor = tensors[1];
  const auto& dets_shape = dets_tensor.shape;
  const auto& protos_shape = protos_tensor.shape;

  if (dets_shape.size() < 3 || protos_shape.size() < 4) {
    return {};
  }

  int num_dets = dets_shape[1];    // 300
  int det_dims = dets_shape[2];    // 38
  int num_protos = protos_shape[1]; // 32
  int proto_h = protos_shape[2];    // 160
  int proto_w = protos_shape[3];    // 160

  if (det_dims < 38 || num_protos != 32) {
    return {};
  }

  std::vector<InstanceSegmentation> segmentations;

  // Extract prototype data once
  std::vector<float> protos_data(num_protos * proto_h * proto_w);
  for (size_t i = 0; i < protos_data.size(); ++i) {
    protos_data[i] = getTensorFloat(protos_tensor.data[i]);
  }

  for (int i = 0; i < num_dets; ++i) {
    // Extract detection data
    float x1 = getTensorFloat(dets_tensor.data[i * det_dims + 0]);
    float y1 = getTensorFloat(dets_tensor.data[i * det_dims + 1]);
    float x2 = getTensorFloat(dets_tensor.data[i * det_dims + 2]);
    float y2 = getTensorFloat(dets_tensor.data[i * det_dims + 3]);
    float score = getTensorFloat(dets_tensor.data[i * det_dims + 4]);
    int class_id =
        static_cast<int>(getTensorFloat(dets_tensor.data[i * det_dims + 5]));

    // Filter by confidence
    if (score < confidence_threshold_) {
      continue;
    }

    // Extract mask coefficients (32 values starting at index 6)
    std::vector<float> mask_coeffs;
    mask_coeffs.reserve(32);
    for (int c = 0; c < 32; ++c) {
      mask_coeffs.push_back(
          getTensorFloat(dets_tensor.data[i * det_dims + 6 + c]));
    }

    // Generate mask at proto resolution (160x160)
    cv::Mat mask_proto = cv::Mat::zeros(proto_h, proto_w, CV_32F);

    for (int h = 0; h < proto_h; ++h) {
      for (int w = 0; w < proto_w; ++w) {
        float sum = 0.0f;
        for (int c = 0; c < num_protos; ++c) {
          int proto_idx = c * proto_h * proto_w + h * proto_w + w;
          sum += mask_coeffs[c] * protos_data[proto_idx];
        }
        // Apply sigmoid activation
        mask_proto.at<float>(h, w) = 1.0f / (1.0f + std::exp(-sum));
      }
    }

    // Scale bbox coordinates from input space to proto mask space
    float scale_x = static_cast<float>(proto_w) / input_size_.width;
    float scale_y = static_cast<float>(proto_h) / input_size_.height;

    int proto_x1 = static_cast<int>(x1 * scale_x);
    int proto_y1 = static_cast<int>(y1 * scale_y);
    int proto_x2 = static_cast<int>(x2 * scale_x);
    int proto_y2 = static_cast<int>(y2 * scale_y);

    // Clamp to proto dimensions
    proto_x1 = std::max(0, std::min(proto_x1, proto_w - 1));
    proto_y1 = std::max(0, std::min(proto_y1, proto_h - 1));
    proto_x2 = std::max(proto_x1 + 1, std::min(proto_x2, proto_w));
    proto_y2 = std::max(proto_y1 + 1, std::min(proto_y2, proto_h));

    // Crop mask to bbox region in proto space
    cv::Rect proto_bbox(proto_x1, proto_y1, proto_x2 - proto_x1,
                        proto_y2 - proto_y1);

    if (proto_bbox.width <= 0 || proto_bbox.height <= 0) {
      continue;
    }

    cv::Mat mask_cropped = mask_proto(proto_bbox).clone();

    // Check if mask has any content (equivalent to Python's masks.amax() > 0)
    double max_val;
    cv::minMaxLoc(mask_cropped, nullptr, &max_val);
    if (max_val <= 0.0f) {
      continue;
    }

    // Apply letterbox inverse transformation to bounding box
    cv::Rect bbox = scaleToOriginal(x1, y1, x2, y2, frame_size);

    // Ensure bbox is within frame bounds
    bbox.x = std::max(0, std::min(bbox.x, frame_size.width - 1));
    bbox.y = std::max(0, std::min(bbox.y, frame_size.height - 1));
    bbox.width = std::max(1, std::min(bbox.width, frame_size.width - bbox.x));
    bbox.height = std::max(1, std::min(bbox.height, frame_size.height - bbox.y));

    // Resize cropped mask to bbox size in original frame
    cv::Mat mask_resized;
    cv::resize(mask_cropped, mask_resized, cv::Size(bbox.width, bbox.height),
               0, 0, cv::INTER_LINEAR);

    // Threshold mask
    cv::Mat mask_binary;
    cv::threshold(mask_resized, mask_binary, mask_threshold_, 255,
                  cv::THRESH_BINARY);
    mask_binary.convertTo(mask_binary, CV_8UC1);

    // Create full-frame mask and place the bbox mask in it
    cv::Mat mask_full = cv::Mat::zeros(frame_size, CV_8UC1);
    mask_binary.copyTo(mask_full(bbox));

    // Create segmentation result
    InstanceSegmentation seg;
    seg.class_id = class_id;
    seg.class_confidence = score;
    seg.bbox = bbox;
    seg.mask = mask_full;
    seg.mask_height = mask_full.rows;
    seg.mask_width = mask_full.cols;

    segmentations.push_back(seg);
  }

  return segmentations;
}

cv::Rect YoloSegmentationPostprocessor::scaleToOriginal(
    float x1, float y1, float x2, float y2, const cv::Size& frame_size) const {

  // Calculate scale ratios for letterbox inverse
  float r_w = static_cast<float>(input_size_.width) / frame_size.width;
  float r_h = static_cast<float>(input_size_.height) / frame_size.height;
  float r = std::min(r_w, r_h);

  // Calculate padding
  float pad_w = (input_size_.width - r * frame_size.width) / 2.0f;
  float pad_h = (input_size_.height - r * frame_size.height) / 2.0f;

  // Remove padding and scale to original
  float orig_x1 = (x1 - pad_w) / r;
  float orig_y1 = (y1 - pad_h) / r;
  float orig_x2 = (x2 - pad_w) / r;
  float orig_y2 = (y2 - pad_h) / r;

  // Clamp to frame boundaries
  orig_x1 = std::max(0.0f, std::min(orig_x1, static_cast<float>(frame_size.width)));
  orig_y1 = std::max(0.0f, std::min(orig_y1, static_cast<float>(frame_size.height)));
  orig_x2 = std::max(0.0f, std::min(orig_x2, static_cast<float>(frame_size.width)));
  orig_y2 = std::max(0.0f, std::min(orig_y2, static_cast<float>(frame_size.height)));

  return cv::Rect(static_cast<int>(orig_x1), static_cast<int>(orig_y1),
                  static_cast<int>(orig_x2 - orig_x1),
                  static_cast<int>(orig_y2 - orig_y1));
}

float YoloSegmentationPostprocessor::getTensorFloat(
    const TensorElement& element) {
  return std::visit(
      [](auto&& value) -> float { return static_cast<float>(value); }, element);
}

cv::Mat YoloSegmentationPostprocessor::generateMask(
    const std::vector<float>& coeffs, const float* protos_data, int proto_h,
    int proto_w) {

  cv::Mat mask = cv::Mat::zeros(proto_h, proto_w, CV_32F);

  for (int h = 0; h < proto_h; ++h) {
    for (int w = 0; w < proto_w; ++w) {
      float sum = 0.0f;
      for (size_t c = 0; c < coeffs.size(); ++c) {
        int proto_idx = c * proto_h * proto_w + h * proto_w + w;
        sum += coeffs[c] * protos_data[proto_idx];
      }
      // Apply sigmoid activation
      mask.at<float>(h, w) = 1.0f / (1.0f + std::exp(-sum));
    }
  }

  return mask;
}

cv::Mat YoloSegmentationPostprocessor::cropAndResizeMask(
    const cv::Mat& mask, float x1, float y1, float x2, float y2,
    const cv::Rect& bbox, const cv::Size& frame_size) {

  int proto_h = mask.rows;
  int proto_w = mask.cols;

  // Scale bbox coordinates to proto mask space
  float scale_x = static_cast<float>(proto_w) / input_size_.width;
  float scale_y = static_cast<float>(proto_h) / input_size_.height;

  int proto_x1 = static_cast<int>(x1 * scale_x);
  int proto_y1 = static_cast<int>(y1 * scale_y);
  int proto_x2 = static_cast<int>(x2 * scale_x);
  int proto_y2 = static_cast<int>(y2 * scale_y);

  // Clamp to proto dimensions
  proto_x1 = std::max(0, std::min(proto_x1, proto_w - 1));
  proto_y1 = std::max(0, std::min(proto_y1, proto_h - 1));
  proto_x2 = std::max(proto_x1 + 1, std::min(proto_x2, proto_w));
  proto_y2 = std::max(proto_y1 + 1, std::min(proto_y2, proto_h));

  // Crop mask to bbox region
  cv::Rect proto_bbox(proto_x1, proto_y1, proto_x2 - proto_x1,
                      proto_y2 - proto_y1);
  cv::Mat mask_cropped = mask(proto_bbox).clone();

  // Resize to bbox size
  cv::Mat mask_resized;
  cv::resize(mask_cropped, mask_resized, cv::Size(bbox.width, bbox.height), 0,
             0, cv::INTER_LINEAR);

  // Threshold
  cv::Mat mask_binary;
  cv::threshold(mask_resized, mask_binary, mask_threshold_, 255,
                cv::THRESH_BINARY);
  mask_binary.convertTo(mask_binary, CV_8UC1);

  // Create full-frame mask
  cv::Mat mask_full = cv::Mat::zeros(frame_size, CV_8UC1);
  if (bbox.x >= 0 && bbox.y >= 0 &&
      bbox.x + bbox.width <= frame_size.width &&
      bbox.y + bbox.height <= frame_size.height) {
    mask_binary.copyTo(mask_full(bbox));
  }

  return mask_full;
}

std::vector<YoloSegmentationPostprocessor::Detection>
YoloSegmentationPostprocessor::applyNMS(
    const std::vector<Detection>& detections) {

  if (detections.empty()) {
    return {};
  }

  // Sort by confidence (descending)
  std::vector<Detection> sorted_dets = detections;
  std::sort(sorted_dets.begin(), sorted_dets.end(),
            [](const Detection& a, const Detection& b) {
              return a.confidence > b.confidence;
            });

  std::vector<bool> suppressed(sorted_dets.size(), false);
  std::vector<Detection> result;

  for (size_t i = 0; i < sorted_dets.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }

    result.push_back(sorted_dets[i]);

    // Suppress overlapping boxes
    for (size_t j = i + 1; j < sorted_dets.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }

      // Only suppress boxes of the same class
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

float YoloSegmentationPostprocessor::calculateIoU(const Detection& a,
                                                  const Detection& b) {
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

} // namespace vision_core
#include "vision-core/open_vocab_detection/owlv2_postprocessor.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vision_core {

namespace {

float tensorElementToFloat(const TensorElement& element) {
    return std::visit([](const auto& value) { return static_cast<float>(value); }, element);
}

float sigmoid(float value) {
    if (value >= 0.0F) {
        const float exp_neg = std::exp(-value);
        return 1.0F / (1.0F + exp_neg);
    }
    const float exp_pos = std::exp(value);
    return exp_pos / (1.0F + exp_pos);
}

std::string normalizeName(const std::string& value) {
    std::string normalized;
    normalized.reserve(value.size());
    for (char c : value) {
        if (c == '-' || c == '_' || std::isspace(static_cast<unsigned char>(c)) != 0) {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return normalized;
}

const Tensor* findTensorByNames(const std::vector<Tensor>& tensors, const std::vector<std::string>& output_names,
                                const std::vector<std::string>& candidates) {
    for (size_t index = 0; index < output_names.size() && index < tensors.size(); ++index) {
        const std::string output_name = normalizeName(output_names[index]);
        for (const auto& candidate : candidates) {
            if (output_name == normalizeName(candidate)) {
                return &tensors[index];
            }
        }
    }
    return nullptr;
}

cv::Rect makeRectFromCenterBox(float center_x, float center_y, float width, float height, const cv::Size& frame_size) {
    const float x1 = std::max(0.0F, center_x - (width * 0.5F));
    const float y1 = std::max(0.0F, center_y - (height * 0.5F));
    const float x2 = std::min(static_cast<float>(frame_size.width), center_x + (width * 0.5F));
    const float y2 = std::min(static_cast<float>(frame_size.height), center_y + (height * 0.5F));

    return cv::Rect(cv::Point(static_cast<int>(std::round(x1)), static_cast<int>(std::round(y1))),
                    cv::Point(static_cast<int>(std::round(x2)), static_cast<int>(std::round(y2))));
}

} // namespace

OWLv2Postprocessor::OWLv2Postprocessor(const cv::Size& input_size, float confidence_threshold, float text_threshold,
                                       std::vector<std::string> prompt_labels, std::vector<std::string> output_names)
    : input_size_(input_size), confidence_threshold_(confidence_threshold), text_threshold_(text_threshold),
      prompt_labels_(std::move(prompt_labels)), output_names_(std::move(output_names)) {}

std::vector<OpenVocabDetection> OWLv2Postprocessor::postprocess(const std::vector<Tensor>& tensors,
                                                                const cv::Size& frame_size) {
    std::vector<OpenVocabDetection> results;
    if (tensors.size() < 2) {
        return results;
    }

    const Tensor* boxes_ptr = findTensorByNames(tensors, output_names_, {"pred_boxes", "target_pred_boxes", "boxes"});
    const Tensor* logits_ptr = findTensorByNames(tensors, output_names_, {"logits", "class_logits"});
    const Tensor* objectness_ptr = findTensorByNames(tensors, output_names_, {"objectness_logits", "objectness"});

    const Tensor& boxes = boxes_ptr != nullptr ? *boxes_ptr : tensors[0];
    const Tensor& logits = logits_ptr != nullptr ? *logits_ptr : tensors[1];

    if (boxes.shape.empty() || logits.shape.empty()) {
        return results;
    }

    const int64_t num_boxes = boxes.shape.size() >= 2 ? boxes.shape[boxes.shape.size() - 2] : 0;
    const int64_t box_dims = boxes.shape.back();
    const int64_t num_prompts = logits.shape.back();
    if (num_boxes <= 0 || box_dims != 4 || num_prompts <= 0) {
        return results;
    }

    const size_t boxes_per_item = static_cast<size_t>(num_boxes * box_dims);
    const size_t logits_per_item = static_cast<size_t>(num_boxes * num_prompts);
    if (boxes.data.size() < boxes_per_item || logits.data.size() < logits_per_item) {
        return results;
    }

    for (int64_t box_index = 0; box_index < num_boxes; ++box_index) {
        float best_score = -std::numeric_limits<float>::infinity();
        int best_prompt_index = -1;

        for (int64_t prompt_index = 0; prompt_index < num_prompts; ++prompt_index) {
            const size_t offset = static_cast<size_t>(box_index * num_prompts + prompt_index);
            const float score = sigmoid(tensorElementToFloat(logits.data[offset]));
            if (score > best_score) {
                best_score = score;
                best_prompt_index = static_cast<int>(prompt_index);
            }
        }

        if (objectness_ptr != nullptr && !objectness_ptr->shape.empty() &&
            objectness_ptr->data.size() > static_cast<size_t>(box_index)) {
            best_score *= sigmoid(tensorElementToFloat(objectness_ptr->data[static_cast<size_t>(box_index)]));
        }

        if (best_score < confidence_threshold_ || best_score < text_threshold_) {
            continue;
        }

        const size_t box_offset = static_cast<size_t>(box_index * 4);
        float center_x = tensorElementToFloat(boxes.data[box_offset]);
        float center_y = tensorElementToFloat(boxes.data[box_offset + 1]);
        float width = tensorElementToFloat(boxes.data[box_offset + 2]);
        float height = tensorElementToFloat(boxes.data[box_offset + 3]);

        // OWL-style exports commonly emit normalized cx, cy, w, h.
        if (std::max({std::fabs(center_x), std::fabs(center_y), std::fabs(width), std::fabs(height)}) <= 1.5F) {
            center_x *= static_cast<float>(frame_size.width);
            center_y *= static_cast<float>(frame_size.height);
            width *= static_cast<float>(frame_size.width);
            height *= static_cast<float>(frame_size.height);
        } else if (input_size_.width > 0 && input_size_.height > 0 &&
                   (frame_size.width != input_size_.width || frame_size.height != input_size_.height)) {
            const float scale_x = static_cast<float>(frame_size.width) / static_cast<float>(input_size_.width);
            const float scale_y = static_cast<float>(frame_size.height) / static_cast<float>(input_size_.height);
            center_x *= scale_x;
            center_y *= scale_y;
            width *= scale_x;
            height *= scale_y;
        }

        std::string label;
        if (best_prompt_index >= 0 && static_cast<size_t>(best_prompt_index) < prompt_labels_.size()) {
            label = prompt_labels_[static_cast<size_t>(best_prompt_index)];
        }

        results.emplace_back(makeRectFromCenterBox(center_x, center_y, width, height, frame_size), best_score,
                             best_prompt_index, std::move(label));
    }

    return results;
}

} // namespace vision_core

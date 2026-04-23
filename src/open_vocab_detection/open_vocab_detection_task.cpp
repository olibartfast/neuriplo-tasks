#include "vision-core/open_vocab_detection/open_vocab_detection_task.hpp"

#include "vision-core/object_detection/detection_preprocessor.hpp"
#include "vision-core/open_vocab_detection/grounding_dino_postprocessor.hpp"
#include "vision-core/open_vocab_detection/owlv2_postprocessor.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>

namespace vision_core {

namespace {

std::string normalizeModelName(const std::string& name) {
    std::string normalized;
    normalized.reserve(name.size());
    for (char c : name) {
        if (c == '-' || c == '_' || std::isspace(static_cast<unsigned char>(c)) != 0) {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return normalized;
}

std::vector<uint8_t> toByteBuffer(const std::vector<int32_t>& values) {
    // ONNX models typically expect int64 for token inputs
    std::vector<int64_t> values64(values.begin(), values.end());
    std::vector<uint8_t> bytes(values64.size() * sizeof(int64_t));
    if (!values64.empty()) {
        std::memcpy(bytes.data(), values64.data(), bytes.size());
    }
    return bytes;
}

bool isImageInput(const std::string& input_name, const std::vector<int64_t>& input_shape) {
    const std::string normalized = normalizeModelName(input_name);
    return input_shape.size() >= 3 || normalized.find("pixelvalues") != std::string::npos ||
           normalized.find("images") != std::string::npos;
}

} // namespace

OpenVocabDetectionTask::OpenVocabDetectionTask(const ModelInfo& model_info, const std::string& model_name,
                                               const TaskConfig& config)
    : TaskInterface(model_info), model_type_(detectModelType(model_name)), config_(config) {
    if (model_type_ == ModelType::Unknown) {
        throw std::invalid_argument("Unsupported open-vocabulary model: " + model_name);
    }

    const cv::Size input_size = extractInputSize(model_info_);
    input_width_ = input_size.width;
    input_height_ = input_size.height;
    image_preprocessor_ = std::make_unique<RtDetrPreprocessor>(input_size);

    switch (model_type_) {
    case ModelType::OWLV2:
    case ModelType::OWLVIT:
        if (!config_.tokenizer_vocab_json.empty() && !config_.tokenizer_merges_text.empty()) {
            tokenizer_ =
                std::make_unique<ClipTokenizer>(config_.tokenizer_vocab_json, config_.tokenizer_merges_text, true);
        } else if (!config_.tokenizer_vocab_path.empty() && !config_.tokenizer_merges_path.empty()) {
            tokenizer_ = std::make_unique<ClipTokenizer>(config_.tokenizer_vocab_path, config_.tokenizer_merges_path);
        }
        postprocessor_ =
            std::make_unique<OWLv2Postprocessor>(input_size, config_.confidence_threshold, config_.text_threshold,
                                                 extractPrompts(config_), model_info_.output_names);
        break;

    case ModelType::GroundingDino: {
        if (!config_.bert_tokenizer_vocab_text.empty()) {
            bert_tokenizer_ = std::make_unique<BertTokenizer>(config_.bert_tokenizer_vocab_text, true);
        } else if (!config_.bert_tokenizer_vocab_path.empty()) {
            bert_tokenizer_ = std::make_unique<BertTokenizer>(config_.bert_tokenizer_vocab_path);
        }

        // Pre-compute phrase token ranges (fixed for the lifetime of this task)
        const std::vector<std::string> prompts = extractPrompts(config_);
        if (bert_tokenizer_ && !prompts.empty()) {
            auto encoded = bert_tokenizer_->encodePhrases(prompts, config_.max_text_queries);
            phrase_token_ranges_ = std::move(encoded.phrase_token_ranges);
        }

        postprocessor_ = std::make_unique<GroundingDinoPostprocessor>(input_size, config_.confidence_threshold,
                                                                      config_.text_threshold, prompts,
                                                                      model_info_.output_names, phrase_token_ranges_);
        break;
    }

    default:
        throw std::invalid_argument("Unsupported open-vocabulary model type");
    }
}

OpenVocabDetectionTask::ModelType OpenVocabDetectionTask::detectModelType(const std::string& model_name) {
    const std::string normalized = normalizeModelName(model_name);
    if (normalized == "owlv2") {
        return ModelType::OWLV2;
    }
    if (normalized == "owlvit") {
        return ModelType::OWLVIT;
    }
    if (normalized == "groundingdino") {
        return ModelType::GroundingDino;
    }
    return ModelType::Unknown;
}

std::vector<std::string> OpenVocabDetectionTask::extractPrompts(const TaskConfig& config) {
    if (!config.text_prompts.empty()) {
        return config.text_prompts;
    }

    const auto it = config.extra_params.find("text_prompt");
    if (it == config.extra_params.end() || it->second.empty()) {
        return {};
    }

    std::vector<std::string> prompts;
    std::string current;
    for (char c : it->second) {
        if (c == ';' || c == '\n') {
            if (!current.empty()) {
                prompts.push_back(current);
                current.clear();
            }
            continue;
        }
        current.push_back(c);
    }
    if (!current.empty()) {
        prompts.push_back(current);
    }
    return prompts;
}

cv::Size OpenVocabDetectionTask::extractInputSize(const ModelInfo& model_info) {
    for (size_t index = 0; index < model_info.input_shapes.size(); ++index) {
        const auto& shape = model_info.input_shapes[index];
        const std::string format =
            index < model_info.input_formats.size() ? model_info.input_formats[index] : "FORMAT_NCHW";
        const std::string name = index < model_info.input_names.size() ? model_info.input_names[index] : "";
        if (!isImageInput(name, shape)) {
            continue;
        }

        if (shape.size() == 4) {
            if (format == "FORMAT_NHWC") {
                return cv::Size(static_cast<int>(shape[2]), static_cast<int>(shape[1]));
            }
            return cv::Size(static_cast<int>(shape[3]), static_cast<int>(shape[2]));
        }
        if (shape.size() == 3) {
            if (format == "FORMAT_NHWC") {
                return cv::Size(static_cast<int>(shape[1]), static_cast<int>(shape[0]));
            }
            return cv::Size(static_cast<int>(shape[2]), static_cast<int>(shape[1]));
        }
    }

    throw InputDimensionError("No valid image input found for open-vocabulary detection model");
}

std::pair<std::vector<int32_t>, std::vector<int32_t>> OpenVocabDetectionTask::encodePrompts(int context_length) const {
    const std::vector<std::string> prompts = extractPrompts(config_);
    if (prompts.empty()) {
        throw std::invalid_argument("Open-vocabulary detection requires at least one text prompt");
    }
    if (!tokenizer_) {
        throw std::invalid_argument("Tokenizer assets must be configured for open-vocabulary text inputs");
    }
    return tokenizer_->batchEncode(prompts, context_length);
}

std::vector<std::vector<uint8_t>> OpenVocabDetectionTask::preprocess(const std::vector<cv::Mat>& imgs) {
    if (imgs.empty() || imgs[0].empty()) {
        throw std::invalid_argument("Empty input image provided");
    }

    const cv::Mat& image = imgs[0];
    std::vector<std::vector<uint8_t>> results;
    if (model_info_.input_shapes.empty()) {
        results.push_back(image_preprocessor_->preprocess(image));
        return results;
    }

    results.reserve(model_info_.input_shapes.size());

    std::vector<int32_t> input_ids;
    std::vector<int32_t> attention_mask;
    bool needs_text_inputs = false;

    // Determine text-input context length and encode prompts
    for (size_t index = 0; index < model_info_.input_shapes.size(); ++index) {
        const std::string input_name = index < model_info_.input_names.size() ? model_info_.input_names[index] : "";
        const std::string normalized = normalizeModelName(input_name);
        if (normalized.find("inputids") != std::string::npos || normalized.find("attentionmask") != std::string::npos) {
            needs_text_inputs = true;
            const int context_length = !model_info_.input_shapes[index].empty()
                                           ? static_cast<int>(model_info_.input_shapes[index].back())
                                           : config_.max_text_queries;

            if (model_type_ == ModelType::GroundingDino && bert_tokenizer_) {
                // Grounding DINO: encode all phrases as a single concatenated sequence
                const std::vector<std::string> prompts = extractPrompts(config_);
                if (!prompts.empty()) {
                    auto encoded = bert_tokenizer_->encodePhrases(prompts, context_length);
                    input_ids = std::move(encoded.input_ids);
                    attention_mask = std::move(encoded.attention_mask);
                }
            } else {
                // OWL-ViT / OWLv2: encode each prompt independently (CLIP tokenizer)
                const auto encoded = encodePrompts(context_length);
                input_ids = encoded.first;
                attention_mask = encoded.second;
            }
            break;
        }
    }

    if (!needs_text_inputs && !extractPrompts(config_).empty()) {
        if (model_type_ == ModelType::GroundingDino && bert_tokenizer_) {
            auto encoded = bert_tokenizer_->encodePhrases(extractPrompts(config_), config_.max_text_queries);
            input_ids = std::move(encoded.input_ids);
            attention_mask = std::move(encoded.attention_mask);
        } else if (tokenizer_) {
            const auto encoded = encodePrompts(config_.max_text_queries);
            input_ids = encoded.first;
            attention_mask = encoded.second;
        }
    }

    for (size_t index = 0; index < model_info_.input_shapes.size(); ++index) {
        const auto& input_shape = model_info_.input_shapes[index];
        const std::string input_name = index < model_info_.input_names.size() ? model_info_.input_names[index] : "";
        const std::string normalized = normalizeModelName(input_name);

        if (isImageInput(input_name, input_shape)) {
            results.push_back(image_preprocessor_->preprocess(image));
        } else if (normalized.find("inputids") != std::string::npos) {
            results.push_back(toByteBuffer(input_ids));
        } else if (normalized.find("attentionmask") != std::string::npos) {
            results.push_back(toByteBuffer(attention_mask));
        } else if (normalized.find("tokentypeids") != std::string::npos) {
            // All-zero token type IDs required by some models (e.g. Grounding DINO)
            results.push_back(toByteBuffer(std::vector<int32_t>(input_ids.size(), 0)));
        } else {
            results.emplace_back();
        }
    }

    return results;
}

std::vector<Result> OpenVocabDetectionTask::postprocess(const cv::Size& frame_size,
                                                        const std::vector<Tensor>& tensors) {
    std::vector<Result> results;
    if (!postprocessor_) {
        return results;
    }

    for (auto& detection : postprocessor_->postprocess(tensors, frame_size)) {
        results.emplace_back(std::move(detection));
    }
    return results;
}

} // namespace vision_core

#include "neuriplo/tasks/keypoint_matching/keypoint_matching_task.hpp"

#include "neuriplo/tasks/keypoint_matching/keypoint_matching_preprocessor.hpp"
#include "neuriplo/tasks/keypoint_matching/lightglue_postprocessor.hpp"
#include "neuriplo/tasks/keypoint_matching/matchanything_postprocessor.hpp"
#include "neuriplo/tasks/keypoint_matching/superglue_postprocessor.hpp"

#include <cctype>
#include <stdexcept>

namespace neuriplo_tasks {

namespace {

std::string normalizeModelName(const std::string& model_name) {
    std::string normalized;
    normalized.reserve(model_name.size());
    for (char character : model_name) {
        if (std::isspace(static_cast<unsigned char>(character)) != 0 || character == '-' || character == '_') {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
    }
    return normalized;
}

bool startsWith(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

} // namespace

KeypointMatchingTask::KeypointMatchingTask(const ModelInfo& model_info, const std::string& model_name)
    : TaskInterface(model_info), model_type_(detectModelType(model_name)) {
    const Size input_size = extractInputSize(model_info);
    input_width_ = input_size.width;
    input_height_ = input_size.height;

    preprocessor_ = createPreprocessor(model_type_, input_size);
    if (!preprocessor_) {
        throw std::runtime_error("Failed to create preprocessor for keypoint matching model: " + model_name);
    }
    preprocessor_->applyImageInputType(model_info);

    postprocessor_ = createPostprocessor(model_type_);
    if (!postprocessor_) {
        throw std::runtime_error("Failed to create postprocessor for keypoint matching model: " + model_name);
    }
}

KeypointMatchingTask::~KeypointMatchingTask() = default;

std::vector<std::vector<uint8_t>> KeypointMatchingTask::preprocess(const std::vector<Image>& imgs) {
    if (imgs.size() != 2) {
        throw std::invalid_argument("Keypoint matching requires exactly 2 images (one pair)");
    }
    if (imgs[0].empty() || imgs[1].empty()) {
        throw std::invalid_argument("Empty image provided in pair");
    }

    pair_sizes_[0] = imgs[0].size();
    pair_sizes_[1] = imgs[1].size();

    return preprocessor_->preprocess_pair(imgs[0].view(), imgs[1].view());
}

std::vector<Result> KeypointMatchingTask::postprocess(const Size& frame_size, const std::vector<Tensor>& tensors) {
    if (tensors.empty()) {
        return {};
    }

    // preprocess() records the pair's original sizes; tensor-only callers that
    // never ran preprocessing fall back to frame_size for both images.
    const Size source_size = pair_sizes_[0].isEmpty() ? frame_size : pair_sizes_[0];
    const Size target_size = pair_sizes_[1].isEmpty() ? frame_size : pair_sizes_[1];

    std::vector<Result> results;
    results.emplace_back(postprocessor_->postprocess(tensors, source_size, target_size));
    return results;
}

KeypointMatchingTask::ModelType KeypointMatchingTask::detectModelType(const std::string& model_name) {
    const std::string normalized = normalizeModelName(model_name);

    if (startsWith(normalized, "superglue")) {
        return ModelType::SUPERGLUE;
    }
    if (startsWith(normalized, "lightglue")) {
        return ModelType::LIGHTGLUE;
    }
    if (startsWith(normalized, "matchanything")) {
        return ModelType::MATCH_ANYTHING;
    }

    throw std::invalid_argument("Unsupported keypoint matching model type: " + model_name);
}

std::unique_ptr<KeypointMatchingPreprocessor> KeypointMatchingTask::createPreprocessor(ModelType type,
                                                                                       const Size& input_size) {
    switch (type) {
    case ModelType::SUPERGLUE:
    case ModelType::LIGHTGLUE:
    case ModelType::MATCH_ANYTHING:
        return std::make_unique<KeypointMatchingPreprocessor>(input_size);
    }
    return nullptr;
}

std::unique_ptr<KeypointMatchingPostprocessor> KeypointMatchingTask::createPostprocessor(ModelType type) {
    switch (type) {
    case ModelType::SUPERGLUE:
        return std::make_unique<SuperGluePostprocessor>();
    case ModelType::LIGHTGLUE:
        return std::make_unique<LightGluePostprocessor>();
    case ModelType::MATCH_ANYTHING:
        return std::make_unique<MatchAnythingPostprocessor>();
    }
    return nullptr;
}

Size KeypointMatchingTask::extractInputSize(const ModelInfo& model_info) {
    int width = 512;
    int height = 512;

    if (!model_info.input_shapes.empty()) {
        const auto& shape = model_info.input_shapes[0];
        const bool is_nhwc = !model_info.input_formats.empty() && model_info.input_formats[0] == "FORMAT_NHWC";

        if (shape.size() == 4) {
            height = static_cast<int>(is_nhwc ? shape[1] : shape[2]);
            width = static_cast<int>(is_nhwc ? shape[2] : shape[3]);
        } else if (shape.size() == 3) {
            height = static_cast<int>(is_nhwc ? shape[0] : shape[1]);
            width = static_cast<int>(is_nhwc ? shape[1] : shape[2]);
        }
    }

    return Size(width, height);
}

} // namespace neuriplo_tasks

#include "neuriplo/tasks/video_classification/video_classification_task.hpp"

#include "neuriplo/tasks/video_classification/video_classification_postprocessor.hpp"
#include "neuriplo/tasks/video_classification/video_classification_preprocessor.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace neuriplo_tasks {

VideoClassificationTask::VideoClassificationTask(const ModelInfo& model_info, const std::string& model_name, int top_k,
                                                 bool apply_softmax)
    : TaskInterface(model_info), model_type_(detectModelType(model_name)), model_name_(model_name), top_k_(top_k),
      apply_softmax_(apply_softmax), num_frames_(16) {
    input_width_ = 224;
    input_height_ = 224;
    extractVideoInputSize(model_info);

    vision::Size input_size(input_width_, input_height_);
    preprocessor_ = createPreprocessor(model_type_, input_size);

    if (!preprocessor_) {
        throw std::runtime_error("Failed to create preprocessor for video classifier: " + model_name);
    }

    postprocessor_ = createPostprocessor();

    if (!postprocessor_) {
        throw std::runtime_error("Failed to create postprocessor for video classifier: " + model_name);
    }
}

VideoClassificationTask::~VideoClassificationTask() = default;

std::vector<std::vector<uint8_t>> VideoClassificationTask::preprocess(const std::vector<vision::Image>& imgs) {
    if (imgs.empty()) {
        throw std::invalid_argument("Empty input frame set provided");
    }

    // Process each frame through the preprocessor and concatenate into a single buffer
    std::vector<uint8_t> concatenated;
    const size_t frame_size =
        static_cast<size_t>(input_width_) * static_cast<size_t>(input_height_) * 3 * sizeof(float);
    concatenated.reserve(static_cast<size_t>(num_frames_) * frame_size);

    for (int i = 0; i < num_frames_; ++i) {
        // If fewer frames provided than needed, repeat the last frame
        const vision::Image& frame = (i < static_cast<int>(imgs.size())) ? imgs[static_cast<size_t>(i)] : imgs.back();

        if (frame.empty()) {
            throw std::invalid_argument("Empty frame provided at index " + std::to_string(i));
        }

        auto frame_data = preprocessor_->preprocess(frame.view());
        concatenated.insert(concatenated.end(), frame_data.begin(), frame_data.end());
    }

    return {concatenated};
}

std::vector<Result> VideoClassificationTask::postprocess(const vision::Size&, const std::vector<Tensor>& tensors) {

    if (tensors.empty()) {
        return {};
    }

    auto classifications = postprocessor_->postprocess(tensors[0].data, tensors[0].shape);

    std::vector<Result> results;
    results.reserve(classifications.size());
    for (const auto& classification : classifications) {
        results.emplace_back(classification);
    }

    return results;
}

VideoClassificationTask::ModelType VideoClassificationTask::detectModelType(const std::string& model_name) {
    std::string lower_name = model_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });

    if (lower_name.find("vivit") != std::string::npos) {
        return ModelType::VIVIT;
    }
    if (lower_name.find("timesformer") != std::string::npos) {
        return ModelType::TIMESFORMER;
    }
    // Default to VideoMAE
    return ModelType::VIDEOMAE;
}

std::unique_ptr<Preprocessor> VideoClassificationTask::createPreprocessor(ModelType type,
                                                                          const vision::Size& input_size) {
    switch (type) {
    case ModelType::VIDEOMAE:
        return std::make_unique<VideoMAEPreprocessor>(input_size);
    case ModelType::VIVIT:
        return std::make_unique<VivitPreprocessor>(input_size);
    case ModelType::TIMESFORMER:
        return std::make_unique<TimeSformerPreprocessor>(input_size);
    }
    return nullptr;
}

std::unique_ptr<VideoClassificationPostprocessor> VideoClassificationTask::createPostprocessor() {
    return std::make_unique<VideoClassificationPostprocessor>(top_k_, apply_softmax_);
}

void VideoClassificationTask::extractVideoInputSize(const ModelInfo& model_info) {
    if (model_info.input_shapes.empty()) {
        return; // Use defaults
    }

    const auto& shape = model_info.input_shapes[0];

    // 5D shape: [batch, num_frames, channels, height, width]
    if (shape.size() == 5) {
        num_frames_ = static_cast<int>(shape[1]);
        input_height_ = static_cast<int>(shape[3]);
        input_width_ = static_cast<int>(shape[4]);
    }
    // 4D shape fallback: [batch, channels, height, width] (treat as single frame)
    else if (shape.size() == 4) {
        num_frames_ = 1;
        if (!model_info.input_formats.empty() && model_info.input_formats[0] == "FORMAT_NHWC") {
            input_height_ = static_cast<int>(shape[1]);
            input_width_ = static_cast<int>(shape[2]);
        } else {
            input_height_ = static_cast<int>(shape[2]);
            input_width_ = static_cast<int>(shape[3]);
        }
    }
}

} // namespace neuriplo_tasks

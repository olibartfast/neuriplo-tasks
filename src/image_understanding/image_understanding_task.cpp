#include "vision-core/image_understanding/image_understanding_task.hpp"

namespace vision_core {

ImageUnderstandingTask::ImageUnderstandingTask(const ModelInfo& model_info, const std::string& /*model_name*/,
                                               const TaskConfig& config)
    : TaskInterface(model_info) {
    const auto it = config.extra_params.find("prompt");
    prompt_ = (it != config.extra_params.end() && !it->second.empty()) ? it->second : "Describe what you see.";
}

std::vector<std::vector<uint8_t>> ImageUnderstandingTask::preprocess(const std::vector<cv::Mat>& imgs) {
    std::vector<uint8_t> prompt_bytes(prompt_.begin(), prompt_.end());

    if (imgs.empty() || imgs[0].empty()) {
        return {std::move(prompt_bytes)};
    }

    // Encode image as: [nx(4B LE)][ny(4B LE)][RGB pixels]
    cv::Mat rgb;
    cv::cvtColor(imgs[0], rgb, cv::COLOR_BGR2RGB);
    const uint32_t nx = static_cast<uint32_t>(rgb.cols);
    const uint32_t ny = static_cast<uint32_t>(rgb.rows);

    std::vector<uint8_t> image_bytes(8 + static_cast<size_t>(nx) * ny * 3);
    std::memcpy(image_bytes.data() + 0, &nx, 4);
    std::memcpy(image_bytes.data() + 4, &ny, 4);

    if (rgb.isContinuous()) {
        std::memcpy(image_bytes.data() + 8, rgb.data, static_cast<size_t>(nx) * ny * 3);
    } else {
        for (uint32_t row = 0; row < ny; ++row) {
            std::memcpy(image_bytes.data() + 8 + row * nx * 3, rgb.ptr(static_cast<int>(row)), nx * 3);
        }
    }

    return {std::move(prompt_bytes), std::move(image_bytes)};
}

std::vector<Result> ImageUnderstandingTask::postprocess(const cv::Size& /*frame_size*/,
                                                        const std::vector<Tensor>& tensors) {
    if (tensors.empty() || tensors[0].data.empty()) {
        return {ImageUnderstanding{"(no response)"}};
    }
    return {ImageUnderstanding{decodeFloatBytes(tensors[0].data)}};
}

std::string ImageUnderstandingTask::decodeFloatBytes(const std::vector<TensorElement>& floats) {
    std::string result;
    result.reserve(floats.size());
    for (const auto& elem : floats) {
        result.push_back(static_cast<char>(static_cast<unsigned char>(std::get<float>(elem))));
    }
    return result;
}

} // namespace vision_core

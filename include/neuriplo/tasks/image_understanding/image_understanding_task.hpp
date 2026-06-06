#pragma once

#include "neuriplo/tasks/core/task_config.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <string>

namespace neuriplo_tasks {

/**
 * @brief Task for image understanding / text generation via LLM/VLM backends.
 *
 * Routes a text prompt (and optionally an image) through an LLM backend such
 * as llama.cpp.  The llamacpp backend encodes responses as float-cast bytes;
 * preprocess() packs the prompt into raw bytes, postprocess() decodes them
 * back into a human-readable string and returns an ImageUnderstanding result.
 */
class ImageUnderstandingTask : public TaskInterface {
  public:
    ImageUnderstandingTask(const ModelInfo& model_info, const std::string& model_name, const TaskConfig& config);

    TaskType getTaskType() override { return TaskType::ImageUnderstanding; }

    /**
     * @brief Encode the text prompt and (optionally) an image for the backend.
     *
     * Returns one tensor when no image is provided (text-only):
     *   [0] = UTF-8 prompt bytes
     *
     * Returns two tensors when an image is present (VLM / multimodal):
     *   [0] = UTF-8 prompt bytes
     *   [1] = 4-byte little-endian width | 4-byte little-endian height | RGB pixels
     */
    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;

    /**
     * @brief Decode float-encoded response bytes into an ImageUnderstanding result.
     */
    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<Tensor>& tensors) override;

  private:
    std::string prompt_;

    static std::string decodeFloatBytes(const std::vector<TensorElement>& floats);
};

} // namespace neuriplo_tasks

#pragma once

#include "vision-core/core/model_info.hpp"
#include "vision-core/core/result_types.hpp"

#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <tuple>
#include <variant>
#include <vector>

namespace vision_core {

/**
 * @brief Tensor element variant type
 * Supports common inference engine output types
 */
using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

/**
 * @brief Tensor structure combining data and shape
 */
struct Tensor {
    std::vector<TensorElement> data;
    std::vector<int64_t> shape;

    Tensor() = default;
    Tensor(std::vector<TensorElement> data_, std::vector<int64_t> shape_)
        : data(std::move(data_)), shape(std::move(shape_)) {}
};

/**
 * @brief Input dimension error exception
 */
class InputDimensionError : public std::runtime_error {
  public:
    explicit InputDimensionError(const std::string& what_arg) : std::runtime_error(what_arg) {}
};

/**
 * @brief Base task interface
 *
 * Provides a unified interface for preprocessing and postprocessing
 * across different computer vision tasks.
 */
class TaskInterface {
  public:
    explicit TaskInterface(const ModelInfo& model_info);
    virtual ~TaskInterface() = default;

    /**
     * @brief Get the task type
     */
    virtual TaskType getTaskType() = 0;

    /**
     * @brief Get the number of frames required by this task
     * @return Number of frames needed for inference (default: 1)
     */
    virtual int getRequiredFrames() const { return 1; }

    /**
     * @brief Preprocess input images
     * @param imgs Input images
     * @return Preprocessed data as uint8_t vectors
     */
    virtual std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) = 0;

    /**
     * @brief Postprocess inference results
     * @param frame_size Original frame size
     * @param tensors Inference output tensors with shape information
     * @return Vector of Result variants
     */
    virtual std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<Tensor>& tensors) = 0;

    /**
     * @brief Read label names from file
     * @param file_name Path to labels file
     * @return Vector of class names
     */
    [[nodiscard]] std::vector<std::string> readLabelNames(const std::string& file_name) const {
        std::vector<std::string> classes;
        std::ifstream ifs(file_name.c_str());
        std::string line;
        while (std::getline(ifs, line)) {
            classes.push_back(line);
        }
        return classes;
    }

  protected:
    ModelInfo model_info_;
    int input_width_{0};
    int input_height_{0};
    int input_channels_{0};

  private:
    /**
     * @brief Initialize input dimensions from model info
     */
    [[nodiscard]] std::tuple<int, int, int> initializeInputDimensions(const ModelInfo& model_info) const {
        for (size_t i = 0; i < model_info.input_shapes.size(); i++) {
            const auto& shape = model_info.input_shapes[i];
            const bool is_nhwc = (model_info.input_formats[i] == "FORMAT_NHWC");

            if (shape.size() == 3) {
                // Case for 3D inputs: CHW (default) or HWC
                if (is_nhwc) {
                    // HWC: 0->H, 1->W, 2->C
                    return std::make_tuple(static_cast<int>(shape[1]), static_cast<int>(shape[0]),
                                           static_cast<int>(shape[2]));
                }
                // CHW: 0->C, 1->H, 2->W
                return std::make_tuple(static_cast<int>(shape[2]), static_cast<int>(shape[1]),
                                       static_cast<int>(shape[0]));
            } else if (shape.size() >= 4) {
                // Case for 4D inputs: NCHW (default) or NHWC
                int channels = static_cast<int>(is_nhwc ? shape[3] : shape[1]);
                int height = static_cast<int>(is_nhwc ? shape[1] : shape[2]);
                int width = static_cast<int>(is_nhwc ? shape[2] : shape[3]);
                return std::make_tuple(width, height, channels);
            }
        }
        throw InputDimensionError("No valid input shape found");
    }
};

} // namespace vision_core

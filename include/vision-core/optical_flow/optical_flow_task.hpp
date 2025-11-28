#pragma once

#include "vision-core/core/task_interface.hpp"
#include "vision-core/core/preprocessor.hpp"
#include <memory>
#include <string>

namespace vision_core {

/**
 * @brief Unified optical flow task for all optical flow architectures
 * 
 * Handles all optical flow model types with consistent architecture:
 * - RAFT (Recurrent All-Pairs Field Transforms)
 * - Future optical flow models
 * 
 * Design principles:
 * 1. Single task class for all optical flow models
 * 2. Model type enum for different architectures
 * 3. Specialized preprocessing for frame pairs
 * 4. Unified output format with flow fields and visualizations
 */
class OpticalFlowTask : public TaskInterface {
public:
    /**
     * @brief Optical flow model types
     */
    enum class ModelType {
        RAFT           // RAFT-based optical flow models
    };

    /**
     * @brief Create optical flow task for specified model type
     */
    explicit OpticalFlowTask(const ModelInfo& model_info, 
                           const std::string& model_name);

    // TaskInterface implementation
    TaskType getTaskType() override { return TaskType::OpticalFlow; }
    
    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;
    
    std::vector<Result> postprocess(
        const cv::Size& frame_size,
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes) override;

private:
    ModelType model_type_;
    std::string model_name_;
    std::unique_ptr<Preprocessor> preprocessor_;
    int input_width_;
    int input_height_;

    /**
     * @brief Detect model type from model name string
     */
    static ModelType detectModelType(const std::string& model_name);
    
    /**
     * @brief Create appropriate preprocessor for model type
     */
    std::unique_ptr<Preprocessor> createPreprocessor(ModelType type, const cv::Size& input_size);
    
    /**
     * @brief Extract input dimensions from model info
     */
    cv::Size extractInputSize(const ModelInfo& model_info);
    
    /**
     * @brief Internal optical flow postprocessing
     */
    std::vector<OpticalFlow> postprocessRAFT(
        const std::vector<TensorElement>& flow_output,
        const std::vector<int64_t>& shape,
        const cv::Size& frame_size);
    
    /**
     * @brief Helper to get float value from tensor element
     */
    float getTensorFloat(const TensorElement& element);
    
    /**
     * @brief Generate flow visualization
     */
    cv::Mat visualizeFlow(const cv::Mat& flow_x, const cv::Mat& flow_y);
};

} // namespace vision_core
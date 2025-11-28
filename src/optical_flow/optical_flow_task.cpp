#include "vision-core/optical_flow/optical_flow_task.hpp"
#include "vision-core/optical_flow/optical_flow_preprocessor.hpp"
#include "vision-core/core/task_factory.hpp"
#include <algorithm>
#include <stdexcept>

namespace vision_core {

OpticalFlowTask::OpticalFlowTask(const ModelInfo& model_info, 
                                const std::string& model_name)
    : TaskInterface(model_info)
    , model_type_(detectModelType(model_name))
    , model_name_(model_name)
{
    // Extract input dimensions
    cv::Size input_size = extractInputSize(model_info);
    input_width_ = input_size.width;
    input_height_ = input_size.height;
    
    // Create appropriate preprocessor
    preprocessor_ = createPreprocessor(model_type_, input_size);
    
    if (!preprocessor_) {
        throw std::runtime_error("Failed to create preprocessor for optical flow model: " + model_name);
    }
}

std::vector<std::vector<uint8_t>> OpticalFlowTask::preprocess(const std::vector<cv::Mat>& imgs) {
    if (imgs.size() < 2) {
        throw std::invalid_argument("Optical flow requires at least 2 frames");
    }
    
    if (imgs.size() % 2 != 0) {
        throw std::invalid_argument("Optical flow requires even number of frames (frame pairs)");
    }
    
    std::vector<std::vector<uint8_t>> results;
    
    // Process consecutive frame pairs
    for (size_t i = 0; i < imgs.size(); i += 2) {
        if (imgs[i].empty() || imgs[i + 1].empty()) {
            throw std::invalid_argument("Empty frame provided in pair");
        }
        
        // Use RAFT's specialized preprocessing for frame pairs
        auto raft_preprocessor = static_cast<RaftPreprocessor*>(preprocessor_.get());
        auto pair_results = raft_preprocessor->preprocess_pair(imgs[i], imgs[i + 1]);
        
        // Add each result from the pair
        for (const auto& result : pair_results) {
            results.push_back(result);
        }
    }
    
    return results;
}

std::vector<Result> OpticalFlowTask::postprocess(
    const cv::Size& frame_size,
    const std::vector<std::vector<TensorElement>>& infer_results,
    const std::vector<std::vector<int64_t>>& infer_shapes) {
    
    if (infer_results.empty() || infer_shapes.empty()) {
        return {};
    }
    
    std::vector<OpticalFlow> flows;
    
    // Route to appropriate postprocessor based on model type
    switch (model_type_) {
        case ModelType::RAFT: {
            flows = postprocessRAFT(infer_results[0], infer_shapes[0], frame_size);
            break;
        }
        
        default:
            throw std::runtime_error("Unsupported optical flow model type for: " + model_name_);
    }
    
    // Convert flows to results
    std::vector<Result> results;
    results.reserve(flows.size());
    for (const auto& flow : flows) {
        results.emplace_back(flow);
    }
    
    return results;
}

OpticalFlowTask::ModelType OpticalFlowTask::detectModelType(const std::string& model_name) {
    std::string lower_name = model_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    // All current models are RAFT-based
    return ModelType::RAFT;
}

std::unique_ptr<Preprocessor> OpticalFlowTask::createPreprocessor(ModelType type, const cv::Size& input_size) {
    switch (type) {
        case ModelType::RAFT:
            return std::make_unique<RaftPreprocessor>(input_size);
            
        default:
            return nullptr;
    }
}

cv::Size OpticalFlowTask::extractInputSize(const ModelInfo& model_info) {
    int width = 512;  // default for RAFT
    int height = 384; // default for RAFT
    
    if (!model_info.input_shapes.empty() && model_info.input_shapes[0].size() >= 3) {
        const auto& shape = model_info.input_shapes[0];
        if (model_info.input_formats[0] == "FORMAT_NCHW") {
            height = static_cast<int>(shape[2]);
            width = static_cast<int>(shape[3]);
        } else if (model_info.input_formats[0] == "FORMAT_NHWC") {
            height = static_cast<int>(shape[1]);
            width = static_cast<int>(shape[2]);
        }
    }
    
    return cv::Size(width, height);
}

std::vector<OpticalFlow> OpticalFlowTask::postprocessRAFT(
    const std::vector<TensorElement>& flow_output,
    const std::vector<int64_t>& shape,
    const cv::Size& frame_size) {
    
    std::vector<OpticalFlow> flows;
    // TODO: Implement RAFT optical flow postprocessing
    return flows;
}

float OpticalFlowTask::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

cv::Mat OpticalFlowTask::visualizeFlow(const cv::Mat& flow_x, const cv::Mat& flow_y) {
    // TODO: Implement flow visualization
    return cv::Mat();
}

// Registration function for all optical flow models
void registerOpticalFlowTasks() {
    // Optical flow variants
    std::vector<std::string> optical_flow_variants = {
        "raft"
    };
    
    // Register all variants with unified OpticalFlowTask
    for (const auto& variant : optical_flow_variants) {
        TaskFactory::registerTask(variant, [variant](const ModelInfo& info) -> std::unique_ptr<TaskInterface> {
            return std::make_unique<OpticalFlowTask>(info, variant);
        });
    }
}

} // namespace vision_core
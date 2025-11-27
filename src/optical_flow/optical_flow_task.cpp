#include "vision-core/core/task_interface.hpp"
#include "vision-core/core/task_factory.hpp"
#include "vision-core/optical_flow/optical_flow_preprocessor.hpp"
#include "vision-core/optical_flow/raft_postprocessor.hpp"
#include <memory>
#include <stdexcept>

namespace vision_core {

/**
 * @brief Optical flow task implementation
 * 
 * Handles RAFT (Recurrent All-Pairs Field Transforms) optical flow estimation.
 * Requires two consecutive frames as input and produces optical flow fields
 * with both raw flow data and colorized visualization.
 */
class OpticalFlowTask : public TaskInterface {
private:
    std::unique_ptr<RaftPreprocessor> preprocessor_;

public:
    explicit OpticalFlowTask(const ModelInfo& model_info)
        : TaskInterface(model_info) {
        
        if (input_width_ <= 0 || input_height_ <= 0 || input_channels_ != 3) {
            throw InputDimensionError("Invalid optical flow model dimensions: expected 3-channel input");
        }
        
        preprocessor_ = std::make_unique<RaftPreprocessor>(cv::Size(input_width_, input_height_));
    }

    TaskType getTaskType() override {
        return TaskType::OpticalFlow;
    }

    /**
     * @brief Preprocess frame pairs for optical flow
     * 
     * Expects input images in pairs: [frame1, frame2, frame3, frame4, ...]
     * Processes consecutive pairs: (frame1,frame2), (frame3,frame4), etc.
     */
    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override {
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
            
            auto pair_results = preprocessor_->preprocess_pair(imgs[i], imgs[i + 1]);
            
            // Concatenate the two frames' data
            std::vector<uint8_t> combined_data;
            for (const auto& frame_data : pair_results) {
                combined_data.insert(combined_data.end(), frame_data.begin(), frame_data.end());
            }
            results.push_back(std::move(combined_data));
        }
        
        return results;
    }

    std::vector<Result> postprocess(
        const cv::Size& frame_size,
        const std::vector<std::vector<TensorElement>>& infer_results,
        const std::vector<std::vector<int64_t>>& infer_shapes) override {
        
        if (infer_results.empty() || infer_shapes.empty()) {
            return {};
        }

        if (infer_results.size() != infer_shapes.size()) {
            throw std::invalid_argument("Mismatch between inference results and shapes");
        }

        std::vector<Result> results;
        
        for (size_t i = 0; i < infer_results.size(); ++i) {
            if (infer_results[i].empty() || infer_shapes[i].empty()) {
                continue;
            }
            
            const TensorElement* output_data = infer_results[i].data();
            
            auto flow_result = RaftPostprocessor::postprocess(
                output_data,
                infer_shapes[i],
                frame_size
            );
            
            // Convert RaftPostprocessor result to standard OpticalFlow result
            OpticalFlow optical_flow;
            optical_flow.raw_flow = flow_result.raw_flow.clone();
            optical_flow.flow = flow_result.flow_visualization.clone();
            optical_flow.max_displacement = static_cast<float>(flow_result.max_displacement);
            
            results.emplace_back(std::move(optical_flow));
        }
        
        return results;
    }
};

// Explicit registration function for optical flow tasks
void registerOpticalFlowTasks() {
    std::vector<std::string> optical_flow_variants = {
        "raft"
    };
    
    for (const auto& variant : optical_flow_variants) {
        TaskFactory::registerTask(variant, [](const ModelInfo& model_info) -> std::unique_ptr<TaskInterface> {
            return std::make_unique<OpticalFlowTask>(model_info);
        });
    }
}

// Factory function for manual registration
std::unique_ptr<TaskInterface> createOpticalFlowTask(const ModelInfo& model_info) {
    return std::make_unique<OpticalFlowTask>(model_info);
}

} // namespace vision_core
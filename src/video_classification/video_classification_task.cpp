#include "vision-core/core/task_interface.hpp"
#include "vision-core/core/task_factory.hpp"
#include "vision-core/classification/classification_preprocessor.hpp"
#include "vision-core/video_classification/timesformer_postprocessor.hpp"
#include <memory>
#include <stdexcept>

namespace vision_core {

/**
 * @brief Video classification task implementation
 * 
 * Handles TimeSformer and similar temporal transformer models for video classification.
 * Processes sequences of frames and applies temporal attention mechanisms.
 * Returns action/activity classification results.
 */
class VideoClassificationTask : public TaskInterface {
private:
    std::unique_ptr<ClassifierPreprocessor> preprocessor_;
    int top_k_;
    bool apply_softmax_;
    size_t expected_frames_;

public:
    explicit VideoClassificationTask(const ModelInfo& model_info,
                                     int top_k = 5,
                                     bool apply_softmax = true,
                                     size_t expected_frames = 8)
        : TaskInterface(model_info)
        , top_k_(top_k)
        , apply_softmax_(apply_softmax)
        , expected_frames_(expected_frames) {
        
        if (input_width_ <= 0 || input_height_ <= 0 || input_channels_ != 3) {
            throw InputDimensionError("Invalid video classification model dimensions: expected 3-channel input");
        }
        
        // Use ImageNet normalization for video models (similar to ViT)
        preprocessor_ = std::make_unique<ClassifierPreprocessor>(
            cv::Size(input_width_, input_height_), 
            true  // use ImageNet normalization
        );
    }

    TaskType getTaskType() override {
        return TaskType::VideoClassification;
    }

    /**
     * @brief Preprocess video frames
     * 
     * Expects a sequence of frames that will be processed as a temporal sequence.
     * Frames are typically sampled uniformly from a video clip.
     */
    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override {
        if (imgs.empty()) {
            throw std::invalid_argument("Video classification requires at least 1 frame");
        }
        
        // For video models, we typically need to batch all frames together
        std::vector<std::vector<uint8_t>> frame_results;
        frame_results.reserve(imgs.size());
        
        for (const auto& img : imgs) {
            if (img.empty()) {
                throw std::invalid_argument("Empty frame provided in video sequence");
            }
            frame_results.push_back(preprocessor_->preprocess(img));
        }
        
        // Concatenate all frames into a single tensor
        // Video models expect input shape: [batch, frames, channels, height, width]
        // Or flattened: [batch, frames * channels * height * width]
        if (frame_results.size() != expected_frames_) {
            // Log warning but continue - model might handle variable frame counts
            // throw std::invalid_argument("Expected " + std::to_string(expected_frames_) + 
            //                           " frames, got " + std::to_string(frame_results.size()));
        }
        
        std::vector<uint8_t> concatenated_data;
        for (const auto& frame_data : frame_results) {
            concatenated_data.insert(concatenated_data.end(), frame_data.begin(), frame_data.end());
        }
        
        return {concatenated_data};  // Return single batch containing all frames
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
            
            auto video_classifications = TimesformerPostprocessor::postprocess(
                output_data,
                infer_shapes[i],
                top_k_,
                apply_softmax_
            );
            
            // Convert TimeSformer results to standard VideoClassification results
            for (const auto& video_result : video_classifications) {
                VideoClassification video_classification;
                video_classification.class_id = static_cast<float>(video_result.class_id);
                video_classification.class_confidence = video_result.confidence;
                video_classification.action_label = ""; // Would need label mapping
                // video_classification.frame_scores = {}; // Per-frame scores if available
                
                results.emplace_back(std::move(video_classification));
            }
        }
        
        return results;
    }
};

// Explicit registration function for video classification tasks
void registerVideoClassificationTasks() {
    std::vector<std::string> video_variants = {
        "timesformer", "video-classification", "videoclassification", "action-recognition",
        "activity-recognition", "video-transformer", "videotransformer",
        "timesformer-base", "timesformer-large", "timesformer-huge"
    };
    
    for (const auto& variant : video_variants) {
        TaskFactory::registerTask(variant, [](const ModelInfo& model_info) -> std::unique_ptr<TaskInterface> {
            return std::make_unique<VideoClassificationTask>(model_info);
        });
    }
}

// Factory function for manual registration
std::unique_ptr<TaskInterface> createVideoClassificationTask(const ModelInfo& model_info) {
    return std::make_unique<VideoClassificationTask>(model_info);
}

} // namespace vision_core
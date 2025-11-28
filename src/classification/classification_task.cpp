#include "vision-core/classification/classification_task.hpp"
#include "vision-core/classification/classification_preprocessor.hpp"
#include "vision-core/core/task_factory.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace vision_core {

ClassificationTask::ClassificationTask(const ModelInfo& model_info, 
                                     const std::string& model_name,
                                     int top_k,
                                     bool apply_softmax)
    : TaskInterface(model_info)
    , model_type_(detectModelType(model_name))
    , model_name_(model_name)
    , top_k_(top_k)
    , apply_softmax_(apply_softmax) 
{
    // Extract input dimensions
    cv::Size input_size = extractInputSize(model_info);
    input_width_ = input_size.width;
    input_height_ = input_size.height;
    
    // Create appropriate preprocessor
    preprocessor_ = createPreprocessor(model_type_, input_size);
    
    if (!preprocessor_) {
        throw std::runtime_error("Failed to create preprocessor for classifier: " + model_name);
    }
}

std::vector<std::vector<uint8_t>> ClassificationTask::preprocess(const std::vector<cv::Mat>& imgs) {
    std::vector<std::vector<uint8_t>> results;
    results.reserve(imgs.size());
    
    for (const auto& img : imgs) {
        if (img.empty()) {
            throw std::invalid_argument("Empty input image provided");
        }
        results.push_back(preprocessor_->preprocess(img));
    }
    
    return results;
}

std::vector<Result> ClassificationTask::postprocess(
    const cv::Size&, // frame_size unused for classification
    const std::vector<std::vector<TensorElement>>& infer_results,
    const std::vector<std::vector<int64_t>>& infer_shapes) {
    
    if (infer_results.empty() || infer_shapes.empty()) {
        return {};
    }
    
    // Classify using unified postprocessing
    auto classifications = postprocessClassification(infer_results[0], infer_shapes[0]);
    
    // Convert to results
    std::vector<Result> results;
    results.reserve(classifications.size());
    for (const auto& classification : classifications) {
        results.emplace_back(classification);
    }
    
    return results;
}

ClassificationTask::ModelType ClassificationTask::detectModelType(const std::string& model_name) {
    std::string lower_name = model_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    // Video classification models
    if (lower_name.find("timesformer") != std::string::npos ||
        lower_name.find("video") != std::string::npos) {
        return ModelType::VIDEO_CLASSIFIER;
    }
    
    // ViT models
    if (lower_name.find("vit") != std::string::npos ||
        lower_name == "vit-classifier") {
        return ModelType::VIT;
    }
    
    // TensorFlow models
    if (lower_name == "tensorflow-classifier" ||
        lower_name.find("tensorflow") != std::string::npos) {
        return ModelType::TENSORFLOW;
    }
    
    // Default to Torchvision (ResNet, etc.)
    return ModelType::TORCHVISION;
}

std::unique_ptr<Preprocessor> ClassificationTask::createPreprocessor(ModelType type, const cv::Size& input_size) {
    switch (type) {
        case ModelType::TORCHVISION:
            return std::make_unique<TorchvisionPreprocessor>(input_size);
            
        case ModelType::TENSORFLOW:
            return std::make_unique<TensorflowPreprocessor>(input_size);
            
        case ModelType::VIT:
            return std::make_unique<ViTPreprocessor>(input_size);
            
        case ModelType::VIDEO_CLASSIFIER:
            return std::make_unique<ClassifierPreprocessor>(input_size, true); // ImageNet norm
            
        default:
            return nullptr;
    }
}

cv::Size ClassificationTask::extractInputSize(const ModelInfo& model_info) {
    int width = 224;  // default for most classifiers
    int height = 224; // default for most classifiers
    
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

std::vector<Classification> ClassificationTask::postprocessClassification(
    const std::vector<TensorElement>& output,
    const std::vector<int64_t>& shape) {
    
    if (output.empty() || shape.empty()) {
        return {};
    }
    
    // Extract logits/probabilities
    std::vector<float> scores;
    scores.reserve(output.size());
    for (const auto& element : output) {
        scores.push_back(getTensorFloat(element));
    }
    
    // Apply softmax if needed
    if (apply_softmax_) {
        applySoftmax(scores);
    }
    
    // Get top-k classifications
    std::vector<std::pair<int, float>> indexed_scores;
    for (size_t i = 0; i < scores.size(); ++i) {
        indexed_scores.emplace_back(static_cast<int>(i), scores[i]);
    }
    
    // Sort by score (highest first)
    std::partial_sort(indexed_scores.begin(), 
                     indexed_scores.begin() + std::min(top_k_, static_cast<int>(indexed_scores.size())),
                     indexed_scores.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Create classifications
    std::vector<Classification> classifications;
    int limit = std::min(top_k_, static_cast<int>(indexed_scores.size()));
    for (int i = 0; i < limit; ++i) {
        Classification cls;
        cls.class_id = indexed_scores[i].first;
        cls.class_confidence = indexed_scores[i].second;
        classifications.push_back(cls);
    }
    
    return classifications;
}

float ClassificationTask::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

void ClassificationTask::applySoftmax(std::vector<float>& logits) {
    if (logits.empty()) return;
    
    // Find max value for numerical stability
    float max_val = *std::max_element(logits.begin(), logits.end());
    
    // Compute softmax
    float sum = 0.0f;
    for (auto& logit : logits) {
        logit = std::exp(logit - max_val);
        sum += logit;
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (auto& logit : logits) {
            logit /= sum;
        }
    }
}

// Registration function for all classification models
void registerClassificationTasks() {
    // Classification variants
    std::vector<std::string> classification_variants = {
        "torchvision-classifier", "tensorflow-classifier", "vit-classifier"
    };
    
    // Video classification variants  
    std::vector<std::string> video_classification_variants = {
        "timesformer", "video-classifier"
    };
    
    // Register all variants with unified ClassificationTask
    for (const auto& variant : classification_variants) {
        TaskFactory::registerTask(variant, [variant](const ModelInfo& info) -> std::unique_ptr<TaskInterface> {
            return std::make_unique<ClassificationTask>(info, variant);
        });
    }
    
    for (const auto& variant : video_classification_variants) {
        TaskFactory::registerTask(variant, [variant](const ModelInfo& info) -> std::unique_ptr<TaskInterface> {
            return std::make_unique<ClassificationTask>(info, variant);
        });
    }
}

} // namespace vision_core
#pragma once

#include "neuriplo/tasks/core/task_interface.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace neuriplo_tasks {

class KeypointMatchingPostprocessor;
class KeypointMatchingPreprocessor;

/**
 * @brief Unified keypoint matching task
 *
 * Matches keypoints between the two images of a pair (one pair per call) using
 * end-to-end exports that consume the image pair and emit keypoints plus
 * matches.
 *
 * Supported model families: SuperGlue-style pipelines, LightGlue (including
 * the LightGlue-ONNX pipeline exports), FlexGlue (descriptor + sparse
 * matcher), and MatchAnything (EfficientLoFTR / RoMa backbones).
 *
 * Matcher-only exports (keypoints and descriptors in, matches out) cannot be
 * driven through TaskInterface's image contract and are out of scope.
 * FlexGlue is the exception with an explicit forwarding contract:
 * preprocess() emits the descriptor inputs, and postprocess() takes the
 * matcher outputs plus the externally detected keypoints forwarded as
 * tensors, because the matcher returns indices only.
 *
 * SuperGlue, LightGlue, and FlexGlue decoding are implemented against real
 * ONNX exports. The MatchAnything family postprocessor still throws
 * std::logic_error until its output layout is verified against a real ONNX
 * export.
 */
class KeypointMatchingTask : public TaskInterface {
  public:
    enum class ModelType : uint8_t {
        SUPERGLUE,      ///< SuperGlue-style end-to-end pipeline
        LIGHTGLUE,      ///< LightGlue end-to-end pipeline
        MATCH_ANYTHING, ///< MatchAnything (EfficientLoFTR / RoMa) pipeline
        FLEXGLUE,       ///< FlexGlue descriptor + sparse matcher (forwarded keypoints)
    };

    /**
     * @brief Construct a keypoint matching task
     *
     * @param model_info Model input/output metadata
     * @param model_name Model family string (e.g. "superglue", "lightglue_v2")
     * @param match_score_threshold Matches with a score at or below this value
     *        are dropped (TaskConfig::confidence_threshold in TaskFactory)
     */
    explicit KeypointMatchingTask(const ModelInfo& model_info, const std::string& model_name,
                                  float match_score_threshold = 0.0f);
    ~KeypointMatchingTask() override;

    TaskType getTaskType() override { return TaskType::KeypointMatching; }

    /// Keypoint matching consumes one image pair per call.
    [[nodiscard]] int getRequiredFrames() const override { return 2; }

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<Image>& imgs) override;

    std::vector<Result> postprocess(const Size& frame_size, const std::vector<Tensor>& tensors) override;

  private:
    ModelType model_type_;
    float match_score_threshold_{0.0f};
    /// Input size the model runs at (LightGlue keypoints are in this space).
    Size model_input_size_{512, 512};
    std::unique_ptr<KeypointMatchingPreprocessor> preprocessor_;
    std::unique_ptr<KeypointMatchingPostprocessor> postprocessor_;
    /// Original sizes of the last preprocessed pair, in call order.
    std::array<Size, 2> pair_sizes_{};

    static ModelType detectModelType(const std::string& model_name);

    std::unique_ptr<KeypointMatchingPreprocessor> createPreprocessor(ModelType type, const Size& input_size);
    std::unique_ptr<KeypointMatchingPostprocessor> createPostprocessor(ModelType type);
    Size extractInputSize(const ModelInfo& model_info);
};

} // namespace neuriplo_tasks

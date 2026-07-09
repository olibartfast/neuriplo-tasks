#pragma once

#include "neuriplo/tasks/core/preprocessor.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"
#include "neuriplo/tasks/gaussian_splatting/gaussian_splatting_postprocessor.hpp"

#include <memory>
#include <string>

namespace neuriplo_tasks {

/**
 * @brief Unified Gaussian Splatting task
 *
 * Supports feed-forward Gaussian Splatting models that map one or more
 * input images to a set of 3D Gaussian primitives (positions, rotations,
 * scales, opacities, and SH colours).
 *
 * Supported model families:
 * - LGM / LGM-mini  (dylanebert/LGM)
 * - GRM             (GaussianWorld/GRM)
 *
 * Output: GaussianSplatting result containing all predicted Gaussian3D
 * primitives.  Rendering (.ply serialisation, rasterisation) is intentionally
 * left to the caller so that the framework remains rendering-backend agnostic.
 */
class GaussianSplattingTask : public TaskInterface {
  public:
    enum class ModelType : uint8_t {
        LGM, ///< LGM / LGM-mini (dylanebert/LGM)
        GRM, ///< GRM (GaussianWorld/GRM)
    };

    explicit GaussianSplattingTask(const ModelInfo& model_info, const std::string& model_name);
    ~GaussianSplattingTask() override;

    TaskType getTaskType() override { return TaskType::GaussianSplatting; }

    /**
     * @brief Number of input views required for this model
     *
     * LGM/GRM expect 4 input views produced by a multi-view diffusion stage.
     * Single-image models return 1.
     */
    [[nodiscard]] int getRequiredFrames() const override;

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<Image>& imgs) override;

    std::vector<Result> postprocess(const Size& frame_size, const std::vector<Tensor>& tensors) override;

  private:
    ModelType model_type_;
    std::string model_name_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<GaussianSplattingPostprocessor> postprocessor_;

    static ModelType detectModelType(const std::string& model_name);

    std::unique_ptr<Preprocessor> createPreprocessor(ModelType type, const Size& input_size);
    std::unique_ptr<GaussianSplattingPostprocessor> createPostprocessor(ModelType type);
    Size extractInputSize(const ModelInfo& model_info);
};

} // namespace neuriplo_tasks

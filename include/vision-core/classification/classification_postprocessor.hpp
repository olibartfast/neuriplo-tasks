#pragma once

#include "vision-core/core/result_types.hpp"
#include "vision-core/core/task_interface.hpp"

#include <vector>

namespace vision_core {

/**
 * @brief Interface for classification postprocessing
 */
class ClassificationPostprocessor {
  public:
    virtual ~ClassificationPostprocessor() = default;

    /**
     * @brief Postprocess inference results
     * @param output Inference output tensor
     * @param shape Output tensor shape
     * @return Vector of Classification results
     */
    virtual std::vector<Classification> postprocess(const std::vector<TensorElement>& output,
                                                    const std::vector<int64_t>& shape) = 0;
};

/**
 * @brief Default implementation for standard classification models
 */
class DefaultClassificationPostprocessor : public ClassificationPostprocessor {
  public:
    DefaultClassificationPostprocessor(int top_k, bool apply_softmax);

    std::vector<Classification> postprocess(const std::vector<TensorElement>& output,
                                            const std::vector<int64_t>& shape) override;

  private:
    int top_k_;
    bool apply_softmax_;

    void applySoftmax(std::vector<float>& logits);
};

} // namespace vision_core

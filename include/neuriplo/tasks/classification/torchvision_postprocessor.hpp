#pragma once

#include "neuriplo/tasks/classification/classification_postprocessor.hpp"

namespace neuriplo_tasks {

class TorchvisionPostprocessor : public ClassificationPostprocessor {
  public:
    TorchvisionPostprocessor(int top_k, bool apply_softmax);

    std::vector<Classification> postprocess(const std::vector<TensorElement>& output,
                                            const std::vector<int64_t>& shape) override;

  private:
    int top_k_;
    bool apply_softmax_;

    void applySoftmax(std::vector<float>& logits);
};

} // namespace neuriplo_tasks

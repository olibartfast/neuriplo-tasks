#include "neuriplo/tasks/classification/vit_postprocessor.hpp"

#include "neuriplo/tasks/classification/classification_postprocessor.hpp"

namespace neuriplo_tasks {

ViTPostprocessor::ViTPostprocessor(int top_k, bool apply_softmax) : top_k_(top_k), apply_softmax_(apply_softmax) {}

std::vector<Classification> ViTPostprocessor::postprocess(const std::vector<TensorElement>& output,
                                                          const std::vector<int64_t>& shape) {
    return postprocessClassificationLogits(output, shape, top_k_, apply_softmax_);
}

void ViTPostprocessor::applySoftmax(std::vector<float>& logits) { (void)logits; }

} // namespace neuriplo_tasks

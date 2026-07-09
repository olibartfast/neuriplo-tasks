#pragma once

#include "neuriplo/tasks/open_vocab_detection/postprocessor.hpp"

#include <string>
#include <vector>

namespace neuriplo_tasks {

class OWLv2Postprocessor : public OpenVocabPostprocessor {
  public:
    OWLv2Postprocessor(const Size& input_size, float confidence_threshold, float text_threshold,
                       std::vector<std::string> prompt_labels, std::vector<std::string> output_names);

    std::vector<OpenVocabDetection> postprocess(const std::vector<Tensor>& tensors, const Size& frame_size) override;

  private:
    Size input_size_;
    float confidence_threshold_;
    float text_threshold_;
    std::vector<std::string> prompt_labels_;
    std::vector<std::string> output_names_;
};

} // namespace neuriplo_tasks

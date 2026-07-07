#pragma once

#include "neuriplo/tasks/open_vocab_detection/postprocessor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Postprocessor for Grounding DINO open-vocabulary detection
 *
 * Grounding DINO outputs token-level logits of shape [batch, num_queries, seq_len].
 * For each detected query box this postprocessor:
 *   1. Applies sigmoid to every logit in the token dimension.
 *   2. For each input phrase, computes the maximum sigmoid score over that
 *      phrase's token range.
 *   3. Assigns the box to the phrase with the highest score.
 *   4. Filters by confidence_threshold and text_threshold.
 *
 * Expected tensor names (checked against output_names, case-insensitive after
 * stripping '-' / '_'):
 *   - pred_boxes  (or boxes)       – shape [batch, num_queries, 4], normalised cx cy w h
 *   - pred_logits (or logits)      – shape [batch, num_queries, seq_len]
 */
class GroundingDinoPostprocessor : public OpenVocabPostprocessor {
  public:
    /**
     * @param input_size           Model input spatial dimensions
     * @param confidence_threshold Minimum detection score to retain a box
     * @param text_threshold       Additional minimum per-phrase token score
     * @param prompt_labels        Original text prompt strings (one per phrase)
     * @param output_names         Model output tensor names (matched against standard names)
     * @param phrase_token_ranges  [start, end) indices in the tokenised sequence
     *                             for each prompt phrase (end is exclusive)
     */
    GroundingDinoPostprocessor(const Size& input_size, float confidence_threshold, float text_threshold,
                               std::vector<std::string> prompt_labels, std::vector<std::string> output_names,
                               std::vector<std::pair<int, int>> phrase_token_ranges);

    std::vector<OpenVocabDetection> postprocess(const std::vector<Tensor>& tensors, const Size& frame_size) override;

  private:
    Size input_size_;
    float confidence_threshold_;
    float text_threshold_;
    std::vector<std::string> prompt_labels_;
    std::vector<std::string> output_names_;
    std::vector<std::pair<int, int>> phrase_token_ranges_;
};

} // namespace neuriplo_tasks

#pragma once

#include "neuriplo/tasks/keypoint_matching/keypoint_matching_postprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief FlexGlue sparse-matcher postprocessor
 *
 * Decodes the `flexglue_matcher.onnx` sparse outputs (verified against the
 * v0.1 release; see `export/keypoint_matching/probe_flexglue.py`):
 *
 * - `kpts0`, `kpts1` [1, N, 2] float32 — corner-origin pixels of the original
 *   images. These are matcher *inputs*; the consumer forwards the keypoints
 *   its detector produced (in order: source, target) alongside the matcher
 *   outputs because the matcher returns indices only, not coordinates.
 * - `matches0` [1, N0] int64 — for each source keypoint, the target keypoint
 *   index it matches, or -1 when unmatched. N0 equals the kpts0 count.
 * - `mscores0` [1, N0] float32 — match confidence, 0.0 where unmatched.
 * - `covisibility_score` [1] float32 — optional pair overlap probability.
 *   Gating on `min_covisibility` already happened inside the graph, so this
 *   tensor is informational only here.
 *
 * Tensors are identified by shape and element type, not by position, except
 * the two keypoint tensors, which are positional (first forwarded is kpts0).
 * Keypoints pass through unscaled — they already live in each original
 * image's pixel space, unlike SuperGlue (normalized) or LightGlue
 * (model-input space). Matches are kept only when the score is above
 * `score_threshold`, which the task fills from
 * `TaskConfig::confidence_threshold`.
 *
 * The matcher carries no detector score, so `Keypoint::confidence` stays at
 * its default value; use `KeypointMatch::score` for match quality.
 */
class FlexGluePostprocessor : public KeypointMatchingPostprocessor {
  public:
    explicit FlexGluePostprocessor(float score_threshold = 0.0f);

    KeypointMatching postprocess(const std::vector<Tensor>& tensors, const Size& source_size,
                                 const Size& target_size) override;

  private:
    float score_threshold_{0.0f};
};

} // namespace neuriplo_tasks

#pragma once

#include "neuriplo/tasks/core/task_interface.hpp"
#include "neuriplo/tasks/monocular_3d_detection/monocular_3d_detection_types.hpp"

#include <cstddef>
#include <vector>

namespace neuriplo_tasks {

/**
 * @brief UrbanOmniDetect-style monocular 3D cuboid postprocessor.
 *
 * Decodes the single YOLO-pose-style output tensor (Ultralytics export,
 * `output0`), i.e. one detection row = box plus eight cuboid corners:
 * - Channels-first `[B, C, N]` (YOLOv8/v11 style, no objectness):
 *   `cx, cy, w, h` (input pixels), `nc` class scores, then 8×`(x, y, conf)`
 *   corner triples in input pixels, with `C = 4 + nc + 24` (`nc >= 1`).
 * - Rows `[B, N, C]` (YOLOv5 style, with objectness):
 *   `cx, cy, w, h, obj`, `nc` class scores, then 8×`(x, y, conf)`.
 * - Rows `[B, N, 30]` end-to-end (UrbanOmniDetect-2 / YOLO26 style):
 *   `x1, y1, x2, y2, score, class_id`, then 8×`(x, y, conf)`.
 *
 * Boxes and corners are mapped back to the original frame with the YOLO
 * letterbox inverse. Corner triples carry their own confidence; the mean
 * becomes `geometry_confidence`. Detections below `confidence_threshold`
 * are dropped, then class-agnostic NMS with `nms_threshold` runs.
 *
 * Corner order is preserved as emitted. Released UrbanOmniDetect (v1) and
 * UrbanOmniDetect-2 checkpoints both emit ground-first corners (indices 0-3
 * ground, verified empirically on the released weights and in the upstream
 * runtime — the paper text states the opposite). The optional `top_first`
 * remap covers paper-text-conformant checkpoints by swapping halves 0-3/4-7
 * into the canonical ground-first layout.
 * Malformed shapes return empty.
 */
class Monocular3DPostprocessor {
  public:
    Monocular3DPostprocessor(const Size& input_size, float confidence_threshold, float nms_threshold,
                             bool end_to_end = false, bool top_first = false);

    [[nodiscard]] std::vector<Monocular3DDetection> postprocess(const std::vector<Tensor>& tensors,
                                                                const Size& original_size) const;

  private:
    Size input_size_;
    float confidence_threshold_;
    float nms_threshold_;
    bool end_to_end_;
    bool top_first_;

    [[nodiscard]] BoundingBox scaleBoxToOriginal(float cx, float cy, float w, float h, const Size& frame_size) const;
    [[nodiscard]] std::pair<float, float> scalePointToOriginal(float x, float y, const Size& frame_size) const;
    void applyNMS(std::vector<Monocular3DDetection>& detections) const;
};

} // namespace neuriplo_tasks

#pragma once

#include "neuriplo/tasks/core/preprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief Unified YOLO preprocessor
 *
 * Handles letterbox resizing and normalization for ALL YOLO variants:
 * - YOLOv4 (Darknet-based)
 * - YOLOv5, v6, v7, v8, v9, v11, v12 (standard YOLO models)
 * - YOLOv10 (end-to-end detector)
 * - YOLO-NAS (neural architecture search YOLO)
 *
 * Uses consistent letterbox preprocessing across all variants.
 */
class YoloPreprocessor : public Preprocessor {
  public:
    explicit YoloPreprocessor(const Size& input_size);

    using Preprocessor::preprocess;
    [[nodiscard]] std::vector<uint8_t> preprocess(const ImageView& image) const override;
};

/**
 * @brief RT-DETR preprocessor (transformer-based detector)
 */
class RtDetrPreprocessor : public Preprocessor {
  public:
    explicit RtDetrPreprocessor(const Size& input_size);
};

/**
 * @brief EdgeCrafter preprocessor
 *
 * EdgeCrafter previously shared the RT-DETR preprocessor. It keeps ImageNet
 * normalization: no EdgeCrafter model was available to verify a change against,
 * so its behaviour is preserved rather than assumed to match RT-DETR's.
 */
class EdgeCrafterPreprocessor : public Preprocessor {
  public:
    explicit EdgeCrafterPreprocessor(const Size& input_size);
};

/**
 * @brief D-FINE preprocessor (DETR-based fine-grained detector)
 */
class DFinePreprocessor : public Preprocessor {
  public:
    explicit DFinePreprocessor(const Size& input_size);
};

/**
 * @brief RF-DETR preprocessor (receptive field DETR)
 */
class RfDetrPreprocessor : public Preprocessor {
  public:
    explicit RfDetrPreprocessor(const Size& input_size);
};

} // namespace neuriplo_tasks

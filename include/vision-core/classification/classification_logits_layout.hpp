#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vision_core {
namespace classification_detail {

struct LogitBatchLayout {
    int batch_size{0};
    int num_classes{0};
};

/**
 * @brief Parse [N, C] or [C] classification output layouts.
 * @return layout with batch_size=0 when shape/data are inconsistent.
 */
[[nodiscard]] inline LogitBatchLayout parseLogitBatchLayout(const std::vector<int64_t>& shape, size_t element_count) {
    LogitBatchLayout layout;
    if (shape.empty() || element_count == 0) {
        return layout;
    }

    if (shape.size() == 1) {
        layout.batch_size = 1;
        layout.num_classes = static_cast<int>(shape[0]);
        if (static_cast<size_t>(layout.num_classes) == element_count) {
            return layout;
        }
        return {};
    }

    if (shape.size() >= 2) {
        layout.batch_size = static_cast<int>(shape[0]);
        int64_t trailing = 1;
        for (size_t i = 1; i < shape.size(); ++i) {
            trailing *= shape[i];
        }
        layout.num_classes = static_cast<int>(trailing);
        if (layout.batch_size <= 0 || layout.num_classes <= 0) {
            return {};
        }
        const size_t expected = static_cast<size_t>(layout.batch_size) * static_cast<size_t>(layout.num_classes);
        if (expected <= element_count) {
            return layout;
        }
    }

    return {};
}

[[nodiscard]] inline size_t batchSliceOffset(int batch_index, const LogitBatchLayout& layout) noexcept {
    return static_cast<size_t>(batch_index) * static_cast<size_t>(layout.num_classes);
}

} // namespace classification_detail
} // namespace vision_core

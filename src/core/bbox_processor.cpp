#include "neuriplo/tasks/core/bbox_processor.hpp"

#include <algorithm>
#include <stdexcept>

namespace neuriplo_tasks {

vision::Rect BBoxProcessor::calculate_bounding_box(const vision::Size& image_size, const std::vector<float>& bbox,
                                                   int network_width, int network_height) {
    if (bbox.size() < 4) {
        throw std::invalid_argument("Bbox must have at least 4 elements");
    }

    const float ratio_width = static_cast<float>(network_width) / static_cast<float>(image_size.width);
    const float ratio_height = static_cast<float>(network_height) / static_cast<float>(image_size.height);

    int left = 0;
    int right = 0;
    int top = 0;
    int bottom = 0;

    if (ratio_height > ratio_width) {
        // Letterbox padding on top/bottom
        const float padding =
            (static_cast<float>(network_height) - ratio_width * static_cast<float>(image_size.height)) / 2.0f;

        left = static_cast<int>((bbox[0] - bbox[2] / 2.0f) / ratio_width);
        right = static_cast<int>((bbox[0] + bbox[2] / 2.0f) / ratio_width);
        top = static_cast<int>((bbox[1] - bbox[3] / 2.0f - padding) / ratio_width);
        bottom = static_cast<int>((bbox[1] + bbox[3] / 2.0f - padding) / ratio_width);
    } else {
        // Letterbox padding on left/right
        const float padding =
            (static_cast<float>(network_width) - ratio_height * static_cast<float>(image_size.width)) / 2.0f;

        left = static_cast<int>((bbox[0] - bbox[2] / 2.0f - padding) / ratio_height);
        right = static_cast<int>((bbox[0] + bbox[2] / 2.0f - padding) / ratio_height);
        top = static_cast<int>((bbox[1] - bbox[3] / 2.0f) / ratio_height);
        bottom = static_cast<int>((bbox[1] + bbox[3] / 2.0f) / ratio_height);
    }

    vision::Rect result(left, top, right - left, bottom - top);
    clamp_to_bounds(result, image_size);
    return result;
}

vision::Rect BBoxProcessor::calculate_bounding_box_from_xyxy(const vision::Size& image_size,
                                                             const std::vector<float>& bbox, int network_width,
                                                             int network_height) {
    if (bbox.size() < 4) {
        throw std::invalid_argument("Bbox must have at least 4 elements");
    }

    const float ratio_width = static_cast<float>(network_width) / static_cast<float>(image_size.width);
    const float ratio_height = static_cast<float>(network_height) / static_cast<float>(image_size.height);

    int left = 0;
    int right = 0;
    int top = 0;
    int bottom = 0;

    if (ratio_height > ratio_width) {
        // Letterbox padding on top/bottom
        const float padding =
            (static_cast<float>(network_height) - ratio_width * static_cast<float>(image_size.height)) / 2.0f;

        left = static_cast<int>(bbox[0] / ratio_width);
        top = static_cast<int>((bbox[1] - padding) / ratio_width);
        right = static_cast<int>(bbox[2] / ratio_width);
        bottom = static_cast<int>((bbox[3] - padding) / ratio_width);
    } else {
        // Letterbox padding on left/right
        const float padding =
            (static_cast<float>(network_width) - ratio_height * static_cast<float>(image_size.width)) / 2.0f;

        left = static_cast<int>((bbox[0] - padding) / ratio_height);
        top = static_cast<int>(bbox[1] / ratio_height);
        right = static_cast<int>((bbox[2] - padding) / ratio_height);
        bottom = static_cast<int>(bbox[3] / ratio_height);
    }

    vision::Rect result(left, top, right - left, bottom - top);
    clamp_to_bounds(result, image_size);
    return result;
}

vision::Rect BBoxProcessor::scale_to_original(const vision::Rect& bbox, const vision::Size& original_size,
                                              const vision::Size& network_size) noexcept {
    const float scale_width = static_cast<float>(original_size.width) / static_cast<float>(network_size.width);
    const float scale_height = static_cast<float>(original_size.height) / static_cast<float>(network_size.height);

    return vision::Rect(static_cast<int>(static_cast<float>(bbox.x) * scale_width),
                        static_cast<int>(static_cast<float>(bbox.y) * scale_height),
                        static_cast<int>(static_cast<float>(bbox.width) * scale_width),
                        static_cast<int>(static_cast<float>(bbox.height) * scale_height));
}

void BBoxProcessor::clamp_to_bounds(vision::Rect& box, const vision::Size& image_size) noexcept {
    box.x = std::clamp(box.x, 0, image_size.width - 1);
    box.y = std::clamp(box.y, 0, image_size.height - 1);
    box.width = std::clamp(box.width, 0, image_size.width - box.x);
    box.height = std::clamp(box.height, 0, image_size.height - box.y);
}

} // namespace neuriplo_tasks

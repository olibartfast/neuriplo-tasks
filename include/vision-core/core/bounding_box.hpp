#pragma once

#include <algorithm>

namespace vision_core {

/**
 * @brief Axis-aligned bounding box in pixel coordinates.
 *
 * Mirrors cv::Rect field layout without pulling OpenCV into public headers.
 */
struct BoundingBox {
    int x{0};
    int y{0};
    int width{0};
    int height{0};

    BoundingBox() = default;
    BoundingBox(int x_, int y_, int width_, int height_) : x(x_), y(y_), width(width_), height(height_) {}

    [[nodiscard]] int area() const noexcept { return width * height; }

    [[nodiscard]] BoundingBox intersect(const BoundingBox& other) const noexcept {
        const int x1 = std::max(x, other.x);
        const int y1 = std::max(y, other.y);
        const int x2 = std::min(x + width, other.x + other.width);
        const int y2 = std::min(y + height, other.y + other.height);
        const int intersect_width = std::max(0, x2 - x1);
        const int intersect_height = std::max(0, y2 - y1);
        return BoundingBox(x1, y1, intersect_width, intersect_height);
    }

    friend bool operator==(const BoundingBox& lhs, const BoundingBox& rhs) noexcept {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.width == rhs.width && lhs.height == rhs.height;
    }
};

} // namespace vision_core

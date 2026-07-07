#pragma once

#include <algorithm>

namespace neuriplo_tasks::vision {

/**
 * @brief Two-dimensional size in pixels, replacing cv::Size in public headers.
 */
struct Size {
    int width{0};
    int height{0};

    Size() = default;
    Size(int w, int h) : width(w), height(h) {}

    [[nodiscard]] int area() const noexcept { return width * height; }
    [[nodiscard]] bool isEmpty() const noexcept { return width <= 0 || height <= 0; }

    friend bool operator==(const Size& lhs, const Size& rhs) noexcept {
        return lhs.width == rhs.width && lhs.height == rhs.height;
    }
    friend bool operator!=(const Size& lhs, const Size& rhs) noexcept { return !(lhs == rhs); }
};

/**
 * @brief Axis-aligned rectangle in pixel coordinates.
 */
struct Rect {
    int x{0};
    int y{0};
    int width{0};
    int height{0};

    Rect() = default;
    Rect(int x_, int y_, int width_, int height_) : x(x_), y(y_), width(width_), height(height_) {}

    [[nodiscard]] int area() const noexcept { return width * height; }

    [[nodiscard]] Rect intersect(const Rect& other) const noexcept {
        const int x1 = std::max(x, other.x);
        const int y1 = std::max(y, other.y);
        const int x2 = std::min(x + width, other.x + other.width);
        const int y2 = std::min(y + height, other.y + other.height);
        const int intersect_width = std::max(0, x2 - x1);
        const int intersect_height = std::max(0, y2 - y1);
        return Rect(x1, y1, intersect_width, intersect_height);
    }

    friend bool operator==(const Rect& lhs, const Rect& rhs) noexcept {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.width == rhs.width && lhs.height == rhs.height;
    }
};

/**
 * @brief 2D point with integer coordinates.
 */
struct Point {
    int x{0};
    int y{0};

    Point() = default;
    Point(int x_, int y_) : x(x_), y(y_) {}
};

/**
 * @brief 2D point with float coordinates, replacing cv::Point2f.
 */
struct Point2f {
    float x{0.0f};
    float y{0.0f};

    Point2f() = default;
    Point2f(float x_, float y_) : x(x_), y(y_) {}
};

} // namespace neuriplo_tasks::vision

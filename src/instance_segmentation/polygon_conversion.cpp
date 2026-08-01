#include "neuriplo/tasks/instance_segmentation/polygon_conversion.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace neuriplo_tasks {

namespace {

struct Edge {
    vision::Point start;
    vision::Point end;
    bool visited{false};
};

using Ring = std::vector<vision::Point2f>;

int64_t pointKey(const vision::Point& point) {
    return (static_cast<int64_t>(point.y) << 32) | static_cast<uint32_t>(point.x);
}

int direction(const Edge& edge) {
    if (edge.end.x > edge.start.x) {
        return 0;
    }
    if (edge.end.y > edge.start.y) {
        return 1;
    }
    if (edge.end.x < edge.start.x) {
        return 2;
    }
    return 3;
}

int turnRank(int current, int next) {
    const int turn = (next - current + 4) % 4;
    if (turn == 1) {
        return 0;
    }
    if (turn == 0) {
        return 1;
    }
    if (turn == 3) {
        return 2;
    }
    return 3;
}

Ring removeCollinearPoints(Ring ring) {
    if (ring.size() <= 3) {
        return ring;
    }
    Ring simplified;
    simplified.reserve(ring.size());
    for (size_t index = 0; index < ring.size(); ++index) {
        const auto& previous = ring[(index + ring.size() - 1) % ring.size()];
        const auto& current = ring[index];
        const auto& next = ring[(index + 1) % ring.size()];
        const float cross =
            (current.x - previous.x) * (next.y - current.y) - (current.y - previous.y) * (next.x - current.x);
        if (cross != 0.0f) {
            simplified.push_back(current);
        }
    }
    return simplified;
}

float signedArea(const Ring& ring) {
    float area = 0.0f;
    for (size_t index = 0; index < ring.size(); ++index) {
        const auto& current = ring[index];
        const auto& next = ring[(index + 1) % ring.size()];
        area += current.x * next.y - next.x * current.y;
    }
    return area * 0.5f;
}

float cross(const vision::Point2f& origin, const vision::Point2f& a, const vision::Point2f& b) {
    return (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);
}

Ring convexHull(Ring ring) {
    const bool positive_winding = signedArea(ring) > 0.0f;
    std::sort(ring.begin(), ring.end(),
              [](const auto& a, const auto& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
    ring.erase(
        std::unique(ring.begin(), ring.end(), [](const auto& a, const auto& b) { return a.x == b.x && a.y == b.y; }),
        ring.end());
    if (ring.size() < 3) {
        return {};
    }

    Ring hull;
    hull.reserve(ring.size() * 2);
    for (const auto& point : ring) {
        while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull.back(), point) <= 0.0f) {
            hull.pop_back();
        }
        hull.push_back(point);
    }
    const size_t lower_size = hull.size();
    for (auto iterator = ring.rbegin() + 1; iterator != ring.rend(); ++iterator) {
        while (hull.size() > lower_size && cross(hull[hull.size() - 2], hull.back(), *iterator) <= 0.0f) {
            hull.pop_back();
        }
        hull.push_back(*iterator);
    }
    hull.pop_back();
    if ((signedArea(hull) > 0.0f) != positive_winding) {
        std::reverse(hull.begin(), hull.end());
    }
    return hull;
}

bool containsPoint(const Ring& ring, const vision::Point2f& point) {
    bool inside = false;
    for (size_t current = 0, previous = ring.size() - 1; current < ring.size(); previous = current++) {
        const auto& a = ring[current];
        const auto& b = ring[previous];
        const bool crosses = (a.y > point.y) != (b.y > point.y);
        if (crosses && point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x) {
            inside = !inside;
        }
    }
    return inside;
}

std::vector<Edge> createBoundaryEdges(const ImageMatrix& mask) {
    const auto* pixels = mask.data();
    const int width = mask.cols();
    const int height = mask.rows();
    const auto foreground = [pixels, width, height](int x, int y) {
        return x >= 0 && x < width && y >= 0 && y < height && pixels[static_cast<size_t>(y * width + x)] != 0;
    };

    std::vector<Edge> edges;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (!foreground(x, y)) {
                continue;
            }
            if (!foreground(x, y - 1)) {
                edges.push_back({{x, y}, {x + 1, y}});
            }
            if (!foreground(x + 1, y)) {
                edges.push_back({{x + 1, y}, {x + 1, y + 1}});
            }
            if (!foreground(x, y + 1)) {
                edges.push_back({{x + 1, y + 1}, {x, y + 1}});
            }
            if (!foreground(x - 1, y)) {
                edges.push_back({{x, y + 1}, {x, y}});
            }
        }
    }
    return edges;
}

std::vector<Ring> traceRings(std::vector<Edge> edges) {
    std::unordered_map<int64_t, std::vector<size_t>> outgoing;
    for (size_t index = 0; index < edges.size(); ++index) {
        outgoing[pointKey(edges[index].start)].push_back(index);
    }

    std::vector<Ring> rings;
    for (size_t first = 0; first < edges.size(); ++first) {
        if (edges[first].visited) {
            continue;
        }
        Ring ring;
        size_t current = first;
        const vision::Point start = edges[first].start;
        while (!edges[current].visited) {
            Edge& edge = edges[current];
            edge.visited = true;
            ring.push_back({static_cast<float>(edge.start.x), static_cast<float>(edge.start.y)});
            if (edge.end.x == start.x && edge.end.y == start.y) {
                break;
            }

            const auto found = outgoing.find(pointKey(edge.end));
            if (found == outgoing.end()) {
                throw std::runtime_error("Mask boundary is not closed");
            }
            int best_rank = std::numeric_limits<int>::max();
            size_t best = edges.size();
            for (const size_t candidate : found->second) {
                if (!edges[candidate].visited) {
                    const int rank = turnRank(direction(edge), direction(edges[candidate]));
                    if (rank < best_rank) {
                        best_rank = rank;
                        best = candidate;
                    }
                }
            }
            if (best == edges.size()) {
                throw std::runtime_error("Mask boundary traversal is incomplete");
            }
            current = best;
        }
        ring = removeCollinearPoints(std::move(ring));
        if (ring.size() >= 3) {
            rings.push_back(std::move(ring));
        }
    }
    return rings;
}

} // namespace

std::vector<SegmentationPolygon> maskToPolygons(const ImageMatrix& mask) {
    if (mask.empty()) {
        return {};
    }
    if (mask.pixelType() != PixelType::UInt8 || mask.channels() != 1) {
        throw std::invalid_argument("Polygon conversion requires a single-channel UINT8 mask");
    }

    auto rings = traceRings(createBoundaryEdges(mask));
    for (auto& ring : rings) {
        ring = convexHull(std::move(ring));
    }
    std::vector<SegmentationPolygon> polygons;
    std::vector<Ring> holes;
    for (auto& ring : rings) {
        if (signedArea(ring) > 0.0f) {
            polygons.push_back({std::move(ring), {}});
        } else {
            holes.push_back(std::move(ring));
        }
    }

    for (auto& hole : holes) {
        size_t owner = polygons.size();
        float owner_area = std::numeric_limits<float>::max();
        for (size_t index = 0; index < polygons.size(); ++index) {
            const float area = signedArea(polygons[index].exterior);
            if (area < owner_area && containsPoint(polygons[index].exterior, hole.front())) {
                owner = index;
                owner_area = area;
            }
        }
        if (owner == polygons.size()) {
            std::reverse(hole.begin(), hole.end());
            polygons.push_back({std::move(hole), {}});
        } else {
            polygons[owner].holes.push_back(std::move(hole));
        }
    }
    return polygons;
}

} // namespace neuriplo_tasks

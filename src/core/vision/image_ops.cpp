#include "image_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace neuriplo_tasks::vision::ops {

namespace {

[[nodiscard]] inline double readScalar(const std::uint8_t* base, std::size_t idx, PixelType pt) noexcept {
    switch (pt) {
    case PixelType::UInt8:
        return static_cast<double>(base[idx]);
    case PixelType::Float32:
        return static_cast<double>(reinterpret_cast<const float*>(base)[idx]);
    case PixelType::Int32:
        return static_cast<double>(reinterpret_cast<const int32_t*>(base)[idx]);
    }
    return 0.0;
}

inline void writeScalar(std::uint8_t* base, std::size_t idx, PixelType pt, double v) noexcept {
    switch (pt) {
    case PixelType::UInt8: {
        const double clamped = std::clamp(v, 0.0, 255.0);
        base[idx] = static_cast<uint8_t>(clamped);
        break;
    }
    case PixelType::Float32:
        reinterpret_cast<float*>(base)[idx] = static_cast<float>(v);
        break;
    case PixelType::Int32:
        reinterpret_cast<int32_t*>(base)[idx] = static_cast<int32_t>(v);
        break;
    }
}

[[nodiscard]] inline std::size_t elemSize(PixelType pt) noexcept { return pixelTypeSize(pt); }

template <typename View> [[nodiscard]] Image resizeBilinear(const View& src, int dw, int dh) {
    const int sw = src.width();
    const int sh = src.height();
    const int c = src.channels();
    const PixelType pt = src.pixelType();
    const std::size_t es = elemSize(pt);
    const std::size_t s_width = static_cast<std::size_t>(sw);
    const std::size_t s_channels = static_cast<std::size_t>(c);
    Image out = Image::uninit(dw, dh, c, pt);
    const double sx = static_cast<double>(sw) / static_cast<double>(dw);
    const double sy = static_cast<double>(sh) / static_cast<double>(dh);
    const std::uint8_t* sptr = static_cast<const std::uint8_t*>(src.raw());
    std::uint8_t* dptr = out.raw();
    for (int dy = 0; dy < dh; ++dy) {
        const double fy = (static_cast<double>(dy) + 0.5) * sy - 0.5;
        int y0 = static_cast<int>(std::floor(fy));
        int y1 = y0 + 1;
        const double wy = fy - static_cast<double>(y0);
        const double wy0 = 1.0 - wy;
        y0 = std::clamp(y0, 0, sh - 1);
        y1 = std::clamp(y1, 0, sh - 1);
        const std::size_t y0s = static_cast<std::size_t>(y0);
        const std::size_t y1s = static_cast<std::size_t>(y1);
        for (int dx = 0; dx < dw; ++dx) {
            const double fx = (static_cast<double>(dx) + 0.5) * sx - 0.5;
            int x0 = static_cast<int>(std::floor(fx));
            int x1 = x0 + 1;
            const double wx = fx - static_cast<double>(x0);
            const double wx0 = 1.0 - wx;
            x0 = std::clamp(x0, 0, sw - 1);
            x1 = std::clamp(x1, 0, sw - 1);
            const std::size_t x0s = static_cast<std::size_t>(x0);
            const std::size_t x1s = static_cast<std::size_t>(x1);
            const std::size_t base00 = (y0s * s_width + x0s) * s_channels;
            const std::size_t base01 = (y0s * s_width + x1s) * s_channels;
            const std::size_t base10 = (y1s * s_width + x0s) * s_channels;
            const std::size_t base11 = (y1s * s_width + x1s) * s_channels;
            std::uint8_t* dst_pixel =
                dptr + (static_cast<std::size_t>(dy) * static_cast<std::size_t>(dw) + static_cast<std::size_t>(dx)) *
                           s_channels * es;
            for (int ch = 0; ch < c; ++ch) {
                const std::size_t chs = static_cast<std::size_t>(ch);
                const double v00 = readScalar(sptr, base00 + chs, pt);
                const double v01 = readScalar(sptr, base01 + chs, pt);
                const double v10 = readScalar(sptr, base10 + chs, pt);
                const double v11 = readScalar(sptr, base11 + chs, pt);
                const double v0 = v00 * wx0 + v01 * wx;
                const double v1 = v10 * wx0 + v11 * wx;
                const double v = v0 * wy0 + v1 * wy;
                writeScalar(dst_pixel, 0, pt, v);
                dst_pixel += es;
            }
        }
    }
    return out;
}

template <typename View> [[nodiscard]] Image resizeBicubic(const View& src, int dw, int dh) {
    // Bicubic kernel (a = -0.75), matching OpenCV INTER_CUBIC.
    const int sw = src.width();
    const int sh = src.height();
    const int c = src.channels();
    const PixelType pt = src.pixelType();
    const std::size_t es = elemSize(pt);
    const std::size_t s_width = static_cast<std::size_t>(sw);
    const std::size_t s_channels = static_cast<std::size_t>(c);
    Image out = Image::uninit(dw, dh, c, pt);
    const double sx = static_cast<double>(sw) / static_cast<double>(dw);
    const double sy = static_cast<double>(sh) / static_cast<double>(dh);
    const std::uint8_t* sptr = static_cast<const std::uint8_t*>(src.raw());
    std::uint8_t* dptr = out.raw();

    auto cubic_weight = [](double t) {
        const double a = -0.75;
        const double at = std::fabs(t);
        if (at < 1.0) {
            return (a + 2.0) * at * at * at - (a + 3.0) * at * at + 1.0;
        }
        if (at < 2.0) {
            return a * at * at * at - 5.0 * a * at * at + 8.0 * a * at - 4.0 * a;
        }
        return 0.0;
    };

    for (int dy = 0; dy < dh; ++dy) {
        const double fy = (static_cast<double>(dy) + 0.5) * sy - 0.5;
        int y_base = static_cast<int>(std::floor(fy));
        const double wy = fy - static_cast<double>(y_base);
        for (int dx = 0; dx < dw; ++dx) {
            const double fx = (static_cast<double>(dx) + 0.5) * sx - 0.5;
            int x_base = static_cast<int>(std::floor(fx));
            const double wx = fx - static_cast<double>(x_base);
            std::uint8_t* dst_pixel =
                dptr + (static_cast<std::size_t>(dy) * static_cast<std::size_t>(dw) + static_cast<std::size_t>(dx)) *
                           s_channels * es;
            for (int ch = 0; ch < c; ++ch) {
                double acc = 0.0;
                for (int j = -1; j <= 2; ++j) {
                    int yy = y_base + j;
                    yy = std::clamp(yy, 0, sh - 1);
                    const double wyv = cubic_weight(static_cast<double>(j) - wy);
                    for (int i = -1; i <= 2; ++i) {
                        int xx = x_base + i;
                        xx = std::clamp(xx, 0, sw - 1);
                        const double wxv = cubic_weight(static_cast<double>(i) - wx);
                        const std::size_t idx =
                            (static_cast<std::size_t>(yy) * s_width + static_cast<std::size_t>(xx)) * s_channels +
                            static_cast<std::size_t>(ch);
                        const double v = readScalar(sptr, idx, pt);
                        acc += v * wxv * wyv;
                    }
                }
                writeScalar(dst_pixel, 0, pt, acc);
                dst_pixel += es;
            }
        }
    }
    return out;
}

template <typename View> [[nodiscard]] Image resizeArea(const View& src, int dw, int dh) {
    // Area-pixel (box-filter) resampling, matching OpenCV INTER_AREA for
    // integer-ratio downsampling and approximating it for non-integer ratios.
    const int sw = src.width();
    const int sh = src.height();
    const int c = src.channels();
    const PixelType pt = src.pixelType();
    const std::size_t es = elemSize(pt);
    const std::size_t s_width = static_cast<std::size_t>(sw);
    const std::size_t s_channels = static_cast<std::size_t>(c);
    Image out = Image::uninit(dw, dh, c, pt);
    const std::uint8_t* sptr = static_cast<const std::uint8_t*>(src.raw());
    std::uint8_t* dptr = out.raw();
    const double sx = static_cast<double>(sw) / static_cast<double>(dw);
    const double sy = static_cast<double>(sh) / static_cast<double>(dh);

    for (int dy = 0; dy < dh; ++dy) {
        const int y0 = static_cast<int>(std::floor(static_cast<double>(dy) * sy));
        const int y1 = static_cast<int>(std::floor(static_cast<double>(dy + 1) * sy));
        const int ry0 = std::clamp(y0, 0, sh);
        const int ry1 = std::clamp(y1, 0, sh);
        for (int dx = 0; dx < dw; ++dx) {
            const int x0 = static_cast<int>(std::floor(static_cast<double>(dx) * sx));
            const int x1 = static_cast<int>(std::floor(static_cast<double>(dx + 1) * sx));
            const int rx0 = std::clamp(x0, 0, sw);
            const int rx1 = std::clamp(x1, 0, sw);
            const double area = static_cast<double>((rx1 - rx0)) * static_cast<double>((ry1 - ry0));
            std::uint8_t* dst_pixel =
                dptr + (static_cast<std::size_t>(dy) * static_cast<std::size_t>(dw) + static_cast<std::size_t>(dx)) *
                           s_channels * es;
            if (area <= 0.0) {
                for (int ch = 0; ch < c; ++ch) {
                    writeScalar(dst_pixel, 0, pt, 0.0);
                    dst_pixel += es;
                }
                continue;
            }
            for (int ch = 0; ch < c; ++ch) {
                double acc = 0.0;
                for (int yy = ry0; yy < ry1; ++yy) {
                    for (int xx = rx0; xx < rx1; ++xx) {
                        const std::size_t idx =
                            (static_cast<std::size_t>(yy) * s_width + static_cast<std::size_t>(xx)) * s_channels +
                            static_cast<std::size_t>(ch);
                        acc += readScalar(sptr, idx, pt);
                    }
                }
                writeScalar(dst_pixel, 0, pt, acc / area);
                dst_pixel += es;
            }
        }
    }
    return out;
}

} // namespace

Image resize(const Image& src, int dst_width, int dst_height, Interpolation interp) {
    return resize(src.view(), dst_width, dst_height, interp);
}

Image resize(const ImageView& src, int dst_width, int dst_height, Interpolation interp) {
    if (src.empty() || dst_width <= 0 || dst_height <= 0) {
        throw std::invalid_argument("resize: invalid source or destination geometry");
    }
    switch (interp) {
    case Interpolation::Area:
        return resizeArea(src, dst_width, dst_height);
    case Interpolation::Cubic:
        return resizeBicubic(src, dst_width, dst_height);
    case Interpolation::Linear:
    default:
        return resizeBilinear(src, dst_width, dst_height);
    }
}

void swapBgrRgb(Image& img) {
    if (img.channels() != 3) {
        return;
    }
    const std::size_t pixels = img.totalPixels();
    const std::size_t es = pixelTypeSize(img.pixelType());
    std::uint8_t* base = img.raw();
    for (std::size_t i = 0; i < pixels; ++i) {
        std::uint8_t* p = base + i * 3 * es;
        std::uint8_t* c0 = p;
        std::uint8_t* c2 = p + 2 * es;
        for (std::size_t b = 0; b < es; ++b) {
            std::swap(c0[b], c2[b]);
        }
    }
}

Image thresholdBinary(const ImageView& src, double threshold, double max_value) {
    if (src.channels() != 1) {
        throw std::invalid_argument("thresholdBinary expects a single-channel image");
    }
    Image out(src.width(), src.height(), 1, src.pixelType());
    const std::size_t n = src.totalPixels();
    const PixelType pt = src.pixelType();
    const std::uint8_t* sptr = static_cast<const std::uint8_t*>(src.raw());
    std::uint8_t* dptr = out.raw();
    for (std::size_t i = 0; i < n; ++i) {
        const double v = readScalar(sptr, i, pt);
        writeScalar(dptr, i, pt, (v > threshold) ? max_value : 0.0);
    }
    return out;
}

std::vector<Image> splitChannels(const ImageView& src) {
    const int c = src.channels();
    if (c <= 0) {
        return {};
    }
    const int w = src.width();
    const int h = src.height();
    const PixelType pt = src.pixelType();
    const std::size_t es = pixelTypeSize(pt);
    const std::size_t s_width = static_cast<std::size_t>(w);
    const std::size_t s_channels = static_cast<std::size_t>(c);
    std::vector<Image> out;
    out.reserve(static_cast<std::size_t>(c));
    for (int ch = 0; ch < c; ++ch) {
        out.push_back(Image::uninit(w, h, 1, pt));
    }
    const std::uint8_t* sptr = static_cast<const std::uint8_t*>(src.raw());
    for (int y = 0; y < h; ++y) {
        const std::size_t ys = static_cast<std::size_t>(y);
        for (int x = 0; x < w; ++x) {
            const std::size_t xs = static_cast<std::size_t>(x);
            for (int ch = 0; ch < c; ++ch) {
                const std::size_t src_idx = (ys * s_width + xs) * s_channels + static_cast<std::size_t>(ch);
                std::uint8_t* dst = out[static_cast<std::size_t>(ch)].ptr<std::uint8_t>(y) + xs * es;
                std::memcpy(dst, sptr + src_idx * es, es);
            }
        }
    }
    return out;
}

Image mergeChannels(const std::vector<Image>& channels) {
    if (channels.empty()) {
        throw std::invalid_argument("mergeChannels: no channels provided");
    }
    const int w = channels[0].width();
    const int h = channels[0].height();
    const PixelType pt = channels[0].pixelType();
    const int c = static_cast<int>(channels.size());
    const std::size_t es = pixelTypeSize(pt);
    const std::size_t s_width = static_cast<std::size_t>(w);
    const std::size_t s_channels = static_cast<std::size_t>(c);
    Image out = Image::uninit(w, h, c, pt);
    for (int ch = 0; ch < c; ++ch) {
        if (channels[static_cast<std::size_t>(ch)].width() != w ||
            channels[static_cast<std::size_t>(ch)].height() != h ||
            channels[static_cast<std::size_t>(ch)].pixelType() != pt) {
            throw std::invalid_argument("mergeChannels: channel geometry/type mismatch");
        }
        const std::size_t chs = static_cast<std::size_t>(ch);
        const std::uint8_t* sptr = channels[chs].raw();
        for (int y = 0; y < h; ++y) {
            const std::size_t ys = static_cast<std::size_t>(y);
            for (int x = 0; x < w; ++x) {
                const std::size_t xs = static_cast<std::size_t>(x);
                const std::size_t src_idx = ys * s_width + xs;
                std::uint8_t* dst = out.ptr<std::uint8_t>(y) + (xs * s_channels + chs) * es;
                std::memcpy(dst, sptr + src_idx * es, es);
            }
        }
    }
    return out;
}

void minMax(const ImageView& src, double& out_min, double& out_max) {
    if (src.channels() != 1) {
        throw std::invalid_argument("minMax expects a single-channel image");
    }
    const std::size_t n = src.totalPixels();
    const PixelType pt = src.pixelType();
    const std::uint8_t* sptr = static_cast<const std::uint8_t*>(src.raw());
    if (n == 0) {
        out_min = 0.0;
        out_max = 0.0;
        return;
    }
    double mn = readScalar(sptr, 0, pt);
    double mx = mn;
    for (std::size_t i = 1; i < n; ++i) {
        const double v = readScalar(sptr, i, pt);
        if (v < mn) {
            mn = v;
        }
        if (v > mx) {
            mx = v;
        }
    }
    out_min = mn;
    out_max = mx;
}

void copyRegion(const ImageView& src, const BoundingBox& src_roi, Image& dst, const BoundingBox& dst_roi) {
    if (src_roi.width != dst_roi.width || src_roi.height != dst_roi.height) {
        throw std::invalid_argument("copyRegion: source and destination ROI sizes must match");
    }
    const int c = src.channels();
    const PixelType pt = src.pixelType();
    if (dst.channels() != c || dst.pixelType() != pt) {
        throw std::invalid_argument("copyRegion: source and destination pixel layout must match");
    }
    const std::size_t es = pixelTypeSize(pt);
    const std::size_t s_channels = static_cast<std::size_t>(c);
    const std::size_t row_bytes = static_cast<std::size_t>(src_roi.width) * s_channels * es;
    const std::uint8_t* sptr = static_cast<const std::uint8_t*>(src.raw());
    for (int y = 0; y < src_roi.height; ++y) {
        const std::uint8_t* s = sptr + static_cast<std::size_t>(src_roi.y + y) * src.stride() +
                                static_cast<std::size_t>(src_roi.x) * s_channels * es;
        std::uint8_t* d = dst.ptr<std::uint8_t>(dst_roi.y + y) + static_cast<std::size_t>(dst_roi.x) * s_channels * es;
        std::memcpy(d, s, row_bytes);
    }
}

std::vector<int> nms(const std::vector<DetectionBox>& detections, float iou_threshold) {
    const std::size_t n = detections.size();
    std::vector<int> order(n);
    for (std::size_t i = 0; i < n; ++i) {
        order[i] = static_cast<int>(i);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return detections[static_cast<std::size_t>(a)].score > detections[static_cast<std::size_t>(b)].score;
    });

    std::vector<bool> suppressed(n, false);
    std::vector<int> keep;
    keep.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t idx = static_cast<std::size_t>(order[i]);
        if (suppressed[idx]) {
            continue;
        }
        keep.push_back(order[i]);
        for (std::size_t j = i + 1; j < n; ++j) {
            const std::size_t jdx = static_cast<std::size_t>(order[j]);
            if (suppressed[jdx]) {
                continue;
            }
            const BoundingBox inter = detections[idx].bbox.intersect(detections[jdx].bbox);
            const float inter_area = static_cast<float>(inter.area());
            const float union_area =
                static_cast<float>(detections[idx].bbox.area() + detections[jdx].bbox.area()) - inter_area;
            if (union_area > 0.0f && inter_area / union_area > iou_threshold) {
                suppressed[jdx] = true;
            }
        }
    }
    return keep;
}

} // namespace neuriplo_tasks::vision::ops

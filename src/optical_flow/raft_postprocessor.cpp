#include "neuriplo/tasks/optical_flow/raft_postprocessor.hpp"

#include "image_ops.hpp"
#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace neuriplo_tasks {

RaftPostprocessor::RaftPostprocessor() {}

std::vector<OpticalFlow> RaftPostprocessor::postprocess(const std::vector<TensorElement>& flow_output,
                                                        const std::vector<int64_t>& shape,
                                                        const vision::Size& frame_size) {

    if (flow_output.empty() || shape.empty()) {
        return {};
    }

    if (shape.size() < 4)
        return {};

    int channels = static_cast<int>(shape[1]);
    int height = static_cast<int>(shape[2]);
    int width = static_cast<int>(shape[3]);

    if (channels != 2)
        return {};

    const int u_channel_offset = 0;
    const int v_channel_offset = height * width;

    vision::Image flow_u = vision::Image::uninit(width, height, 1, vision::PixelType::Float32);
    vision::Image flow_v = vision::Image::uninit(width, height, 1, vision::PixelType::Float32);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            flow_u.ptr<float>(y)[x] =
                tensorElementToFloat(flow_output[static_cast<size_t>(u_channel_offset + y * width + x)]);
            flow_v.ptr<float>(y)[x] =
                tensorElementToFloat(flow_output[static_cast<size_t>(v_channel_offset + y * width + x)]);
        }
    }

    if (frame_size.width != width || frame_size.height != height) {
        flow_u = image_ops::resize(flow_u, frame_size.width, frame_size.height, image_ops::Interpolation::Linear);
        flow_v = image_ops::resize(flow_v, frame_size.width, frame_size.height, image_ops::Interpolation::Linear);

        float scale_u = static_cast<float>(frame_size.width) / static_cast<float>(width);
        float scale_v = static_cast<float>(frame_size.height) / static_cast<float>(height);

        float* pu = flow_u.data<float>();
        float* pv = flow_v.data<float>();
        const size_t total = flow_u.totalPixels();
        for (size_t i = 0; i < total; ++i) {
            pu[i] *= scale_u;
            pv[i] *= scale_v;
        }
    }

    const int fw = frame_size.width;
    const int fh = frame_size.height;

    vision::Image raw_flow = vision::Image::uninit(fw, fh, 2, vision::PixelType::Float32);
    for (int y = 0; y < fh; ++y) {
        float* dst = raw_flow.ptr<float>(y);
        const float* su = flow_u.ptr<float>(y);
        const float* sv = flow_v.ptr<float>(y);
        for (int x = 0; x < fw; ++x) {
            dst[x * 2] = su[x];
            dst[x * 2 + 1] = sv[x];
        }
    }

    OpticalFlow result;
    result.raw_flow = fromImage(raw_flow.clone());

    vision::Image magnitude = vision::Image::uninit(fw, fh, 1, vision::PixelType::Float32);
    for (int y = 0; y < fh; ++y) {
        const float* pu = flow_u.ptr<float>(y);
        const float* pv = flow_v.ptr<float>(y);
        float* pm = magnitude.ptr<float>(y);
        for (int x = 0; x < fw; ++x) {
            pm[x] = std::sqrt(pu[x] * pu[x] + pv[x] * pv[x]);
        }
    }

    double dummy_min = 0.0;
    double max_disp = 0.0;
    image_ops::minMax(magnitude.view(), dummy_min, max_disp);
    result.max_displacement = static_cast<float>(max_disp);

    result.flow = fromImage(visualizeFlow(flow_u.view(), flow_v.view()));

    return {result};
}

vision::Image RaftPostprocessor::makeColorwheel() {
    const int RY = 15, YG = 6, GC = 4, CB = 11, BM = 13, MR = 6;
    const int ncols = RY + YG + GC + CB + BM + MR;
    vision::Image colorwheel(ncols, 1, 3, vision::PixelType::UInt8);

    int col = 0;
    auto setPixel = [&colorwheel](int c, int b, int g, int r) {
        uint8_t* p = colorwheel.ptr<uint8_t>(0) + c * 3;
        p[0] = static_cast<uint8_t>(b);
        p[1] = static_cast<uint8_t>(g);
        p[2] = static_cast<uint8_t>(r);
    };

    for (int i = 0; i < RY; ++i, ++col) {
        setPixel(col, 255, 255 * i / RY, 0);
    }
    for (int i = 0; i < YG; ++i, ++col) {
        setPixel(col, 255 - 255 * i / YG, 255, 0);
    }
    for (int i = 0; i < GC; ++i, ++col) {
        setPixel(col, 0, 255, 255 * i / GC);
    }
    for (int i = 0; i < CB; ++i, ++col) {
        setPixel(col, 0, 255 - 255 * i / CB, 255);
    }
    for (int i = 0; i < BM; ++i, ++col) {
        setPixel(col, 255 * i / BM, 0, 255);
    }
    for (int i = 0; i < MR; ++i, ++col) {
        setPixel(col, 255 - 255 * i / MR, 0, 255);
    }

    return colorwheel;
}

vision::Image RaftPostprocessor::visualizeFlow(const vision::ImageView& flow_x, const vision::ImageView& flow_y) {
    const int rows = flow_x.height();
    const int cols = flow_x.width();

    if (flow_x.empty() || flow_y.empty() || rows != flow_y.height() || cols != flow_y.width()) {
        return vision::Image();
    }

    vision::Image magnitude = vision::Image::uninit(cols, rows, 1, vision::PixelType::Float32);
    vision::Image angle = vision::Image::uninit(cols, rows, 1, vision::PixelType::Float32);

    for (int i = 0; i < rows; ++i) {
        const float* fx = flow_x.ptr<float>(i);
        const float* fy = flow_y.ptr<float>(i);
        float* mag = magnitude.ptr<float>(i);
        float* ang = angle.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            mag[j] = std::sqrt(fx[j] * fx[j] + fy[j] * fy[j]);
            ang[j] = std::atan2(fy[j], fx[j]);
        }
    }

    double dummy_min = 0.0;
    double mag_max = 0.0;
    image_ops::minMax(magnitude.view(), dummy_min, mag_max);

    constexpr double kPi = 3.14159265358979323846;
    const float inv_two_pi = 1.0f / static_cast<float>(2.0 * kPi);

    if (mag_max > 0) {
        const float inv_mag_max = 1.0f / static_cast<float>(mag_max);
        for (int i = 0; i < rows; ++i) {
            float* mag = magnitude.ptr<float>(i);
            float* ang = angle.ptr<float>(i);
            for (int j = 0; j < cols; ++j) {
                mag[j] *= inv_mag_max;
                ang[j] = ang[j] * inv_two_pi + 0.5f;
            }
        }
    }

    vision::Image colorwheel = makeColorwheel();
    const int ncols = colorwheel.width();
    vision::Image flow_color(cols, rows, 3, vision::PixelType::UInt8);

    for (int i = 0; i < rows; ++i) {
        const float* mag = magnitude.ptr<float>(i);
        const float* ang = angle.ptr<float>(i);
        uint8_t* out = flow_color.ptr<uint8_t>(i);
        for (int j = 0; j < cols; ++j) {
            float m = mag[j];
            float a = ang[j];

            while (a < 0.0f)
                a += 1.0f;
            while (a >= 1.0f)
                a -= 1.0f;

            int k0 = static_cast<int>(a * static_cast<float>(ncols - 1));
            int k1 = (k0 + 1) % ncols;
            float f = (a * static_cast<float>(ncols - 1)) - static_cast<float>(k0);

            const uint8_t* col0 = colorwheel.ptr<uint8_t>(0) + k0 * 3;
            const uint8_t* col1 = colorwheel.ptr<uint8_t>(0) + k1 * 3;

            for (int ch = 0; ch < 3; ++ch) {
                float channel_value = (1.0f - f) * static_cast<float>(col0[ch]) + f * static_cast<float>(col1[ch]);
                if (m <= 1.0f) {
                    channel_value = 255.0f - m * (255.0f - channel_value);
                } else {
                    channel_value *= 0.75f;
                }
                out[j * 3 + ch] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, channel_value)));
            }
        }
    }

    return flow_color;
}

} // namespace neuriplo_tasks

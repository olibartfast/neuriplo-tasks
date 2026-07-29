#include "neuriplo/tasks/core/task_factory.hpp"
#include "neuriplo/tasks/instance_segmentation/yolo_segmentation_postprocessor.hpp"

#include <gtest/gtest.h>
#include <variant>
#include <vector>

using namespace neuriplo_tasks;

class YoloSegmentationPostprocessorTest : public ::testing::Test {
  protected:
    // Helper to create mock YOLO Seg output data
    std::pair<Tensor, Tensor> createMockYoloSegOutput() {
        // Output 0: Detections [1, 4+cls+32, 8400]
        // Output 1: Mask Protos [1, 32, 160, 160]

        int num_boxes = 100; // Reduced for test
        int num_classes = 10;
        int num_masks = 32;
        int channels = 4 + num_classes + num_masks;

        // Detection tensor: [1, channels, num_boxes]
        std::vector<TensorElement> det_data(channels * num_boxes, 0.0f);

        // Add one valid detection at index 0
        // Box (cx, cy, w, h)
        det_data[0 * num_boxes + 0] = 320.0f; // cx
        det_data[1 * num_boxes + 0] = 320.0f; // cy - Centered vertically to match mask center
        det_data[2 * num_boxes + 0] = 100.0f; // w
        det_data[3 * num_boxes + 0] = 100.0f; // h

        // Class score (class 0)
        det_data[(4 + 0) * num_boxes + 0] = 0.9f;

        // Mask coefficients (32 values) - set first one to 10.0 (high activation)
        det_data[(4 + num_classes) * num_boxes + 0] = 10.0f;

        Tensor det_tensor(det_data, {1, channels, num_boxes});

        // Prototype tensor: [1, 32, 160, 160]
        int proto_h = 160;
        int proto_w = 160;
        std::vector<TensorElement> proto_data(num_masks * proto_h * proto_w, 0.0f);

        // Activate center region of first mask (5x5 block)
        int center_y = proto_h / 2;
        int center_x = proto_w / 2;

        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                int idx = 0 * (proto_h * proto_w) + (center_y + dy) * proto_w + (center_x + dx);
                proto_data[idx] = 1.0f;
            }
        }

        Tensor proto_tensor(proto_data, {1, num_masks, proto_h, proto_w});

        return {det_tensor, proto_tensor};
    }
};

TEST_F(YoloSegmentationPostprocessorTest, TaskConfigSelectsMaskOrPolygonOutput) {
    auto [det_tensor, proto_tensor] = createMockYoloSegOutput();
    const std::vector<Tensor> tensors = {det_tensor, proto_tensor};
    ModelInfo info;
    info.input_shapes = {{1, 3, 640, 640}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"images"};
    info.output_names = {"output0", "output1"};

    auto mask_task = TaskFactory::createTaskInstance("yoloseg", info);
    const auto mask_results = mask_task->postprocess({640, 480}, tensors);
    ASSERT_EQ(mask_results.size(), 1U);
    const auto& mask_segmentation = std::get<InstanceSegmentation>(mask_results[0]);
    EXPECT_FALSE(mask_segmentation.mask.empty());
    EXPECT_TRUE(mask_segmentation.polygons.empty());

    TaskConfig config;
    config.segmentation_output = SegmentationOutput::Polygon;
    auto polygon_task = TaskFactory::createTaskInstance("yoloseg", info, config);
    const auto polygon_results = polygon_task->postprocess({640, 480}, tensors);
    ASSERT_EQ(polygon_results.size(), 1U);
    const auto& polygon_segmentation = std::get<InstanceSegmentation>(polygon_results[0]);
    EXPECT_TRUE(polygon_segmentation.mask.empty());
    EXPECT_TRUE(polygon_segmentation.mask_data.empty());
    EXPECT_EQ(polygon_segmentation.mask_height, 0);
    EXPECT_EQ(polygon_segmentation.mask_width, 0);
    EXPECT_FALSE(polygon_segmentation.polygons.empty());
}

TEST_F(YoloSegmentationPostprocessorTest, StandardOrder) {
    YoloSegmentationPostprocessor processor(InstanceSegmentationTask::ModelType::YOLO_SEG,
                                            neuriplo_tasks::Size(640, 640), 0.25f, 0.45f, 0.45f);

    auto [det_tensor, proto_tensor] = createMockYoloSegOutput();
    std::vector<Tensor> tensors = {det_tensor, proto_tensor};
    neuriplo_tasks::Size frame_size(640, 480);

    auto results = processor.postprocess(tensors, frame_size);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].class_id, 0);
    EXPECT_NEAR(results[0].class_confidence, 0.9f, 0.001f);
}

TEST_F(YoloSegmentationPostprocessorTest, SwappedOrder) {
    YoloSegmentationPostprocessor processor(InstanceSegmentationTask::ModelType::YOLO_SEG,
                                            neuriplo_tasks::Size(640, 640), 0.25f, 0.45f, 0.45f);

    auto [det_tensor, proto_tensor] = createMockYoloSegOutput();
    // Swap order: Proto first, then Detection
    std::vector<Tensor> tensors = {proto_tensor, det_tensor};
    neuriplo_tasks::Size frame_size(640, 480);

    auto results = processor.postprocess(tensors, frame_size);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].class_id, 0);
    EXPECT_NEAR(results[0].class_confidence, 0.9f, 0.001f);
}

TEST_F(YoloSegmentationPostprocessorTest, NmsFreeSwappedOrder) {
    YoloSegmentationPostprocessor processor(InstanceSegmentationTask::ModelType::YOLO_V10_SEG,
                                            neuriplo_tasks::Size(640, 640), 0.25f, 0.45f, 0.5f);

    // Mock NMS-free output
    // Output 0: Detections [1, 300, 38]
    int num_dets = 300;
    int dims = 38;
    std::vector<TensorElement> det_data(num_dets * dims, 0.0f);

    // Add valid detection
    det_data[0] = 100.0f; // x1
    det_data[1] = 100.0f; // y1
    det_data[2] = 200.0f; // x2
    det_data[3] = 200.0f; // y2
    det_data[4] = 0.9f;   // score
    det_data[5] = 0.0f;   // class 0
    det_data[6] = 10.0f;  // first mask coeff

    Tensor det_tensor(det_data, {1, num_dets, dims});

    // Prototype tensor: [1, 32, 160, 160]
    int proto_h = 160;
    int proto_w = 160;
    std::vector<TensorElement> proto_data(32 * proto_h * proto_w, 0.0f);
    int center_idx = (proto_h / 2) * proto_w + (proto_w / 2);
    proto_data[center_idx] = 1.0f;

    Tensor proto_tensor(proto_data, {1, 32, proto_h, proto_w});

    // Swap order
    std::vector<Tensor> tensors = {proto_tensor, det_tensor};
    neuriplo_tasks::Size frame_size(640, 480);

    auto results = processor.postprocess(tensors, frame_size);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].class_id, 0);
}

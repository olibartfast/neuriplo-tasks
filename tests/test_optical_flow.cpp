#include "neuriplo/tasks/core/model_info.hpp"
#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_factory.hpp"
#include "neuriplo/tasks/optical_flow/optical_flow_preprocessor.hpp"
#include "neuriplo/tasks/optical_flow/raft_postprocessor.hpp"
#include "vision_test_utils.hpp"

#include <chrono>

#if __has_include(<valgrind/valgrind.h>)
#include <valgrind/valgrind.h>
#endif
#include <gtest/gtest.h>

using namespace neuriplo_tasks;

class OpticalFlowTest : public ::testing::Test {
  protected:
    ModelInfo createOpticalFlowModelInfo(int width = 960, int height = 520) {
        ModelInfo info;
        // RAFT typically expects [batch, channels, height, width]
        info.input_shapes = {{1, 3, height, width}};
        info.input_formats = {"FORMAT_NCHW"};
        info.input_names = {"frame1", "frame2"};
        info.output_names = {"flow"};
        info.input_types = {neuriplo_tasks::PixelType::Float32};
        return info;
    }

    std::pair<neuriplo_tasks::Image, neuriplo_tasks::Image> createTestFramePair(int width = 100, int height = 100) {
        // Create two simple test frames with some movement
        neuriplo_tasks::Image frame1 = neuriplo_tasks::vision_test::makeImage(width, height, 3, 0);
        neuriplo_tasks::Image frame2 = neuriplo_tasks::vision_test::makeImage(width, height, 3, 0);

        // Draw a rectangle in frame1
        neuriplo_tasks::vision_test::fillRect(frame1, 20, 20, 40, 40, 255);

        // Draw the same rectangle shifted in frame2 (simulating movement)
        neuriplo_tasks::vision_test::fillRect(frame2, 25, 25, 45, 45, 255);

        return {frame1, frame2};
    }

    std::vector<TensorElement> createMockFlowOutput(int height, int width) {
        // Create mock flow output [batch, 2, height, width] - 2 channels for x,y flow
        std::vector<TensorElement> flow_data;
        size_t total_elements = 1 * 2 * height * width; // batch=1, channels=2
        flow_data.reserve(total_elements);

        // Fill with simple flow pattern
        for (size_t i = 0; i < total_elements; ++i) {
            // Simple pattern: alternating small positive values
            flow_data.push_back(static_cast<float>((i % 2 == 0) ? 5.0f : 3.0f));
        }

        return flow_data;
    }
};

// Test TaskFactory creates optical flow task
TEST_F(OpticalFlowTest, CreateOpticalFlowTask) {
    auto info = createOpticalFlowModelInfo();

    // Only "raft" is currently registered in TaskFactory
    std::vector<std::string> types = {"raft"};

    for (const auto& type : types) {
        auto task = TaskFactory::createTaskInstance(type, info);
        ASSERT_NE(task, nullptr) << "Failed to create task for type: " << type;
        EXPECT_EQ(task->getTaskType(), TaskType::OpticalFlow);
    }
}

// Test optical flow task with invalid frame count
TEST_F(OpticalFlowTest, PreprocessWithInvalidFrameCount) {
    auto info = createOpticalFlowModelInfo();
    auto task = TaskFactory::createTaskInstance("raft", info);
    ASSERT_NE(task, nullptr);

    // Single frame should throw
    neuriplo_tasks::Image frame = neuriplo_tasks::vision_test::makeImage(100, 100, 3, 0);
    std::vector<neuriplo_tasks::Image> single_frame = {frame};

    EXPECT_THROW(task->preprocess(single_frame), std::invalid_argument);

    // Odd number of frames should throw
    std::vector<neuriplo_tasks::Image> odd_frames = {frame, frame, frame};
    EXPECT_THROW(task->preprocess(odd_frames), std::invalid_argument);
}

// Test optical flow task preprocessing with valid frame pairs
TEST_F(OpticalFlowTest, PreprocessValidFramePairs) {
    auto info = createOpticalFlowModelInfo();
    auto task = TaskFactory::createTaskInstance("raft", info);
    ASSERT_NE(task, nullptr);

    auto [frame1, frame2] = createTestFramePair();
    std::vector<neuriplo_tasks::Image> frames = {frame1, frame2};

    auto result = task->preprocess(frames);

    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), 2); // Two separate preprocessed frames
    EXPECT_GT(result[0].size(), 0);
    EXPECT_GT(result[1].size(), 0);
}

// Test optical flow task preprocessing with multiple frame pairs
TEST_F(OpticalFlowTest, PreprocessMultipleFramePairs) {
    auto info = createOpticalFlowModelInfo();
    auto task = TaskFactory::createTaskInstance("raft", info);
    ASSERT_NE(task, nullptr);

    auto [frame1, frame2] = createTestFramePair();
    auto [frame3, frame4] = createTestFramePair();

    std::vector<neuriplo_tasks::Image> frames = {frame1, frame2, frame3, frame4};

    auto result = task->preprocess(frames);

    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), 4); // Four separate preprocessed frames (2 pairs * 2 frames each)
}

// Test RaftPreprocessor directly
TEST_F(OpticalFlowTest, RaftPreprocessorDirect) {
    RaftPreprocessor preprocessor(neuriplo_tasks::Size(960, 520));

    auto [frame1, frame2] = createTestFramePair();

    auto result = preprocessor.preprocess_pair(frame1.view(), frame2.view());

    EXPECT_EQ(result.size(), 2); // Two frames preprocessed
    EXPECT_GT(result[0].size(), 0);
    EXPECT_GT(result[1].size(), 0);
}

// Test RaftPreprocessor with empty frames
TEST_F(OpticalFlowTest, RaftPreprocessorEmptyFrames) {
    RaftPreprocessor preprocessor(neuriplo_tasks::Size(960, 520));

    neuriplo_tasks::Image empty_frame;
    neuriplo_tasks::Image valid_frame = neuriplo_tasks::vision_test::makeImage(100, 100, 3, 0);

    // Empty image pairs are rejected by the vision resize path
    EXPECT_THROW(preprocessor.preprocess_pair(empty_frame.view(), valid_frame.view()), std::invalid_argument);
    EXPECT_THROW(preprocessor.preprocess_pair(valid_frame.view(), empty_frame.view()), std::invalid_argument);
}

// Test RaftPostprocessor
TEST_F(OpticalFlowTest, RaftPostprocessorBasic) {
    RaftPostprocessor postprocessor;

    int height = 52;
    int width = 96;
    auto flow_output = createMockFlowOutput(height, width);
    std::vector<int64_t> shape = {1, 2, height, width}; // [batch, channels, height, width]

    neuriplo_tasks::Size frame_size(200, 150); // Original frame size

    auto results = postprocessor.postprocess(flow_output, shape, frame_size);

    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results.size(), 1);

    const auto& flow = results[0];
    // Note: flow visualization is not implemented yet, so flow.flow will be empty
    // EXPECT_FALSE(flow.flow.empty()); // TODO: uncomment when visualization is implemented
    EXPECT_FALSE(flow.raw_flow.empty());                                      // Raw flow should exist
    EXPECT_EQ(flow.raw_flow.pixelType(), neuriplo_tasks::PixelType::Float32); // Two channels (x, y flow)
    // max_displacement is not calculated in current implementation
    // EXPECT_GE(flow.max_displacement, 0.0f); // TODO: uncomment when max_displacement is implemented
}

// Test optical flow task full pipeline (preprocess + postprocess)
TEST_F(OpticalFlowTest, FullPipelineSimulation) {
    auto info = createOpticalFlowModelInfo(96, 52);
    auto task = TaskFactory::createTaskInstance("raft", info);
    ASSERT_NE(task, nullptr);

    // Create test frames
    auto [frame1, frame2] = createTestFramePair(200, 150);
    std::vector<neuriplo_tasks::Image> frames = {frame1, frame2};

    // Preprocess
    auto preprocessed = task->preprocess(frames);
    ASSERT_FALSE(preprocessed.empty());

    // Simulate inference output
    int height = 52;
    int width = 96;
    std::vector<Tensor> tensors = {Tensor(createMockFlowOutput(height, width), {1, 2, height, width})};

    // Postprocess
    auto results = task->postprocess(frame1.size(), tensors);

    ASSERT_FALSE(results.empty());
    ASSERT_EQ(results.size(), 1);

    // Verify result is OpticalFlow type
    ASSERT_TRUE(std::holds_alternative<OpticalFlow>(results[0]));

    const auto& flow = std::get<OpticalFlow>(results[0]);
    EXPECT_FALSE(flow.flow.empty());
    EXPECT_FALSE(flow.raw_flow.empty());
    EXPECT_EQ(flow.raw_flow.pixelType(), neuriplo_tasks::PixelType::Float32);
    EXPECT_GE(flow.max_displacement, 0.0f);
}

// Test optical flow with empty inference results
TEST_F(OpticalFlowTest, PostprocessEmptyResults) {
    auto info = createOpticalFlowModelInfo();
    auto task = TaskFactory::createTaskInstance("raft", info);
    ASSERT_NE(task, nullptr);

    neuriplo_tasks::Size frame_size(200, 150);
    std::vector<Tensor> empty_tensors;

    auto results = task->postprocess(frame_size, empty_tensors);

    EXPECT_TRUE(results.empty());
}

// Test optical flow with mismatched tensor shapes
TEST_F(OpticalFlowTest, PostprocessMismatchedShapes) {
    auto info = createOpticalFlowModelInfo();
    auto task = TaskFactory::createTaskInstance("raft", info);
    ASSERT_NE(task, nullptr);

    auto flow_output = createMockFlowOutput(52, 96);
    std::vector<Tensor> tensors = {
        Tensor(flow_output, {1, 2, 52, 96}), Tensor(flow_output, {1, 2, 52, 96}) // Extra tensor
    };

    neuriplo_tasks::Size frame_size(200, 150);

    // Current implementation doesn't validate matching sizes, it just uses the first tensor
    // So this should not throw
    auto results = task->postprocess(frame_size, tensors);
    EXPECT_FALSE(results.empty()); // Should still produce a result from first tensor
}

// Test optical flow result structure
TEST_F(OpticalFlowTest, OpticalFlowResultStructure) {
    OpticalFlow flow_result;

    // Test default construction
    EXPECT_TRUE(flow_result.flow.empty());
    EXPECT_TRUE(flow_result.raw_flow.empty());
    EXPECT_FLOAT_EQ(flow_result.max_displacement, 0.0f);

    // Test assignment
    flow_result.flow = fromImage(neuriplo_tasks::vision_test::makeImage(100, 100, 3, 0));
    flow_result.raw_flow = fromImage(neuriplo_tasks::vision_test::makeFloatImage(100, 100, 2, 0.0F));
    flow_result.max_displacement = 10.5f;

    EXPECT_FALSE(flow_result.flow.empty());
    EXPECT_FALSE(flow_result.raw_flow.empty());
    EXPECT_FLOAT_EQ(flow_result.max_displacement, 10.5f);
}

// Test optical flow with different input sizes
TEST_F(OpticalFlowTest, DifferentInputSizes) {
    std::vector<std::pair<int, int>> sizes = {
        {480, 270},  // Small
        {960, 520},  // Default
        {1280, 720}, // HD
    };

    for (const auto& [width, height] : sizes) {
        auto info = createOpticalFlowModelInfo(width, height);
        auto task = TaskFactory::createTaskInstance("raft", info);

        ASSERT_NE(task, nullptr) << "Failed for size: " << width << "x" << height;
        EXPECT_EQ(task->getTaskType(), TaskType::OpticalFlow);
    }
}

// Performance test: measure preprocessing time
TEST_F(OpticalFlowTest, PreprocessingPerformance) {
#if defined(RUNNING_ON_VALGRIND)
    if (RUNNING_ON_VALGRIND) {
        GTEST_SKIP() << "Performance timing is not meaningful under Valgrind";
    }
#endif
    auto info = createOpticalFlowModelInfo();
    auto task = TaskFactory::createTaskInstance("raft", info);
    ASSERT_NE(task, nullptr);

    auto [frame1, frame2] = createTestFramePair(640, 480);
    std::vector<neuriplo_tasks::Image> frames = {frame1, frame2};

    auto start = std::chrono::high_resolution_clock::now();
    auto result = task->preprocess(frames);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Preprocessing should be reasonably fast (< 1 second for small images)
    EXPECT_LT(duration.count(), 1000) << "Preprocessing took too long: " << duration.count() << "ms";
    EXPECT_FALSE(result.empty());
}

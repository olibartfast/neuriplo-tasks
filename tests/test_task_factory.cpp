#include <gtest/gtest.h>
#include "vision-core/core/task_factory.hpp"
#include "vision-core/core/model_info.hpp"

using namespace vision_core;

// Minimal test task for registration tests
class TestTask : public TaskInterface {
public:
    TestTask(const ModelInfo& model_info) : TaskInterface(model_info) {}
    
    TaskType getTaskType() override { return TaskType::Detection; }
    
    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override {
        return {};
    }
    
    std::vector<Result> postprocess(
        const cv::Size&,
        const std::vector<std::vector<TensorElement>>&,
        const std::vector<std::vector<int64_t>>&) override {
        return {};
    }
};

class TaskFactoryTest : public ::testing::Test {
protected:
    ModelInfo createValidModelInfo() {
        ModelInfo info;
        info.input_shapes = {{1, 3, 640, 640}};
        info.input_formats = {"FORMAT_NCHW"};
        info.input_names = {"images"};
        info.output_names = {"output0"};
        info.input_types = {CV_32F};
        info.max_batch_size_ = 1;
        info.batch_size_ = 1;
        return info;
    }
};

TEST_F(TaskFactoryTest, ValidateInputSizesEmpty) {
    ModelInfo info;
    info.input_shapes = {};
    
    EXPECT_THROW({
        TaskFactory::createTaskInstance("yolov8", info);
    }, InputDimensionError);
}

TEST_F(TaskFactoryTest, ValidateInputSizesNegative) {
    ModelInfo info;
    info.input_shapes = {{1, 3, -640, 640}};
    info.input_formats = {"FORMAT_NCHW"};
    
    EXPECT_THROW({
        TaskFactory::createTaskInstance("yolov8", info);
    }, InputDimensionError);
}

TEST_F(TaskFactoryTest, UnrecognizedModelType) {
    auto info = createValidModelInfo();
    
    EXPECT_THROW({
        TaskFactory::createTaskInstance("unknown_model", info);
    }, std::invalid_argument);
}

TEST_F(TaskFactoryTest, EmptyModelType) {
    auto info = createValidModelInfo();
    
    EXPECT_THROW({
        TaskFactory::createTaskInstance("", info);
    }, std::invalid_argument);
}

TEST_F(TaskFactoryTest, InvalidInputDimensions) {
    ModelInfo info;
    info.input_shapes = {{1, 0}};
    info.input_formats = {"FORMAT_NCHW"};
    
    // Register a test task first
    TaskFactory::registerTask("yolov8", 
        [](const ModelInfo& model_info) { return std::make_unique<TestTask>(model_info); });
    
    EXPECT_THROW({
        TaskFactory::createTaskInstance("yolov8", info);
    }, InputDimensionError);
}

TEST_F(TaskFactoryTest, CustomTaskRegistration) {
    auto info = createValidModelInfo();

    // Register custom task
    TaskFactory::registerTask("custom-test-model", 
        [](const ModelInfo& model_info) -> std::unique_ptr<TaskInterface> {
            return std::make_unique<TestTask>(model_info);
        });

    // Should be able to create it
    auto task = TaskFactory::createTaskInstance("custom-test-model", info);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Detection);
}

TEST_F(TaskFactoryTest, RegisterEmptyModelTypeThrows) {
    EXPECT_THROW({
        TaskFactory::registerTask("", 
            [](const ModelInfo& info) { return std::make_unique<TestTask>(info); });
    }, std::invalid_argument);
}

TEST_F(TaskFactoryTest, RegisterNullCreatorThrows) {
    EXPECT_THROW({
        TaskFactory::registerTask("test", nullptr);
    }, std::invalid_argument);
}

// Note: Actual task creation tests would require implementing the concrete task classes
// These tests verify the factory's error handling and validation logic

TEST_F(TaskFactoryTest, ModelInfoCopy) {
    auto info1 = createValidModelInfo();
    ModelInfo info2 = info1;
    
    EXPECT_EQ(info1.input_shapes.size(), info2.input_shapes.size());
    EXPECT_EQ(info1.input_formats.size(), info2.input_formats.size());
    EXPECT_EQ(info1.input_names.size(), info2.input_names.size());
    EXPECT_EQ(info1.max_batch_size_, info2.max_batch_size_);
}

TEST_F(TaskFactoryTest, ModelInfoDefaultConstruction) {
    ModelInfo info;
    EXPECT_TRUE(info.input_shapes.empty());
    EXPECT_TRUE(info.input_formats.empty());
    EXPECT_TRUE(info.input_names.empty());
    EXPECT_TRUE(info.output_names.empty());
    EXPECT_EQ(info.max_batch_size_, 1);
    EXPECT_EQ(info.batch_size_, 1);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

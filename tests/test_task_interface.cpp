#include "neuriplo/tasks/core/model_info.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"
#include "vision_test_utils.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

using namespace neuriplo_tasks;

// Concrete implementation for testing
class TestTask : public TaskInterface {
  public:
    TestTask(const ModelInfo& model_info) : TaskInterface(model_info) {}

    TaskType getTaskType() override { return TaskType::Detection; }

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<neuriplo_tasks::Image>& imgs) override {
        // Simple test implementation
        std::vector<std::vector<uint8_t>> result;
        for (const auto& img : imgs) {
            result.push_back(std::vector<uint8_t>(img.totalPixels() * img.channels()));
        }
        return result;
    }

    std::vector<Result> postprocess(const neuriplo_tasks::Size& frame_size,
                                    const std::vector<Tensor>& tensors) override {

        // Simple test implementation - return one detection
        Detection det(BoundingBox(10, 10, 50, 50), 0.9f, 0);
        return {det};
    }

    // Expose protected members for testing
    int get_input_width() const { return input_width_; }
    int get_input_height() const { return input_height_; }
    int get_input_channels() const { return input_channels_; }
};

class TaskInterfaceTest : public ::testing::Test {
  protected:
    ModelInfo createValidModelInfo() {
        ModelInfo info;
        info.input_shapes = {{1, 3, 640, 640}};
        info.input_formats = {"FORMAT_NCHW"};
        info.input_names = {"images"};
        info.output_names = {"output0"};
        info.input_types = {neuriplo_tasks::PixelType::Float32};
        return info;
    }

    ModelInfo createInvalidModelInfo() {
        ModelInfo info;
        info.input_shapes = {{1, 0}}; // Invalid dimensions
        info.input_formats = {"FORMAT_NCHW"};
        return info;
    }
};

TEST_F(TaskInterfaceTest, ConstructWithValidModelInfo) {
    auto model_info = createValidModelInfo();

    EXPECT_NO_THROW({
        TestTask task(model_info);
        EXPECT_EQ(task.get_input_width(), 640);
        EXPECT_EQ(task.get_input_height(), 640);
        EXPECT_EQ(task.get_input_channels(), 3);
    });
}

TEST_F(TaskInterfaceTest, ConstructWithInvalidModelInfoThrows) {
    auto model_info = createInvalidModelInfo();

    EXPECT_THROW({ TestTask task(model_info); }, InputDimensionError);
}

TEST_F(TaskInterfaceTest, NCHWFormatParsing) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 480, 640}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"input"};
    info.output_names = {"output"};

    TestTask task(info);
    EXPECT_EQ(task.get_input_width(), 640);
    EXPECT_EQ(task.get_input_height(), 480);
    EXPECT_EQ(task.get_input_channels(), 3);
}

TEST_F(TaskInterfaceTest, NHWCFormatParsing) {
    ModelInfo info;
    info.input_shapes = {{1, 480, 640, 3}};
    info.input_formats = {"FORMAT_NHWC"};
    info.input_names = {"input"};
    info.output_names = {"output"};

    TestTask task(info);
    EXPECT_EQ(task.get_input_width(), 640);
    EXPECT_EQ(task.get_input_height(), 480);
    EXPECT_EQ(task.get_input_channels(), 3);
}

TEST_F(TaskInterfaceTest, GetTaskType) {
    auto model_info = createValidModelInfo();
    TestTask task(model_info);

    EXPECT_EQ(task.getTaskType(), TaskType::Detection);
}

TEST_F(TaskInterfaceTest, PreprocessReturnsData) {
    auto model_info = createValidModelInfo();
    TestTask task(model_info);

    neuriplo_tasks::Image image = neuriplo_tasks::vision_test::makeImage(640, 480, 3, 0);
    std::vector<neuriplo_tasks::Image> images = {image};

    auto result = task.preprocess(images);

    EXPECT_EQ(result.size(), 1);
    EXPECT_GT(result[0].size(), 0);
}

TEST_F(TaskInterfaceTest, PostprocessReturnsResults) {
    auto model_info = createValidModelInfo();
    TestTask task(model_info);

    neuriplo_tasks::Size frame_size(640, 480);
    std::vector<Tensor> tensors;

    auto results = task.postprocess(frame_size, tensors);

    EXPECT_EQ(results.size(), 1);
    EXPECT_TRUE(std::holds_alternative<Detection>(results[0]));
}

TEST_F(TaskInterfaceTest, ReadLabelNamesNonExistentFile) {
    auto model_info = createValidModelInfo();
    TestTask task(model_info);

    auto labels = task.readLabelNames("nonexistent_file.txt");
    EXPECT_TRUE(labels.empty());
}

TEST_F(TaskInterfaceTest, ReadLabelNamesValidFile) {
    // Create a temporary labels file
    const auto temp_file =
        std::filesystem::temp_directory_path() / std::filesystem::path("vision-core-test-labels.txt");
    std::ofstream ofs(temp_file);
    ofs << "person\n";
    ofs << "car\n";
    ofs << "dog\n";
    ofs.close();

    auto model_info = createValidModelInfo();
    TestTask task(model_info);

    auto labels = task.readLabelNames(temp_file.string());
    EXPECT_EQ(labels.size(), 3);
    EXPECT_EQ(labels[0], "person");
    EXPECT_EQ(labels[1], "car");
    EXPECT_EQ(labels[2], "dog");

    // Cleanup
    std::filesystem::remove(temp_file);
}

TEST_F(TaskInterfaceTest, InputDimensionErrorMessage) {
    try {
        auto model_info = createInvalidModelInfo();
        TestTask task(model_info);
        FAIL() << "Expected InputDimensionError";
    } catch (const InputDimensionError& e) {
        std::string msg = e.what();
        EXPECT_FALSE(msg.empty());
    }
}

TEST_F(TaskInterfaceTest, MultipleImagesPreprocess) {
    auto model_info = createValidModelInfo();
    TestTask task(model_info);

    std::vector<neuriplo_tasks::Image> images;
    images.push_back(neuriplo_tasks::vision_test::makeImage(640, 480, 3, 0));
    images.push_back(neuriplo_tasks::vision_test::makeImage(640, 480, 3, 0));
    images.push_back(neuriplo_tasks::vision_test::makeImage(640, 480, 3, 0));

    auto result = task.preprocess(images);

    EXPECT_EQ(result.size(), 3);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

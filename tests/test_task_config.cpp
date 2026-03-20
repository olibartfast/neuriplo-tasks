#include "vision-core/core/model_info.hpp"
#include "vision-core/core/result_types.hpp"
#include "vision-core/core/task_config.hpp"
#include "vision-core/core/task_factory.hpp"
#include "vision-core/core/task_interface.hpp"

#include <gtest/gtest.h>
#include <variant>
#include <vector>

using namespace vision_core;

// ============================================================
// Helpers
// ============================================================

static ModelInfo makeDetectionInfo() {
    ModelInfo info;
    info.input_shapes = {{1, 3, 640, 640}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"images"};
    info.output_names = {"output0"};
    info.input_types = {CV_32F};
    return info;
}

static ModelInfo makeClassificationInfo() {
    ModelInfo info;
    info.input_shapes = {{1, 3, 224, 224}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"input"};
    info.output_names = {"output"};
    info.input_types = {CV_32F};
    return info;
}

// Build a minimal classification tensor: [1, num_classes] with one hot value.
static Tensor makeClassificationTensor(int num_classes, int hot_class = 0, float hot_val = 10.0f) {
    std::vector<TensorElement> data(static_cast<size_t>(num_classes), 0.0f);
    data[static_cast<size_t>(hot_class)] = hot_val;
    return Tensor{data, {1, static_cast<int64_t>(num_classes)}};
}

// Build a YOLOv8-style detection tensor: [1, 4+num_classes, num_boxes].
// Places one detection with given score at box index 0.
static Tensor makeYoloTensor(int num_classes, int num_boxes, float detection_score) {
    int channels = 4 + num_classes;
    std::vector<TensorElement> data(static_cast<size_t>(channels * num_boxes), 0.0f);
    // Box at index 0: centre x=320, y=240, w=100, h=150
    // Class 0 score = detection_score
    data[0] = 320.0f;                                           // x
    data[static_cast<size_t>(num_boxes)] = 240.0f;              // y
    data[static_cast<size_t>(2 * num_boxes)] = 100.0f;          // w
    data[static_cast<size_t>(3 * num_boxes)] = 150.0f;          // h
    data[static_cast<size_t>(4 * num_boxes)] = detection_score; // class 0 score
    return Tensor{data, {1, static_cast<int64_t>(channels), static_cast<int64_t>(num_boxes)}};
}

// ============================================================
// TaskConfig struct tests
// ============================================================

TEST(TaskConfigTest, DefaultValues) {
    TaskConfig cfg;
    EXPECT_FLOAT_EQ(cfg.confidence_threshold, 0.25f);
    EXPECT_FLOAT_EQ(cfg.nms_threshold, 0.45f);
    EXPECT_FLOAT_EQ(cfg.mask_threshold, 0.50f);
    EXPECT_FLOAT_EQ(cfg.text_threshold, 0.10f);
    EXPECT_EQ(cfg.top_k, 5);
    EXPECT_EQ(cfg.max_text_queries, 16);
    EXPECT_TRUE(cfg.apply_softmax);
    EXPECT_TRUE(cfg.cache_text_features);
    EXPECT_TRUE(cfg.tokenizer_vocab_path.empty());
    EXPECT_TRUE(cfg.tokenizer_merges_path.empty());
    EXPECT_TRUE(cfg.tokenizer_vocab_json.empty());
    EXPECT_TRUE(cfg.tokenizer_merges_text.empty());
    EXPECT_TRUE(cfg.text_prompts.empty());
    EXPECT_TRUE(cfg.extra_params.empty());
}

TEST(TaskConfigTest, FieldAssignment) {
    TaskConfig cfg;
    cfg.confidence_threshold = 0.5f;
    cfg.nms_threshold = 0.3f;
    cfg.mask_threshold = 0.7f;
    cfg.text_threshold = 0.2f;
    cfg.top_k = 3;
    cfg.max_text_queries = 8;
    cfg.apply_softmax = false;
    cfg.cache_text_features = false;
    cfg.tokenizer_vocab_path = "vocab.json";
    cfg.tokenizer_merges_path = "merges.txt";
    cfg.tokenizer_vocab_json = "{\"cat\":1}";
    cfg.tokenizer_merges_text = "c a";
    cfg.text_prompts = {"cat", "dog"};

    EXPECT_FLOAT_EQ(cfg.confidence_threshold, 0.5f);
    EXPECT_FLOAT_EQ(cfg.nms_threshold, 0.3f);
    EXPECT_FLOAT_EQ(cfg.mask_threshold, 0.7f);
    EXPECT_FLOAT_EQ(cfg.text_threshold, 0.2f);
    EXPECT_EQ(cfg.top_k, 3);
    EXPECT_EQ(cfg.max_text_queries, 8);
    EXPECT_FALSE(cfg.apply_softmax);
    EXPECT_FALSE(cfg.cache_text_features);
    EXPECT_EQ(cfg.tokenizer_vocab_path, "vocab.json");
    EXPECT_EQ(cfg.tokenizer_merges_path, "merges.txt");
    EXPECT_EQ(cfg.tokenizer_vocab_json, "{\"cat\":1}");
    EXPECT_EQ(cfg.tokenizer_merges_text, "c a");
    ASSERT_EQ(cfg.text_prompts.size(), 2u);
    EXPECT_EQ(cfg.text_prompts[0], "cat");
}

TEST(TaskConfigTest, ExtraParams) {
    TaskConfig cfg;
    cfg.extra_params["text_prompt"] = "detect all cars";
    cfg.extra_params["temperature"] = "0.7";

    ASSERT_EQ(cfg.extra_params.size(), 2u);
    EXPECT_EQ(cfg.extra_params.at("text_prompt"), "detect all cars");
    EXPECT_EQ(cfg.extra_params.at("temperature"), "0.7");
}

TEST(TaskConfigTest, CopyAndEquality) {
    TaskConfig a;
    a.confidence_threshold = 0.6f;
    a.top_k = 2;
    a.extra_params["key"] = "val";

    TaskConfig b = a;
    EXPECT_FLOAT_EQ(b.confidence_threshold, 0.6f);
    EXPECT_EQ(b.top_k, 2);
    EXPECT_EQ(b.extra_params.at("key"), "val");
}

// ============================================================
// Factory backward-compatibility (two-arg call)
// ============================================================

TEST(TaskConfigTest, FactoryTwoArgCallStillWorks) {
    auto info = makeDetectionInfo();
    auto task = TaskFactory::createTaskInstance("yolov8", info);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Detection);
}

// ============================================================
// Factory accepts explicit config for every task category
// ============================================================

TEST(TaskConfigTest, FactoryDetectionWithCustomConfig) {
    TaskConfig cfg;
    cfg.confidence_threshold = 0.5f;
    cfg.nms_threshold = 0.3f;

    auto task = TaskFactory::createTaskInstance("yolov8", makeDetectionInfo(), cfg);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Detection);
}

TEST(TaskConfigTest, FactoryRtDetrWithCustomConfig) {
    TaskConfig cfg;
    cfg.confidence_threshold = 0.4f;

    auto task = TaskFactory::createTaskInstance("rtdetr", makeDetectionInfo(), cfg);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Detection);
}

TEST(TaskConfigTest, FactorySegmentationWithCustomConfig) {
    TaskConfig cfg;
    cfg.confidence_threshold = 0.4f;
    cfg.nms_threshold = 0.3f;
    cfg.mask_threshold = 0.6f;

    auto task = TaskFactory::createTaskInstance("yoloseg", makeDetectionInfo(), cfg);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::InstanceSegmentation);
}

TEST(TaskConfigTest, FactoryClassificationWithCustomConfig) {
    TaskConfig cfg;
    cfg.top_k = 1;
    cfg.apply_softmax = false;

    auto task = TaskFactory::createTaskInstance("resnet50", makeClassificationInfo(), cfg);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::Classification);
}

TEST(TaskConfigTest, FactoryDepthEstimationIgnoresUnusedFields) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 518, 518}};
    info.input_formats = {"FORMAT_NCHW"};
    info.input_names = {"input"};
    info.output_names = {"output"};
    info.input_types = {CV_32F};

    TaskConfig cfg;
    cfg.confidence_threshold = 0.99f; // irrelevant to depth estimation — must not crash

    auto task = TaskFactory::createTaskInstance("depth-anything-v2", info, cfg);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::DepthEstimation);
}

TEST(TaskConfigTest, FactoryOpticalFlowIgnoresUnusedFields) {
    ModelInfo info;
    info.input_shapes = {{1, 3, 384, 512}, {1, 3, 384, 512}};
    info.input_formats = {"FORMAT_NCHW", "FORMAT_NCHW"};
    info.input_names = {"image1", "image2"};
    info.output_names = {"flow"};
    info.input_types = {CV_32F, CV_32F};

    TaskConfig cfg;
    cfg.confidence_threshold = 0.5f; // irrelevant to optical flow — must not crash

    auto task = TaskFactory::createTaskInstance("raft", info, cfg);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::OpticalFlow);
}

// ============================================================
// Behavioral tests — config values affect postprocess output
// ============================================================

TEST(TaskConfigTest, ClassificationTopKAffectsResultCount) {
    const int num_classes = 10;
    auto tensor = makeClassificationTensor(num_classes, 3, 10.0f);
    cv::Size frame(224, 224);

    {
        TaskConfig cfg;
        cfg.top_k = 1;
        auto task = TaskFactory::createTaskInstance("resnet50", makeClassificationInfo(), cfg);
        auto results = task->postprocess(frame, {tensor});
        ASSERT_EQ(results.size(), 1u);
        const auto& cls = std::get<Classification>(results[0]);
        EXPECT_EQ(cls.class_id, 3); // hot class wins
    }

    {
        TaskConfig cfg;
        cfg.top_k = 5;
        auto task = TaskFactory::createTaskInstance("resnet50", makeClassificationInfo(), cfg);
        auto results = task->postprocess(frame, {tensor});
        EXPECT_EQ(results.size(), 5u);
    }
}

TEST(TaskConfigTest, DetectionConfidenceThresholdFiltersResults) {
    const int num_classes = 80;
    // num_boxes must be > (4 + num_classes) so shape[2] > shape[1] and the
    // postprocessor correctly identifies YOLOv8+ format (no objectness score).
    const int num_boxes = 8400;
    cv::Size frame(640, 640);

    // Score below threshold → no detections
    {
        TaskConfig cfg;
        cfg.confidence_threshold = 0.90f;
        auto tensor = makeYoloTensor(num_classes, num_boxes, 0.50f);
        auto task = TaskFactory::createTaskInstance("yolov8", makeDetectionInfo(), cfg);
        auto results = task->postprocess(frame, {tensor});
        EXPECT_TRUE(results.empty());
    }

    // Score above threshold → at least one detection
    {
        TaskConfig cfg;
        cfg.confidence_threshold = 0.25f;
        auto tensor = makeYoloTensor(num_classes, num_boxes, 0.95f);
        auto task = TaskFactory::createTaskInstance("yolov8", makeDetectionInfo(), cfg);
        auto results = task->postprocess(frame, {tensor});
        EXPECT_FALSE(results.empty());
    }
}

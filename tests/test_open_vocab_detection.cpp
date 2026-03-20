#include "vision-core/core/task_factory.hpp"

#include <gtest/gtest.h>

using namespace vision_core;

namespace {

ModelInfo makeOwlv2Info() {
    ModelInfo info;
    info.input_shapes = {{1, 3, 960, 960}, {2, 8}, {2, 8}};
    info.input_formats = {"FORMAT_NCHW", "FORMAT_NCHW", "FORMAT_NCHW"};
    info.input_names = {"pixel_values", "input_ids", "attention_mask"};
    info.output_names = {"pred_boxes", "logits"};
    info.input_types = {CV_32F, CV_32S, CV_32S};
    return info;
}

} // namespace

TEST(OpenVocabDetectionTest, PreprocessProducesImageAndTextInputs) {
    TaskConfig cfg;
    cfg.text_prompts = {"cat", "dog"};
    cfg.tokenizer_vocab_json = R"({"<|endoftext|>":49407,"cat":10,"dog":11,"cat</w>":10,"dog</w>":11})";
    cfg.tokenizer_merges_text = "#version: 0.2\nc a\n";

    auto task = TaskFactory::createTaskInstance("owlv2", makeOwlv2Info(), cfg);
    cv::Mat image = cv::Mat::zeros(32, 32, CV_8UC3);
    const auto buffers = task->preprocess({image});

    ASSERT_EQ(buffers.size(), 3U);
    EXPECT_FALSE(buffers[0].empty());
    EXPECT_EQ(buffers[1].size(), static_cast<size_t>(2 * 8 * sizeof(int32_t)));
    EXPECT_EQ(buffers[2].size(), static_cast<size_t>(2 * 8 * sizeof(int32_t)));
}

TEST(OpenVocabDetectionTest, PostprocessReturnsOpenVocabDetectionResults) {
    TaskConfig cfg;
    cfg.text_prompts = {"cat", "dog"};

    auto task = TaskFactory::createTaskInstance("owlv2", makeOwlv2Info(), cfg);

    Tensor boxes(std::vector<TensorElement>{0.5f, 0.5f, 0.25f, 0.25f}, {1, 1, 4});
    Tensor logits(std::vector<TensorElement>{-3.0f, 4.0f}, {1, 1, 2});

    const auto results = task->postprocess(cv::Size(640, 480), {boxes, logits});
    ASSERT_EQ(results.size(), 1U);
    ASSERT_TRUE(std::holds_alternative<OpenVocabDetection>(results[0]));
    const auto& det = std::get<OpenVocabDetection>(results[0]);
    EXPECT_EQ(det.prompt_index, 1);
    EXPECT_EQ(det.label, "dog");
    EXPECT_GT(det.score, 0.9f);
}

TEST(OpenVocabDetectionTest, PostprocessUsesOutputNamesAndObjectness) {
    ModelInfo info = makeOwlv2Info();
    info.output_names = {"objectness_logits", "pred_boxes", "logits"};

    TaskConfig cfg;
    cfg.text_prompts = {"cat", "dog"};

    auto task = TaskFactory::createTaskInstance("owlv2", info, cfg);

    Tensor objectness(std::vector<TensorElement>{-10.0f}, {1, 1});
    Tensor boxes(std::vector<TensorElement>{0.5f, 0.5f, 0.25f, 0.25f}, {1, 1, 4});
    Tensor logits(std::vector<TensorElement>{-3.0f, 4.0f}, {1, 1, 2});

    const auto filtered = task->postprocess(cv::Size(640, 480), {objectness, boxes, logits});
    EXPECT_TRUE(filtered.empty());
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

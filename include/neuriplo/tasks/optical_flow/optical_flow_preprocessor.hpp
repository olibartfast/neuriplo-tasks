#pragma once

#include "neuriplo/tasks/core/preprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief RAFT optical flow preprocessor
 *
 * Preprocesses two consecutive frames for optical flow estimation.
 */
class RaftPreprocessor : public Preprocessor {
  public:
    explicit RaftPreprocessor(const Size& input_size = Size(960, 520));

    /**
     * @brief Preprocess a pair of frames for optical flow
     *
     * @param frame1 First frame
     * @param frame2 Second frame
     * @return Vector containing preprocessed data for both frames
     */
    [[nodiscard]] std::vector<std::vector<uint8_t>> preprocess_pair(const ImageView& frame1,
                                                                    const ImageView& frame2) const;
};

} // namespace neuriplo_tasks
